# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 13:21:49 2025

@author: Divya
"""
import warnings

warnings.filterwarnings("ignore")
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import logging
tf.get_logger().setLevel(logging.ERROR)

from multiprocessing import Process
from utils.io_utils import file_write_resultsdata
import global_vars as gv
from aq_agents_strsaga import aq_agent_strsaga


from utils.eval_utils import eval_func
from aq_agents import aq_agent
from utils.synclass1_utils import aggregate_with_rbf_and_aging
from aq_agents_svrg import aq_agent_svrg
from aq_agent_fedprox import aq_agent_fedprox

def get_round_slice(X, y, t, T):
    n = X.shape[0]
    s = (t * n) // T
    e = ((t + 1) * n) // T
    print(f"Getting sizes for rounds - total {n} start {s} and end {e}")
    return X[s:e], y[s:e]


def aq_train_fn(X_Y_train_shards, X_test, Y_test, y_scaler,return_dict, results_dict, master_rng):
    # Start the training process
    num_agents_per_time = int(gv.C * gv.k)
    simul_agents = gv.num_gpus * gv.max_agents_per_gpu
    simul_num = min(num_agents_per_time, simul_agents)
    agent_indices = np.arange(gv.k)


    t = 0
    eval_loss_list = []
    lr = 0.1
    param_dict = dict()
    param_dict['offset'] = [0]
    param_dict['shape'] = []

    r = [1 for i in range(0,gv.k)]
    
    NUM_AGENTS_ROUND = gv.k
    train_offsets = np.zeros(NUM_AGENTS_ROUND, dtype=np.int32)
    round_idx = 0
         
    while round_idx < gv.T:
        
        print('-----------------Training client in server round %s----------------' % round_idx)
    

        lmbda = gv.C*(1-gv.C)
        probs = [gv.C + lmbda*ri for ri in r]
        probs_sum = sum(probs)
        probs = [elem/probs_sum for elem in probs]

        process_list = []
        curr_agents = np.random.choice(agent_indices, num_agents_per_time,
                                       replace=False,p=probs)
        print('Set of agents chosen in this round: %s' % curr_agents)
        
        client_seed = int(master_rng.integers(1_000_000_000))
        k = 0
        agents_left = 1e4

        while k < num_agents_per_time:
            true_simul = min(simul_num, agents_left)
            print('Training %s agents' % true_simul)
            for l in range(true_simul):
                gpu_index = int(l / gv.max_agents_per_gpu)
                gpu_id = gv.gpu_ids[gpu_index]
                current_agent = curr_agents[k]
                X_batch, Y_batch = X_Y_train_shards[current_agent]  
                
                X_round, Y_round = get_round_slice(X_batch, Y_batch, round_idx, gv.T)   
                
                print("Size of train X_batch, Y_batch:", X_round.shape, Y_round.shape)
                if(('adam' in gv.optimizer) or ('strobfl_learn' in gv.optimizer)):
                  p = Process(target=aq_agent, args=(current_agent, X_round, Y_round, round_idx, gpu_id, return_dict, results_dict, X_test, Y_test, y_scaler, client_seed))
                elif('strsaga' in gv.optimizer):
                    p = Process(target=aq_agent_strsaga, args=(current_agent, X_round, Y_round,  round_idx, gpu_id, return_dict, results_dict, X_test, Y_test, y_scaler))
                elif('svrg' in gv.optimizer):
                   p = Process(target=aq_agent_svrg, args=(current_agent, X_round, Y_round, round_idx,  gpu_id, return_dict, results_dict, X_test, Y_test, y_scaler, client_seed))
                elif('fedprox' in gv.optimizer):
                   p = Process(target=aq_agent_fedprox, args=(current_agent, X_round, Y_round,  round_idx,  gpu_id, return_dict, results_dict, X_test, Y_test, y_scaler, client_seed))             
                p.start()
                process_list.append(p)
                k += 1    
                 
            for item in process_list:
                item.join()
            agents_left = num_agents_per_time - k
            print('Agents left:%s' % agents_left)

        print('Joined all processes for time step %s' % round_idx)

        global_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % round_idx, allow_pickle=True)
        
#-------------------------------------Aggregation
        if 'avg' in gv.gar:
          arrived_updates = [
          k for k, v in return_dict.items()
             if k.endswith("_round_arrived") and v == round_idx
            ]
          current_updates = []

          for arrival_key in arrived_updates:
                  prefix = arrival_key.replace("_round_arrived", "")

                  created_round = return_dict[f"{prefix}_round_created"]
                  arrived_round = return_dict[f"{prefix}_round_arrived"]

                  if created_round == round_idx and arrived_round == round_idx:
                        current_updates.append(arrival_key)
          total_samples = 0
          agg_delta = [np.zeros_like(w) for w in global_weights]

          for arrival_key in current_updates:
                 prefix = arrival_key.replace("_round_arrived", "")

                 update = return_dict[f"{prefix}_weights"]      # local_delta
                 num_samples = return_dict[f"{prefix}_num_samples"]

                 total_samples += num_samples

                 for layer_idx in range(len(global_weights)):
                       agg_delta[layer_idx] += num_samples * update[layer_idx]

                 print(f"FedAvg no-delay aggregating {arrival_key}")

          if total_samples > 0:
               for layer_idx in range(len(global_weights)):
                     global_weights[layer_idx] += agg_delta[layer_idx] / total_samples
          else:
               print(f"No no-delay FedAvg updates at round {round_idx}")
               
        elif 'strobfl' in gv.gar:
              global_weights= aggregate_with_rbf_and_aging(
                 round_idx,
                 global_weights,
                 num_agents_per_time,
                 return_dict,                 
                 curr_agents,
                 gamma=1.0,
                 eps=1e-12,
                 age_lambda=0.5)    

        # Saving for the next update
        np.save(gv.dir_name + 'global_weights_t%s.npy' %
                (round_idx + 1), global_weights)

        # Evaluate global weight
        p_eval = Process(target=eval_func, args=(
                X_test, Y_test, round_idx + 1, return_dict, y_scaler), kwargs={'global_weights': global_weights})
        p_eval.start()
        p_eval.join()        

        eval_loss_list.append(return_dict['eval_loss'])
     
        file_write_resultsdata(results_dict)

        round_idx += 1

    return round_idx