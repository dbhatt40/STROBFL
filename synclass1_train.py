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
from utils.eval_utils import eval_func
from utils.synclass1_utils import federated_mixed_drift_stream_with_queues, aggregate_with_rbf_and_aging, aggregate_with_sw_fedavg
from synclass1_agents import synclass1_agent
from synclass1_agents_adam import synclass1_agent_adam
from synclass1_agents_strsaga import synclass1_agent_strsaga
from synclass1_agents_svrg import synclass1_agent_svrg
from synclass1_agents_fedprox import synclass1_agent_fedprox
from synclass1_agents_cdafed import synclass1_agent_cdafed
import time


def synclass1_train_fn(return_dict, results_dict):
	# Start the training process

    T = gv.T
    C = gv.C
    k = gv.k
    total_clients = int( k*C)

    num_agents_per_time = int(C*k)
    simul_agents = gv.num_gpus * gv.max_agents_per_gpu
    simul_num = min(num_agents_per_time, simul_agents)
    agent_indices = np.arange(k)
	
    round_idx = 0
    eval_loss_list = []
    lr = 1e3
    param_dict = dict()
    param_dict['offset'] = [0]
    param_dict['shape'] = []

    r = [1 for i in range(0,k)]
	
    print('number drifted:{},driftmode:{},arrivalrate:{},imbalance:{},training_batch:{}'.format(gv.ndrift, gv.dmode, gv.arate, gv.ifactor, gv.B))
    ndrift = gv.ndrift
    dmode = gv.dmode
    arate = gv.arate
    ifactor = gv.ifactor
    
    gen = federated_mixed_drift_stream_with_queues(
    num_rounds=T,
    num_clients=total_clients,
    batch_size=gv.WINDOW_SIZE,
    num_drifted_clients=ndrift,
    drift_clients_mode=dmode,  # or "shared"
    arrival_rate=arate,
    test_batch_size=500,
    noise_std=0.05,
    imbalance_factor=ifactor,
    samples_per_cycle=80000,
    random_state=42,
    queue_maxlen=2000
    )
    


    for round_idx, client_batches, global_test_batch in gen:
        print("Round:", round_idx)
        


        X_test, y_test, t_test = global_test_batch
        # print("  Test batch shape:", X_test.shape, y_test.shape, t_test.shape)
        print('-----------------Training client in server round %s----------------' % round_idx)

        lmbda = C*(1-C)
        probs = [C + lmbda*ri for ri in r]
        probs_sum = sum(probs)
        probs = [elem/probs_sum for elem in probs]

        process_list = []
        curr_agents = np.random.choice(agent_indices, num_agents_per_time,
									   replace=False,p=probs)
        print('Set of agents chosen in this round: %s' % curr_agents)
		
        agents_left = 1e4
        activeclient = 0
        
        initial_global_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % round_idx, allow_pickle=True)
        
        start = time.perf_counter()

    # after this many steps without drift -> go back to stable
        lr=None
        while activeclient < num_agents_per_time:
            true_simul = min(simul_num, agents_left)
            print('Training %s agents' % true_simul)
            for l in range(true_simul):
                gpu_index = int(l / gv.max_agents_per_gpu)
                gpu_id = gv.gpu_ids[gpu_index]
                i = curr_agents[activeclient]
                print('Client training %s agent' % i)
                X_batch, y_batch, t_batch= client_batches[activeclient]  
             
                print("Size of train X_batch, Y_batch:", X_batch.shape, y_batch.shape)
                if('adam' in gv.optimizer):
                  p = Process(target=synclass1_agent_adam, args=(i, X_batch, y_batch, round_idx, gpu_id, return_dict, results_dict, X_test, y_test, lr))
                elif('strobfl_learn' in gv.optimizer):
                  p = Process(target=synclass1_agent, args=(i, X_batch, y_batch, round_idx, gpu_id, return_dict, results_dict, X_test, y_test, lr))
                elif('strsaga' in gv.optimizer):
                  p = Process(target=synclass1_agent_strsaga, args=(i, X_batch, y_batch, round_idx, gpu_id, return_dict, results_dict, X_test, y_test, lr))
                elif('svrg' in gv.optimizer):
                  p = Process(target=synclass1_agent_svrg, args=(i, X_batch, y_batch, round_idx, gpu_id, return_dict, results_dict, X_test, y_test, lr))
                elif('fedprox' in gv.optimizer):
                    p = Process(target=synclass1_agent_fedprox, args=(i, X_batch, y_batch, round_idx, gpu_id, return_dict, results_dict, X_test, y_test, lr))
                elif('cdafed' in gv.optimizer):
                     p = Process(target=synclass1_agent_cdafed, args=(i, X_batch, y_batch, round_idx, gpu_id, return_dict, results_dict, X_test, y_test, lr))
                
                p.start()
                process_list.append(p)
                activeclient += 1	
 				 
            for item in process_list:
                item.join()
            agents_left = num_agents_per_time - activeclient
            print('Agents left:%s' % agents_left)

        print('Joined all processes for time step %s' % round_idx)

        global_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % round_idx, allow_pickle=True)

        
        if 'avg' in gv.gar:
            n_total = sum(return_dict[str(cid) + "_num_samples"] for cid in curr_agents)
            for client_agents in range(num_agents_per_time):
                cid = curr_agents[client_agents]  
                p_i = return_dict[str(cid) + "_num_samples"]/n_total
                global_weights += p_i* return_dict[str(curr_agents[client_agents])]
        elif 'strobfl' in gv.gar:
              client_num_samples = np.array(
                    [return_dict[f"{cid}_num_samples"] for cid in curr_agents],
                    dtype=np.float64,
                  )
              global_weights= aggregate_with_rbf_and_aging(
                 global_weights,
                 num_agents_per_time,
                 return_dict,
                 curr_agents,
                 client_num_samples,
                 gamma=1.0,
                 eps=1e-12,
                 age_lambda=0.6)              
        elif 'sw-fedavg' in gv.gar:
              client_num_samples = np.array(
                    [return_dict[f"{cid}_num_samples"] for cid in curr_agents],
                    dtype=np.float64,
                  )
              global_weights= aggregate_with_sw_fedavg(
                 global_weights,
                 num_agents_per_time,
                 return_dict,
                 curr_agents,
                 client_num_samples,
                 gamma=1.0,
                 eps=1e-12,
                 age_lambda=1.0)
              
        elif 'fednova' in gv.gar:
            update_sum = np.zeros_like(initial_global_weights)
            n_total = sum(return_dict[str(cid) + "_num_samples"] for cid in curr_agents)
            for j in range(num_agents_per_time):
                cid = curr_agents[j]  
                p_i = return_dict[str(cid) + "_num_samples"]/n_total
                client_weights = return_dict[str(cid)]        
                delta_i = client_weights        
                a_i = return_dict[str(cid) + "_lrsum"]    
                normalized_update = delta_i / (a_i + 1e-12)        
                update_sum += p_i * normalized_update

            global_weights = initial_global_weights + update_sum
              
        end = time.perf_counter()
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print(f"Elapsed time: {end - start:.6f} seconds")
        
        
		# Saving for the next update
        np.save(gv.dir_name + 'global_weights_t%s.npy' %
				(round_idx + 1), global_weights)

		# Evaluate global weight
        p_eval = Process(target=eval_func, args=(
				X_test, y_test, round_idx + 1, return_dict), kwargs={'global_weights': global_weights})
        p_eval.start()
        p_eval.join()		

        eval_loss_list.append(return_dict['eval_loss'])
 	
        file_write_resultsdata(results_dict)

    return round_idx