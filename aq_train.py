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

def get_round_slice(X, y, t, T):
    n = X.shape[0]
    s = (t * n) // T
    e = ((t + 1) * n) // T
    print(f"Getting sizes for rounds - total {n} start {s} and end {e}")
    return X[s:e], y[s:e]


def aq_train_fn(X_Y_train_shards, X_test, Y_test, y_scaler,return_dict, results_dict):
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
		 
	while t < gv.T:
		
		print('-----------------Training client in server round %s----------------' % t)
	

		lmbda = gv.C*(1-gv.C)
		probs = [gv.C + lmbda*ri for ri in r]
		probs_sum = sum(probs)
		probs = [elem/probs_sum for elem in probs]

		process_list = []
		curr_agents = np.random.choice(agent_indices, num_agents_per_time,
									   replace=False,p=probs)
		print('Set of agents chosen in this round: %s' % curr_agents)
		
	       
		k = 0
		agents_left = 1e4

		while k < num_agents_per_time:
			true_simul = min(simul_num, agents_left)
			print('Training %s agents' % true_simul)
			for l in range(true_simul):
				gpu_index = int(l / gv.max_agents_per_gpu)
				gpu_id = gv.gpu_ids[gpu_index]
				i = curr_agents[k]
				X_batch, Y_batch = X_Y_train_shards[i]  
                
				X_round, Y_round = get_round_slice(X_batch, Y_batch, t, gv.T)   
                
				print("Size of train X_batch, Y_batch:", X_round.shape, Y_round.shape)
				if(('adam' in gv.optimizer) or ('strobfl_learn' in gv.optimizer)):
				  p = Process(target=aq_agent, args=(i, X_round, Y_round,  t, gpu_id, return_dict, results_dict, X_test, Y_test, y_scaler))
				elif('strsaga' in gv.optimizer):
  				  p = Process(target=aq_agent_strsaga, args=(i, X_round, Y_round, t, gpu_id, return_dict, results_dict, X_test, Y_test, y_scaler))
				elif('svrg' in gv.optimizer):
 				  p = Process(target=aq_agent_svrg, args=(i, X_round, Y_round, t, gpu_id, return_dict, results_dict, X_test, Y_test, y_scaler))
				p.start()
				process_list.append(p)
				k += 1	
				 
			for item in process_list:
				item.join()
			agents_left = num_agents_per_time - k
			print('Agents left:%s' % agents_left)

		print('Joined all processes for time step %s' % t)

		global_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)
        

		if 'avg' in gv.gar:
 			print('Using standard mean aggregation')		            
 			for k in range(num_agents_per_time):
 				 global_weights += (1/num_agents_per_time) * return_dict[str(curr_agents[k])]
		elif 'strobfl' in gv.gar:
 			client_num_samples = np.array(
                    [return_dict[f"{cid}_num_samples"] for cid in curr_agents],
                    dtype=np.float64,
                  )
 			global_weights = aggregate_with_rbf_and_aging(
                 global_weights,
                 num_agents_per_time,
                 return_dict,
                 curr_agents,
                 client_num_samples,
                 gamma=1.0,
                 eps=1e-12,
                 age_lambda=1.0)              
		# Saving for the next update
		np.save(gv.dir_name + 'global_weights_t%s.npy' %
				(t + 1), global_weights)

		# Evaluate global weight
		p_eval = Process(target=eval_func, args=(
				X_test, Y_test, t + 1, return_dict, y_scaler), kwargs={'global_weights': global_weights})
		p_eval.start()
		p_eval.join()		

		eval_loss_list.append(return_dict['eval_loss'])
 	
		file_write_resultsdata(results_dict)

		t += 1

	return t