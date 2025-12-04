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


from utils.eval_utils import eval_func
from aq_agents import aq_agent


def aq_train_fn(X_Y_train_shards, X_test, Y_test, return_dict, results_dict):
	# Start the training process
	num_agents_per_time = int(gv.C * gv.k)
	simul_agents = gv.num_gpus * gv.max_agents_per_gpu
	simul_num = min(num_agents_per_time, simul_agents)
	agent_indices = np.arange(gv.k)


	t = 0
	eval_loss_list = []
	lr = 1e3
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
				print("Size of train X_batch, Y_batch:", X_batch.shape, Y_batch.shape)
				p = Process(target=aq_agent			 , args=(i, X_batch, Y_batch, train_offsets, t, gpu_id, return_dict, results_dict, X_test, Y_test, lr))
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
 	
		# Saving for the next update
		np.save(gv.dir_name + 'global_weights_t%s.npy' %
				(t + 1), global_weights)

		# Evaluate global weight
		p_eval = Process(target=eval_func, args=(
				X_test, Y_test, t + 1, return_dict), kwargs={'global_weights': global_weights})
		p_eval.start()
		p_eval.join()		

		eval_loss_list.append(return_dict['eval_loss'])
 	
		file_write_resultsdata(results_dict)

		t += 1

	return t