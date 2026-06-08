#########################
# Purpose: Main function to perform federated training
########################
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

from multiprocessing import Process, Manager
from utils.io_utils import file_write_resultsdata
import global_vars as gv
from agents import agent, master
from utils.eval_utils import eval_func
from utils.air_quality_utils import data_air_quality
from synclass1_train import synclass1_train_fn
from aq_train import aq_train_fn



def train_fn(X_train_shards, Y_train_shards, X_test, Y_test, return_dict, results_dict,
			 mal_data_X=None, mal_data_Y=None):
	# Start the training process
	num_agents_per_time = int(args.C * args.k)
	simul_agents = gv.num_gpus * gv.max_agents_per_gpu
	simul_num = min(num_agents_per_time, simul_agents)

	agent_indices = np.arange(args.k)
	print("In agent train, X_train_shard:", len(X_train_shards), X_train_shards[0].shape)
	print("In agent train, Y_train_shard:", len(Y_train_shards), Y_train_shards[0].shape)
	t = 0
	eval_loss_list = []
	lr = args.eta
	param_dict = dict()
	param_dict['offset'] = [0]
	param_dict['shape'] = []

	r = [1 for i in range(0,args.k)]
	if (args.dataset == 'uci-sensor'):
		agent_offsets = np.zeros((args.k,1))
		block_size = np.zeros((args.k,1))
		for ii in range(args.k):
		 shard_size = X_train_shards[ii].shape[0]
		 block_size[ii] = int(shard_size/int(args.T))
		 print("Block size:", block_size[ii])
		 
	while t < args.T:
		
	# while return_dict['eval_success'] < gv.max_acc and t < args.T:
		print('-----------------Training client in server round %s----------------' % t)
		
		lmbda = args.C*(1-args.C)
		probs = [args.C + lmbda*ri for ri in r]
		probs_sum = sum(probs)
		probs = [elem/probs_sum for elem in probs]

		process_list = []
		curr_agents = np.random.choice(agent_indices, num_agents_per_time,
									   replace=False,p=probs)
		print('Set of agents chosen: %s' % curr_agents)

		k = 0
		bsize = gv.BATCH_SIZE
		agents_left = 1e4

		while k < num_agents_per_time:
			true_simul = min(simul_num, agents_left)
			print('training %s agents' % true_simul)
			for l in range(true_simul):
				gpu_index = int(l / gv.max_agents_per_gpu)
				gpu_id = gv.gpu_ids[gpu_index]
				i = curr_agents[k]
				offset = agent_offsets[i] 
				print("Agent, offset indexes, training block size:", i, offset, offset + bsize, bsize)
				X_batch = X_train_shards[i][offset: (offset + bsize)]
				Y_batch = Y_train_shards[i][offset: (offset + bsize)]
				print("Size of train X_batch, Y_batch:", X_batch.shape, Y_batch.shape)
				agent_offsets[i] = agent_offsets[i] + bsize
				p = Process(target=agent, args=(i, X_batch,Y_batch, t, gpu_id, return_dict, results_dict, X_test, Y_test, lr))
				
				p.start()
				process_list.append(p)
				
				k += 1
			for item in process_list:
				item.join()
			agents_left = num_agents_per_time - k
			print('Agents left:%s' % agents_left)

		print('Joined all processes for time step %s' % t)

		global_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)
        

		if 'avg' in args.gar:
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
		
# 		print("Number of results_dict items - main :", len(results_dict))
# 		for k, v in results_dict.items():
# 			print(f"Results dict: {k}: {v}\n")
		
		file_write_resultsdata(results_dict)

		t += 1

	return t


def main(args):
    if args.train:
            p = Process(target=master)
            p.start()
            p.join()
    
            manager = Manager()
            return_dict = manager.dict()
            return_dict['eval_success'] = 0.0
            return_dict['eval_loss'] = 0.0    		
            results_dict = manager.dict()
    random_state = 100
    master_rng = np.random.default_rng(random_state)
    if(args.dataset == 'synthetic-class1'):
            _ = synclass1_train_fn(return_dict, results_dict, master_rng)
    elif (args.dataset == 'air-quality'):
            X_Y_train_shards, X_test, Y_test, y_scaler = data_air_quality()
            _ = aq_train_fn( X_Y_train_shards, X_test, Y_test, y_scaler, return_dict, results_dict,master_rng))

         			

if __name__ == "__main__":
	args = gv.init()
	tf.set_random_seed(777)
	np.random.seed(777)
	main(args)
    