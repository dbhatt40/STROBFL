#########################
# Purpose: Main function to perform federated training and all model poisoning attacks
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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans
from utils.io_utils import data_setup, mal_data_setup
import global_vars as gv
from agents import agent, master
from utils.eval_utils import eval_func, eval_minimal
from malicious_agent import mal_agent
from utils.dist_utils import collate_weights, model_shape_size
from math import log, inf


def train_fn(X_train_shards, Y_train_shards, X_test, Y_test, return_dict,
			 mal_data_X=None, mal_data_Y=None):
	# Start the training process
	num_agents_per_time = int(args.C * args.k)
	simul_agents = gv.num_gpus * gv.max_agents_per_gpu
	simul_num = min(num_agents_per_time, simul_agents)
	alpha_i = 1.0 / args.k
	agent_indices = np.arange(args.k)
	if args.mal:
		mal_agent_index = gv.mal_agent_index

	unupated_frac = (args.k - num_agents_per_time) / float(args.k)
	t = 0
	mal_visible = []
	eval_loss_list = []
	loss_track_list = []
	lr = args.eta
	loss_count = 0
	E = None
	beta = 0.5
	param_dict = dict()
	param_dict['offset'] = [0]
	param_dict['shape'] = []

	G = [None for i in range(0,args.k)]
	r = [1 for i in range(0,args.k)]
	Delta = 0.1

	while t < args.T:
	# while return_dict['eval_success'] < gv.max_acc and t < args.T:
		print('Time step %s' % t)
		
		lmbda = args.C*(1-args.C)
		probs = [args.C + lmbda*ri for ri in r]
		probs_sum = sum(probs)
		probs = [elem/probs_sum for elem in probs]

		process_list = []
		mal_active = 0
		curr_agents = np.random.choice(agent_indices, num_agents_per_time,
									   replace=False,p=probs)
		print('Set of agents chosen: %s' % curr_agents)

		k = 0
		agents_left = 1e4
		while k < num_agents_per_time:
			true_simul = min(simul_num, agents_left)
			print('training %s agents' % true_simul)
			for l in range(true_simul):
				gpu_index = int(l / gv.max_agents_per_gpu)
				gpu_id = gv.gpu_ids[gpu_index]
				i = curr_agents[k]
				if args.mal is False or i < mal_agent_index:
					p = Process(target=agent, args=(i, X_train_shards[i],
													Y_train_shards[i], t, gpu_id, return_dict, X_test, Y_test, lr))
				elif args.mal is True and i >= mal_agent_index:
					p = Process(target=mal_agent, args=(i, X_train_shards[i],
														Y_train_shards[i], mal_data_X, mal_data_Y, t,
														gpu_id, return_dict, mal_visible, X_test, Y_test))
					mal_active = 1

				p.start()
				process_list.append(p)
				k += 1
			for item in process_list:
				item.join()
			agents_left = num_agents_per_time - k
			print('Agents left:%s' % agents_left)

		if mal_active == 1:
			mal_visible.append(t)

		print('Joined all processes for time step %s' % t)
		global_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)
        
        
		if 'avg' in args.gar:
			print('Using standard mean aggregation')
			if args.mal:
				count = 0
				for k in range(num_agents_per_time):
					if curr_agents[k] != mal_agent_index:
						if count == 0:
							ben_delta = alpha_i * return_dict[str(curr_agents[k])]
							np.save(gv.dir_name + 'ben_delta_sample%s.npy' % t, return_dict[str(curr_agents[k])])
							count += 1
						else:
							ben_delta += alpha_i * return_dict[str(curr_agents[k])]

				np.save(gv.dir_name + 'ben_delta_t%s.npy' % t, ben_delta)
				global_weights += alpha_i * return_dict[str(mal_agent_index)]
				global_weights += ben_delta
			else:
				for k in range(num_agents_per_time):
					global_weights += alpha_i * return_dict[str(curr_agents[k])]
		
		# Saving for the next update
		np.save(gv.dir_name + 'global_weights_t%s.npy' %
				(t + 1), global_weights)

		# Evaluate global weight
		if args.mal:
			p_eval = Process(target=eval_func, args=(
				X_test, Y_test, t + 1, return_dict, mal_data_X, mal_data_Y), kwargs={'global_weights': global_weights})
		else:
			p_eval = Process(target=eval_func, args=(
				X_test, Y_test, t + 1, return_dict), kwargs={'global_weights': global_weights})
		p_eval.start()
		p_eval.join()

		eval_loss_list.append(return_dict['eval_loss'])

		t += 1

	return t


def main(args):
	X_train, Y_train, X_test, Y_test, Y_test_uncat = data_setup()
	
	keys = []
	for i in range(Y_train.shape[1]):
		keys.append(Y_train[:,i])
	keys = tuple(keys)
	
	sort_indices = np.lexsort(keys)
	
	Y_train = Y_train[sort_indices]
	X_train = X_train[sort_indices]
	
	num_slices = round(((len(X_train)-args.k)*args.iid+args.k) / args.k) * args.k
	
	slices_per_client = round(num_slices/args.k)
	
	X_slices = np.array_split(X_train, num_slices)
	Y_slices = np.array_split(Y_train, num_slices)
	
	slice_indices = np.random.choice(
		num_slices, num_slices, replace=False)
	
	X_train_shards = []
	Y_train_shards = []
	for i in range(0,num_slices,slices_per_client):
		idxs = slice_indices[i:i+slices_per_client]
		X_train_shards.append(np.concatenate([X_slices[slice_idx] for slice_idx in idxs]))
		Y_train_shards.append(np.concatenate([Y_slices[slice_idx] for slice_idx in idxs]))
	
	
	if args.mal:
		# Load malicious data
		mal_data_X, mal_data_Y, true_labels = mal_data_setup(X_test, Y_test, Y_test_uncat)

	if args.train:
		p = Process(target=master)
		p.start()
		p.join()

		manager = Manager()
		return_dict = manager.dict()
		return_dict['eval_success'] = 0.0
		return_dict['eval_loss'] = 0.0

		if args.mal:
			return_dict['mal_suc_count'] = 0
			t_final = train_fn(X_train_shards, Y_train_shards, X_test, Y_test_uncat,
							   return_dict, mal_data_X, mal_data_Y)
			print('Malicious agent succeeded in %s of %s iterations' %
				  (return_dict['mal_suc_count'], t_final * args.mal_num))
		else:
			_ = train_fn(X_train_shards, Y_train_shards, X_test, Y_test_uncat,
						 return_dict)
	else:
		manager = Manager()
		return_dict = manager.dict()
		return_dict['eval_success'] = 0.0
		return_dict['eval_loss'] = 0.0
		if args.mal:
			return_dict['mal_suc_count'] = 0
		for t in range(args.T):
			if not os.path.exists(gv.dir_name + 'global_weights_t%s.npy' % t):
				print('No directory found for iteration %s' % t)
				break
			if args.mal:
				p_eval = Process(target=eval_func, args=(
					X_test, Y_test_uncat, t, return_dict, mal_data_X, mal_data_Y))
			else:
				p_eval = Process(target=eval_func, args=(
					X_test, Y_test_uncat, t, return_dict))

			p_eval.start()
			p_eval.join()

		if args.mal:
			print('Malicious agent succeeded in %s of %s iterations' %
				  (return_dict['mal_suc_count'], (t - 1) * args.mal_num))


if __name__ == "__main__":
	args = gv.init()
	tf.set_random_seed(777)
	np.random.seed(777)
	main(args)
    