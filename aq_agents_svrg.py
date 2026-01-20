# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 13:40:33 2025

@author: Divya
"""

#########################
# Purpose: Mimics a benign agent in the federated learning setting and sets up the master agent 
########################
import warnings

warnings.filterwarnings("ignore")
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import logging
tf.get_logger().setLevel(logging.ERROR)

import numpy as np
tf.set_random_seed(777)
np.random.seed(777)

from utils.eval_utils import eval_minimal
import global_vars as gv
from  utils.air_quality_utils import airquality_model

import time
from utils.svrg_utils import svrg_client_learn_tf1_regression


def aq_agent_svrg(current_agent, x_batch, y_batch, round_idx, gpu_id, return_dict, results_dict, X_test, Y_test,y_scaler):
    tf.keras.backend.set_learning_phase(1)
    print('Agent %s on GPU %s' % (current_agent,gpu_id))
    # set environment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
	
    args = gv.init()
    tf.reset_default_graph()
    if args.k > 1:
        config = tf.ConfigProto(gpu_options=gv.gpu_options)
        config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = 0.05
        sess = tf.Session(config=config)
    elif args.k == 1:
        sess = tf.Session()
    else:
        return
    tf.compat.v1.keras.backend.set_session(sess)
    sess.run(tf.global_variables_initializer())


    shared_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % round_idx, allow_pickle=True)
    pre_theta = None
    
    agent_model = airquality_model()
    if pre_theta is not None:
        theta = pre_theta - gv.moving_rate * (pre_theta - shared_weights)
    else:
        theta = shared_weights
    agent_model.set_weights(theta)
    


# 
    # print('loaded shared weights')

    agent_drift = []
    start_offset = 0
    batch_size = len(x_batch)
    train_batchsize = gv.B
    num_steps = int(batch_size/train_batchsize)
    
    print("Num training steps: {}".format(num_steps))

    data_dim = gv.DATA_DIM
    num_classes = gv.NUM_CLASSES
    for step in range(num_steps):
        reset_now = (step==0)
        start = step * train_batchsize
        end   = min(start + train_batchsize, batch_size)

        X_batch = x_batch[start:end].astype(np.float32)
        Y_batch = y_batch[start:end].astype(np.float32).reshape(-1, 1)


        svrg_client_learn_tf1_regression(
                sess,
                agent_model,
                X_batch,
                Y_batch,
                data_dim=gv.DATA_DIM,
                lr=3e-3,
                buffer_size=2048,
                refresh_every=50,
                mu_batch_size=256,
                clip_norm=1.0,
                reset_state=reset_now,
                loss_type="mse",          # "mse" | "huber"
                huber_delta=1.0,
                sample_weights=None,      # None or np.ndarray shape [B] or [B,1]
             )

        if step % 50 == 0:
                 print(f"[agent {current_agent}] step {step}/{num_steps}", flush=True)

        

        # print('Agent %s, Step %s, Loss %s, Train step %s' % (i, step, loss_val, step_val))

    
    local_weights = agent_model.get_weights()
    # print("Local weights shape:", local_weights[0].shape, local_weights[0])
    local_delta = local_weights - shared_weights

    # eval_success, eval_loss = eval_minimal(X_test,Y_test,x, y, sess, prediction, loss)
    # print("Y test in agents:", Y_test.shape
  
    eval_success, eval_loss = eval_minimal(X_test, Y_test, local_weights, y_scaler=y_scaler)
    
    seed=None
    delayedclient = "false"
    max_delay_s = 0.1 # max .1 sec delay
    rng = np.random.default_rng(seed if seed is not None else (12345 + current_agent))
    if rng.random() < 0.3:    # delay only some clients
      delay = rng.exponential(scale=0.05)   # mean 0.05s
      delay = min(delay, max_delay_s)      # cap it
      time.sleep(float(delay))	
      delayedclient="true"
    
    client_str = "client_" + str(current_agent) + "_t_" + str(round_idx)
    driftstr = "-".join(agent_drift)
    delayedstr = delayedclient
    results_dict[client_str] = {"t": round_idx, "i": current_agent, "eval_success": eval_success, "eval_loss": eval_loss, "drift": driftstr, "delayed":delayedstr}  
    # print("Results dict:", results_dict[client_str])
    # print("Number of results_dict items - client:", len(results_dict))
 	
 	
    print('Agent {}: success {}, loss {}'.format(current_agent, eval_success, eval_loss))#  
    return_dict[str(current_agent)] = np.array(local_delta)
    return_dict["theta{}".format(current_agent)] = np.array(local_weights)
    return_dict[str(current_agent) + "_num_samples"] = batch_size
    return_dict[str(current_agent) + "_time"] = time.time()

    np.save(gv.dir_name + 'ben_delta_%s_t%s.npy' % (current_agent, round_idx), local_delta)

    return


