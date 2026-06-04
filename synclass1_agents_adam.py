# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 09:20:54 2026

@author: Divya
"""

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
tf.set_random_seed(99)
np.random.seed(99)

from utils.eval_utils import eval_minimal
import global_vars as gv
import time
from utils.synclass1_utils import synclass1_model

def synclass1_agent_adam(current_agent, x_batch, y_batch,round_idx, gpu_id, return_dict, results_dict, X_test, Y_test, lr=None):
    tf.keras.backend.set_learning_phase(1)
#--------------------------intialize-----------------------------------
    args = gv.init()
    if args.k > 1:
        config = tf.ConfigProto(gpu_options=gv.gpu_options)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
    elif args.k == 1:
        sess = tf.Session()
    else:
        return
    tf.compat.v1.keras.backend.set_session(sess)

    train_batchsize = gv.B
    if lr is None:
        lr = args.eta
    print('Agent %s on GPU %s' % (current_agent,gpu_id))
    # set environment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    shared_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % round_idx, allow_pickle=True)
    pre_theta = None
    agent_model = synclass1_model()
#---------------------------------training init------------------------------------------------------
   	
    if pre_theta is not None:
        theta = pre_theta - gv.moving_rate * (pre_theta - shared_weights)
    else:
        theta = shared_weights
    agent_model.set_weights(theta)
    
    x = tf.placeholder(shape=[None, gv.DATA_DIM], dtype=tf.float32, name="x")
    y = tf.placeholder(shape=[None],dtype=tf.int64, name="y")
    lr_var = tf.Variable(1e-1, trainable=False, name="lr")
    logits = agent_model(x)


    batch_size = len(x_batch)
    num_steps = int(batch_size/train_batchsize)   
    
#-------------------------------------optimizer---------------------------------------------------
    per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(per_example_loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr_var)
    global_step = tf.compat.v1.Variable(0, trainable=False, dtype=tf.int64, name="global_step")
    train_op = optimizer.minimize(loss, global_step=global_step)
    
    init_op = tf.compat.v1.global_variables_initializer()
    sess.run(init_op)
    start_offset = 0

    print("Num training steps: {}".format(num_steps))
#-----------------------------------------------------training ----------------------
    for step in range(num_steps):
        start_offset = start_offset
        end_offset = start_offset + train_batchsize
        if(end_offset>batch_size):
            end_offset = batch_size

        X_batch = x_batch[start_offset: end_offset]
        Y_batch = y_batch[start_offset: end_offset]
        feed = {x: X_batch, y: Y_batch, lr_var: lr}

        _, loss_val, step = sess.run([train_op, loss, global_step], feed_dict=feed)
        start_offset = end_offset

        # print('Agent %s, Step %s, Loss %s, Train step %s' % (i, step, loss_val, step_val))
    local_weights = agent_model.get_weights()
    # print("Local weights shape:", local_weights[0].shape, local_weights[0])
    local_delta = local_weights - shared_weights

    # eval_success, eval_loss = eval_minimal(X_test,Y_test,x, y, sess, prediction, loss)
    # print("Y test in agents:", Y_test.shape
  
    eval_success, eval_loss = eval_minimal(X_test, Y_test, local_weights)
    
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

    delayedstr = delayedclient
    results_dict[client_str] = {"t": round_idx, "i": current_agent, "eval_success": eval_success, "eval_loss": eval_loss, "drift": "", "delayed":delayedstr}  
    # print("Results dict:", results_dict[client_str])
    # print("Number of results_dict items - client:", len(results_dict))
 	
 	
    print('Agent {}: success {}, loss {}'.format(current_agent, eval_success, eval_loss))#  
    return_dict[str(current_agent)] = np.array(local_delta)
    return_dict["theta{}".format(current_agent)] = np.array(local_weights)
    return_dict[str(current_agent) + "_num_samples"] = batch_size
    return_dict[str(current_agent) + "_time"] = time.time()

    np.save(gv.dir_name + 'ben_delta_%s_t%s.npy' % (current_agent, round_idx), local_delta)


    return


