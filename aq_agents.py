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
from  utils.air_quality_utils import airquality_model
import global_vars as gv

from customSGD import CustomRuleSGD, gradient_update_rule_factory

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gv.mem_frac)

PER_LABEL_STATS = {
    "sum": None,       # shape: [C, D]
    "count": None,     # shape: [C]
    "means": None      # shape: [C, D] (derived)
}

def aq_agent(i, x_batch, y_batch, train_offsets, t, gpu_id, return_dict, results_dict, X_test, Y_test, lr=None):
    
    tf.keras.backend.set_learning_phase(1)    
    tf.reset_default_graph()
    sess = tf.Session()

    args = gv.init()
    if lr is None:
        lr = args.eta
    print('Agent %s on GPU %s' % (i,gpu_id))
    # set environment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    shared_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)
    pre_theta = None
    
    agent_model = airquality_model()
    x = tf.placeholder(shape=(None, gv.DATA_DIM), dtype=tf.float32)
    y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="y")
    logits = agent_model(x)
    if pre_theta is not None:
        theta = pre_theta - gv.moving_rate * (pre_theta - shared_weights)
    else:
        theta = shared_weights
    agent_model.set_weights(theta)
    
    
    prediction = logits
    loss = tf.reduce_mean(tf.losses.mean_squared_error(y, logits))
  		
   	
    if args.optimizer == 'adam':
        lr=0.001
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr).minimize(loss)
    elif args.optimizer == 'strobfl_learn':
       lr=0.1
       alpha = 0.3
       alpha_var = tf.Variable(alpha, trainable=False, dtype=tf.float32, name="ema_alpha")
       ema_rule = gradient_update_rule_factory(alpha_var, name_prefix="grad_ema")
       optimizer = CustomRuleSGD(learning_rate=lr, update_rule=ema_rule)
       train_op = optimizer.minimize(loss)
  
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
# 

    # print('loaded shared weights')
 	

    b_start_offset = train_offsets[i]
    batch_size = len(x_batch)
    train_size = (batch_size-b_start_offset)/(args.T-t)
   	
    num_steps = int(train_size/args.B)
    start_offset = b_start_offset
 	
    for step in range(num_steps):
        offset = (start_offset + step * args.B) 
        X_batch = x_batch[offset: (offset + args.B)]
        Y_batch = y_batch[offset: (offset + args.B)]
        Y_batch = Y_batch.astype('float32').reshape(-1, 1) 
        _, loss_val = sess.run([optimizer, loss], feed_dict={x: X_batch, y: Y_batch})	
        start_offset = offset
        # print('Agent %s, Step %s, Loss %s, Train step %s' % (i, step, loss_val, step_val))
    b_new_offset = b_start_offset + train_size
    train_offsets[i] = b_new_offset
    local_weights = agent_model.get_weights()
    # print("Local weights shape:", local_weights[0].shape, local_weights[0])
    local_delta = local_weights - shared_weights

    # eval_success, eval_loss = eval_minimal(X_test,Y_test,x, y, sess, prediction, loss)
    # print("Y test in agents:", Y_test.shape
  
    eval_success, eval_loss = eval_minimal(X_test, Y_test, local_weights)
 	
    client_str = "client_" + str(i) + "_t_" + str(t)
    results_dict[client_str] = {"t": t, "i": i, "eval_success": eval_success, "eval_loss": eval_loss}  
    # print("Results dict:", results_dict[client_str])
    # print("Number of results_dict items - client:", len(results_dict))
 	
 	
    print('Agent {}: success {}, loss {}'.format(i, eval_success, eval_loss))#  
    return_dict[str(i)] = np.array(local_delta)
    return_dict["theta{}".format(i)] = np.array(local_weights)

    np.save(gv.dir_name + 'ben_delta_%s_t%s.npy' % (i, t), local_delta)



    return


def aq_master():
    tf.keras.backend.set_learning_phase(1)
    print('Initializing server models')
    config = tf.ConfigProto(gpu_options=gv.gpu_options)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)
    sess.run(tf.global_variables_initializer())
	
    global_model = airquality_model()

    global_weights_np = global_model.get_weights()
    np.save(gv.dir_name + 'global_weights_t0.npy', global_weights_np)
    print("[server] save global weights t0")
    return
