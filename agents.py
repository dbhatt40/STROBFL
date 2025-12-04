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
from utils.census_utils import census_model_1
from utils.gas_sensor_utils import uci_sensor_model, write_rbf_history
from utils.air_quality_utils import airquality_model
from utils.eval_utils import eval_minimal
from utils.synclass1_utils import synclass1_model

import global_vars as gv
import utils.streaming_utils as su
from customSGD import CustomRuleSGD
import strobfl_learn as SFL
from collections import deque, defaultdict
from sklearn.metrics import f1_score



# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gv.mem_frac)

PER_LABEL_STATS = {
    "sum": None,       # shape: [C, D]
    "count": None,     # shape: [C]
    "means": None      # shape: [C, D] (derived)
}

def agent(i, X_shard, Y_shard, t, gpu_id, return_dict, results_dict, X_test, Y_test, lr=None):
    tf.keras.backend.set_learning_phase(1)

    args = gv.init()
    if lr is None:
        lr = args.eta
    print('Agent %s on GPU %s' % (i,gpu_id))
    # set environment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    shared_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)
    shard_size = len(X_shard)

    pre_theta = None


    num_steps = 0		
    if args.steps is not None:
        num_steps = int(args.steps)
    else:
        num_steps = int(args.E * shard_size / args.B)


    # with tf.device('/gpu:'+str(gpu_id)):
    if (args.dataset == 'census'):
        x = tf.placeholder(shape=(None, gv.DATA_DIM), dtype=tf.float32)
        y = tf.placeholder(dtype=tf.int64)
    elif (args.dataset == 'uci-sensor'):
	        x = tf.placeholder(shape=(None, gv.DATA_DIM), dtype=tf.float32)
	        y = tf.placeholder(shape=(None,gv.NUM_CLASSES), dtype=tf.int64)
    # print("x shape & y shape:", x.shape, y.shape)
	
	
    if args.dataset == 'census':
        agent_model = census_model_1()
    elif args.dataset == 'uci-sensor':
        agent_model = uci_sensor_model()
    else:
        return

    logits = agent_model(x)
    # print("Logits:", logits)
    # print("y labels:", y)
    # print("logits & y shape:", logits.shape, y.shape)
    if (args.dataset == 'census'):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits))
    elif (args.dataset == 'uci-sensor') and (args.optimizer != 'strobfl_learn'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=logits))
		
        # loss = tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=y, logits=logits)
		
    # prediction = tf.nn.softmax(logits)
	
    lr=0.001
    alpha = 0.3

    if args.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr).minimize(loss)
    elif args.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=lr).minimize(loss)
    elif args.optimizer == 'strsgd':
        optimizer = CustomRuleSGD(
            learning_rate=lr).minimize(loss)
    elif (args.dataset == 'uci-sensor') and (args.optimizer == 'strobfl_learn'):
        per_example_loss = tf.nn.softmax_cross_entropy_with_logits(
                                  labels=y, logits=logits)  # shape [B]
        loss = per_example_loss
        # class_ids = tf.argmax(y, axis=1, output_type=tf.int32)  # shape [B]

        # loss_sum_per_class = tf.math.unsorted_segment_sum(per_example_loss, class_ids, gv.NUM_CLASSES)
        # counts_per_class  = tf.math.unsorted_segment_sum(tf.ones_like(per_example_loss), class_ids, gv.NUM_CLASSES)
        # per_class_loss    = tf.math.divide_no_nan(loss_sum_per_class, counts_per_class)
        # loss = tf.reduce_mean(per_class_loss)  
		
        alpha_var = tf.Variable(
                   alpha,                 # initial α value
                   trainable=False,
                   dtype=tf.float32,
                   name="alpha_var"
            )

        sfl_update_rule = SFL.gradient_update_rule_factory1(alpha_var, name_prefix="grad_ema")
        global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='global_step')
        optimizer = SFL.Strobfl_learn(learning_rate=lr, update_rule=sfl_update_rule).minimize(loss, global_step=global_step)
    elif (args.dataset == 'census') and (args.optimizer == 'strobfl_learn'):
        per_example_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=y, logits=logits))
        loss = per_example_loss
       		
        alpha_var = tf.Variable(
                   alpha,                 # initial α value
                   trainable=False,
                   dtype=tf.float32,
                   name="alpha_var"
            )

        sfl_update_rule = SFL.gradient_update_rule_factory1(alpha_var, name_prefix="grad_ema")
        global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='global_step')
        optimizer = SFL.Strobfl_learn(learning_rate=lr, update_rule=sfl_update_rule).minimize(loss, global_step=global_step)

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
    if pre_theta is not None:
        theta = pre_theta - gv.moving_rate * (pre_theta - shared_weights)
    else:
        theta = shared_weights
    agent_model.set_weights(theta)
    # print('loaded shared weights')
	
    start_offset = 0
	
    if args.steps is not None:
        start_offset = (t * args.B * args.steps) % (shard_size - args.B)
		
    print("Number of steps, shard size and batch size:", num_steps, shard_size, args.B)

    K = 3  # window size
    loss_win    = deque(maxlen=K)
    f1m_win     = deque(maxlen=K)  # macro-F1
    f1mi_win    = deque(maxlen=K)  # micro-F1 (optional)
    
    C = gv.NUM_CLASSES
    D = gv.DATA_DIM
    SFL.init_stats(C, D, PER_LABEL_STATS)
    sigma = 0.5
    prev_means = None
    rbf_drift_history = []
    for step in range(num_steps):
      
        offset = (start_offset + step * args.B) % (shard_size - args.B)
        X_batch = X_shard[offset: (offset + args.B)]
        Y_batch = Y_shard[offset: (offset + args.B)]
     
		
        if args.dataset == 'uci-sensor':
          Y_batch_uncat =Y_batch
        else:
          Y_batch_uncat = np.argmax(Y_batch, axis=1)
        
        counts_s, means_s = SFL.update_per_label_stats_batch(X_batch, Y_batch, C, PER_LABEL_STATS)

        if prev_means is not None:
           drift_vec, drift_mean = SFL.rbf_drift(prev_means, means_s, sigma)
           rbf_drift_history.append(drift_mean)
           # print("Drift mean shape:", drift_mean)
           # print("Step, RBF Drift:", step, drift_mean)
           # for c,d in enumerate(drift_vec):
           #     print("Step:C:D:", step, c,d)
        prev_means = means_s.copy()
		
        if(args.optimizer != 'strobfl_learn' or args.dataset != 'uci-sensor'):
          _, loss_val = sess.run([optimizer, loss], feed_dict={x: X_batch, y: Y_batch_uncat})			
        else:			
          _, loss_val, step_val, logits_val = sess.run([optimizer, loss, global_step, logits], feed_dict={x: X_batch, y: Y_batch_uncat})	
        

          y_true = np.argmax(Y_batch_uncat, axis=1) if Y_batch_uncat.ndim == 2 else Y_batch_uncat
          y_pred = np.argmax(logits_val, axis=1)

          f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
          f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)

            # 3) update rolling windows
          loss_val = float(np.mean(loss_val))
          loss_win.append(loss_val)
          f1m_win.append(f1_macro)
          f1mi_win.append(f1_micro)
          alpha = SFL.detect_concept_drift(alpha,loss_win, f1m_win, f1mi_win)		  
          sess.run(alpha_var.assign(alpha))

		  # print('Agent %s, Step %s, Loss %s, Train step %s' % (i, step, loss_val, step_val))
    write_rbf_history(rbf_drift_history)
    local_weights = agent_model.get_weights()
    # print("Local weights shape:", local_weights[0].shape, local_weights[0])
    local_delta = local_weights - shared_weights

    # eval_success, eval_loss = eval_minimal(X_test,Y_test,x, y, sess, prediction, loss)
    # print("Y test in agents:", Y_test.shape)
  
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


def master():
    tf.keras.backend.set_learning_phase(1)

    args = gv.init()
    print('Initializing master model')
    config = tf.ConfigProto(gpu_options=gv.gpu_options)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)
    sess.run(tf.global_variables_initializer())

    if args.dataset == 'air-quality':
        global_model = airquality_model()
    elif (args.dataset == 'synthetic-class1'):
       global_model =  synclass1_model()
        
    global_weights_np = global_model.get_weights()
    np.save(gv.dir_name + 'global_weights_t0.npy', global_weights_np)
    print("[server] save global weights t0")
    return
