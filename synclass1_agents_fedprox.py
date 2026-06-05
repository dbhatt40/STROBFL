# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 19:07:53 2026

@author: Divya
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 13:40:33 2025

@author: Divya
"""

#########################
# Purpose: Mimics a benign agent in the federated learning setting and sets up the master agent
# Rewritten for FedProx local training
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
from utils.synclass1_utils import synclass1_model
import math
import time

# ---------------- FedProx hyperparameters ----------------
LR = 0.1
MU = 0.01          # proximal coefficient; tune this
NUM_CLASSES = 4


def compute_sample_weights(y_batch, class_weight_mode="balanced"):
    B = len(y_batch)

    if B == 0:
        return np.array([], dtype=np.float32)

    if class_weight_mode == "none":
        return np.ones(B, dtype=np.float32)

    classes, counts = np.unique(y_batch, return_counts=True)

    if class_weight_mode == "balanced":
        # inverse-frequency style weighting
        weights = B / (len(classes) * counts)
    else:
        weights = np.ones_like(counts, dtype=np.float32)

    class_to_w = dict(zip(classes, weights))
    return np.array([class_to_w[y] for y in y_batch], dtype=np.float32)


# -------------------------- initialize -----------------------------------

def synclass1_agent_fedprox(
    current_agent,
    x_batch,
    y_batch,
    round_idx,
    gpu_id,
    return_dict,
    results_dict,
    X_test,
    Y_test,
    client_seed,
    lr=LR,
    mu=MU
):
    tf.keras.backend.set_learning_phase(1)

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
    CURRENT_AGENT = current_agent
    train_batchsize = gv.B

    if lr is None:
        lr=LR
        # lr = args.eta if hasattr(args, "eta") else LR
        

    print('Agent %s on GPU %s' % (CURRENT_AGENT, gpu_id))

    # set environment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # global model received from server
    shared_weights = np.load(
        gv.dir_name + 'global_weights_t%s.npy' % round_idx,
        allow_pickle=True
    )

    pre_theta = None
    agent_model = synclass1_model()

    if pre_theta is not None:
        theta = pre_theta - gv.moving_rate * (pre_theta - shared_weights)
    else:
        theta = shared_weights




    batch_size = len(x_batch)
    num_steps = int(math.ceil(batch_size / train_batchsize))

    # ---------------------------------------------------------------------
    # Build graph
    # ---------------------------------------------------------------------
    x = tf.placeholder(shape=[None, gv.DATA_DIM], dtype=tf.float32, name="x")
    y = tf.placeholder(shape=[None], dtype=tf.int32, name="y")
    sample_w = tf.placeholder(tf.float32, shape=[None], name="sample_w")

    global_step = tf.Variable(0, trainable=False, name="global_step")
    lr_var = tf.Variable(lr, trainable=False, dtype=tf.float32, name="lr")

    logits = agent_model(x, training=True)

    per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y,
        logits=logits
    )

    eps = 1e-8
    weighted_data_loss = tf.reduce_sum(sample_w * per_example_loss) / (
        tf.reduce_sum(sample_w) + eps
    )

    # ---------------------------------------------------------------------
    # FedProx proximal term: (mu / 2) * ||w - w_global||^2
    # Need a constant snapshot of the broadcast global weights.
    # ---------------------------------------------------------------------
    trainable_vars = agent_model.trainable_weights


    prox_terms = []
    if len(trainable_vars) != len(shared_weights):
       raise ValueError("Mismatch: {} trainable vars vs {} shared weights".format(
        len(trainable_vars), len(shared_weights)
    ))
    for i, (var, w0) in enumerate(zip(trainable_vars, shared_weights)):
      if tuple(var.shape.as_list()) != np.array(w0).shape:
        raise ValueError("Shape mismatch at {}: var {} vs weight {}".format(
            i, var.shape.as_list(), np.array(w0).shape
        ))

    for var, w0 in zip(trainable_vars, shared_weights):
        w0_const = tf.constant(w0, dtype=var.dtype.base_dtype)
        prox_terms.append(tf.reduce_sum(tf.square(var - w0_const)))

    prox_term = 0.5 * mu * tf.add_n(prox_terms)
    total_loss = weighted_data_loss + prox_term

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_var)
    train_op = optimizer.minimize(total_loss, global_step=global_step)

    sess.run(tf.compat.v1.global_variables_initializer())
    agent_model.set_weights(theta)

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------
    print("Num training steps: {}".format(num_steps))
    start_offset = 0
    LR_SUM = 0
    for step in range(num_steps):
        if start_offset >= batch_size:
            break

        end_offset = min(start_offset + train_batchsize, batch_size)

        X_batch = x_batch[start_offset:end_offset]
        Y_batch = y_batch[start_offset:end_offset]

        wb = compute_sample_weights(Y_batch, class_weight_mode="balanced")
        
        _, data_loss_val, prox_val, total_loss_val, step_val = sess.run(
            [train_op, weighted_data_loss, prox_term, total_loss, global_step],
            feed_dict={x: X_batch, y: Y_batch, sample_w: wb}
        )
        lr_value = sess.run(lr_var)
        LR_SUM += float(lr_value)

        start_offset = end_offset
        
        # print("FedProx client {}, lr={}, mu={}".format(CURRENT_AGENT, lr, mu))

        # print("Agent {}, step {}, gs {}, data_loss {:.6f}, prox {:.6f}, total {:.6f}".format(
        #     CURRENT_AGENT, step, step_val, data_loss_val, prox_val, total_loss_val
        # ))

    # ---------------------------------------------------------------------
    # Final local model and delta
    # ---------------------------------------------------------------------
    local_weights = agent_model.get_weights()
    local_delta = local_weights - shared_weights

    eval_success, eval_loss = eval_minimal(X_test, Y_test, local_weights)

    seed = None
    delayedclient = "false"
    # max_delay_s = 0.1  # max .1 sec delay
    # rng = np.random.default_rng(seed if seed is not None else (12345 + CURRENT_AGENT))
    # if rng.random() < 0.3:    # delay only some clients
    #     delay = rng.exponential(scale=0.05)   # mean 0.05s
    #     delay = min(delay, max_delay_s)       # cap it
    #     time.sleep(float(delay))
    #     delayedclient = "true"

    client_str = "client_" + str(CURRENT_AGENT) + "_t_" + str(round_idx)
    results_dict[client_str] = {
        "t": round_idx,
        "i": CURRENT_AGENT,
        "eval_success": eval_success,
        "eval_loss": eval_loss,
        "drift": "",                 # no drift logic in plain FedProx
        "delayed": delayedclient
    }

    print('Agent {}: success {}, loss {}'.format(CURRENT_AGENT, eval_success, eval_loss))
    
    delay_rng = np.random.default_rng(client_seed)
    delay_prob = 0.5 #delay only 25% of the clients
    max_delay = 2 # max delay is three rounds
    delay = 0
    if delay_rng.random() < delay_prob:
      delay = delay_rng.integers(1, max_delay + 1)


    print('Agent {}: success {}, loss {}'.format(CURRENT_AGENT, eval_success, eval_loss))#  
    return_dict[f"{CURRENT_AGENT}_r{round_idx}_weights"] = np.array(local_delta)
    return_dict["theta{}".format(CURRENT_AGENT)] = np.array(local_weights)
    return_dict[f"{CURRENT_AGENT}_r{round_idx}_num_samples"] = batch_size
    return_dict[str(CURRENT_AGENT) + "_lrsum"] = LR_SUM
    return_dict[f"{CURRENT_AGENT}_r{round_idx}_round_created"] = round_idx
    return_dict[f"{CURRENT_AGENT}_r{round_idx}_round_arrived"] = round_idx + delay
    print(
      f"Added a delay for {CURRENT_AGENT} at round {round_idx} "
      f"to round_arrived {round_idx + delay}, delay={delay}"
    )

    np.save(gv.dir_name + 'ben_delta_%s_t%s.npy' % (CURRENT_AGENT, round_idx), local_delta)

    return