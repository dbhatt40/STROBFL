# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 02:11:42 2026

@author: Divya
"""

# -*- coding: utf-8 -*-
"""
FedProx version for Air Quality dataset
"""

#########################
# Purpose: Mimics a benign agent in the federated learning setting
#          and sets up the master agent for FedProx on air quality data
#########################

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

import time

from utils.eval_utils import eval_minimal
from utils.air_quality_utils import airquality_model
import global_vars as gv

# --------------------------------------------------
# FedProx hyperparameters
# --------------------------------------------------
LR_FEDPROX = 5e-4
MU_FEDPROX = 1e-4  # tune this: e.g. 1e-4, 1e-3, 1e-2


def aq_agent_fedprox(i, x_batch, y_batch, t, gpu_id,
                     return_dict, results_dict,
                     X_test, Y_test, y_scaler, client_seed):
    """
    FedProx local client for air quality regression.

    Local objective:
        mse_loss + (mu/2) * ||w - w_global||^2
    """
    CURRENT_AGENT = i
    round_idx = t

    tf.reset_default_graph()
    tf.keras.backend.set_learning_phase(1)

    # ------------------------------------------
    # GPU environment
    # ------------------------------------------
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    args = gv.init()
    print('Agent %s on GPU %s' % (i, gpu_id))

    shared_weights = np.load(
        gv.dir_name + 'global_weights_t%s.npy' % t,
        allow_pickle=True
    )

    if args.k > 1:
        config = tf.ConfigProto(gpu_options=gv.gpu_options)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
    else:
        sess = tf.Session()

    tf.compat.v1.keras.backend.set_session(sess)

    # ------------------------------------------
    # Build model
    # ------------------------------------------
    agent_model = airquality_model()

    x = tf.placeholder(shape=(None, gv.DATA_DIM), dtype=tf.float32, name="x")
    y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="y")

    preds = agent_model(x)

    # Base regression loss
    mse_loss = tf.reduce_mean(tf.square(y - preds), name="mse_loss")

    # Build variables by running one forward construction
    trainable_vars = tf.trainable_variables()

    # ------------------------------------------
    # FedProx proximal term
    #   mu/2 * ||w - w_global||^2
    # Since tf.nn.l2_loss(z) = 1/2 * sum(z^2),
    # using mu * tf.nn.l2_loss(...) gives mu/2 * ||...||^2
    # ------------------------------------------
    global_w_placeholders = []
    prox_terms = []

    for idx, var in enumerate(trainable_vars):
        ph = tf.placeholder(
            dtype=tf.float32,
            shape=var.shape,
            name="global_w_ph_%d" % idx
        )
        global_w_placeholders.append(ph)
        prox_terms.append(tf.nn.l2_loss(var - ph))

    prox_term = tf.add_n(prox_terms, name="prox_term")
    total_loss = tf.identity(mse_loss + MU_FEDPROX * prox_term, name="total_loss")

    # ------------------------------------------
    # Optimizer
    # ------------------------------------------
    if args.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate=3e-3
        ).minimize(total_loss)

    elif args.optimizer == 'fedprox':
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=LR_FEDPROX
        ).minimize(total_loss)

    else:
        # default fallback to FedProx SGD
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=LR_FEDPROX
        ).minimize(total_loss)

    # ------------------------------------------
    # Initialize and load shared/global weights
    # ------------------------------------------
    sess.run(tf.global_variables_initializer())
    agent_model.set_weights(shared_weights)

    # Need a snapshot of global weights aligned to trainable vars
    # AFTER setting model weights
    global_weights_for_vars = sess.run(trainable_vars)

    # ------------------------------------------
    # Training
    # ------------------------------------------
    batch_size = x_batch.shape[0]
    train_size = args.B
    LR_SUM=0
    num_steps = 0

    for start in range(0, batch_size, train_size):
        end = min(start + train_size, batch_size)

        X_batch = x_batch[start:end].astype(np.float32)
        Y_batch = y_batch[start:end].astype(np.float32)

        feed_dict = {
            x: X_batch,
            y: Y_batch
        }

        # Add global model snapshot for proximal term
        for var_idx, ph in enumerate(global_w_placeholders):
            feed_dict[ph] = global_weights_for_vars[var_idx]

        _, loss_val, mse_val, prox_val = sess.run(
            [optimizer, total_loss, mse_loss, prox_term],
            feed_dict=feed_dict
        )

        num_steps += 1
        # Uncomment if you want step-wise debug
        # print("Agent {}, step {}, total_loss {:.6f}, mse {:.6f}, prox {:.6f}".format(
        #     i, num_steps, loss_val, mse_val, prox_val
        # ))

    # ------------------------------------------
    # Collect local model update
    # ------------------------------------------
    local_weights = agent_model.get_weights()
    local_delta = [lw - sw for lw, sw in zip(local_weights, shared_weights)]

    # ------------------------------------------
    # Evaluate local model
    # ------------------------------------------
    eval_success, eval_loss = eval_minimal(
        X_test, Y_test, local_weights, y_scaler=y_scaler
    )

    # Optional asynchronous delay simulation
    seed = None
    delayedclient = "false"
    # max_delay_s = 0.1
    # rng = np.random.default_rng(seed if seed is not None else (12345 + CURRENT_AGENT))

    # if rng.random() < 0.3:
    #     delay = rng.exponential(scale=0.05)
    #     delay = min(delay, max_delay_s)
    #     time.sleep(float(delay))
    #     delayedclient = "true"

    client_str = "client_" + str(CURRENT_AGENT) + "_t_" + str(round_idx)
    results_dict[client_str] = {
        "t": round_idx,
        "i": CURRENT_AGENT,
        "eval_success": eval_success,
        "eval_loss": eval_loss,
        "drift": "",              # FedProx version does not use your drift detector
        "delayed": delayedclient
    }

    print('Agent {}: success {}, loss {}'.format(i, eval_success, eval_loss))

    client_seed = client_seed + 1000*round_idx + CURRENT_AGENT
    delay_rng = np.random.default_rng(client_seed)
    delay_prob = 0.3 #delay only 25% of the clients
    max_delay = 2 # max delay is three rounds
    delay = 0
    if delay_rng.random() < delay_prob:
      delay = delay_rng.integers(1, max_delay + 1)

 	
    print('Agent {}: success {}, loss {}'.format(CURRENT_AGENT, eval_success, eval_loss))#  
    return_dict[f"{CURRENT_AGENT}_r{round_idx}_weights"] = np.array(local_delta)
    return_dict["theta{}".format(CURRENT_AGENT)] = np.array(local_weights)
    return_dict[f"{CURRENT_AGENT}_r{round_idx}_num_samples"] = batch_size
    return_dict[f"{CURRENT_AGENT}_r{round_idx}_lrsum"] = LR_SUM
    return_dict[f"{CURRENT_AGENT}_r{round_idx}_round_created"] = round_idx
    return_dict[f"{CURRENT_AGENT}_r{round_idx}_round_arrived"] = round_idx + delay
    print(
      f"Added a delay for {CURRENT_AGENT} at round {round_idx} "
      f"to round_arrived {round_idx + delay}, delay={delay}"
    )

    np.save(gv.dir_name + 'ben_delta_%s_t%s.npy' % (i, t), local_delta, allow_pickle=True)

    sess.close()
    return


def aq_master_fedprox():
    """
    Initializes the global model for FedProx training.
    """
    tf.reset_default_graph()
    tf.keras.backend.set_learning_phase(1)

    print('Initializing server models')
    config = tf.ConfigProto(gpu_options=gv.gpu_options)
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    global_model = airquality_model()

    # build variables by dummy forward if needed
    x_dummy = tf.placeholder(shape=(None, gv.DATA_DIM), dtype=tf.float32)
    _ = global_model(x_dummy)

    sess.run(tf.global_variables_initializer())

    global_weights_np = global_model.get_weights()
    np.save(gv.dir_name + 'global_weights_t0.npy', global_weights_np, allow_pickle=True)
    print("[server] save global weights t0")

    sess.close()
    return