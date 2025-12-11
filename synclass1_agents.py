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

from customSGD import CustomRuleSGD, gradient_update_rule_factory
from utils.synclass1_utils import synclass1_model


PER_LABEL_STATS = {
    "sum": None,       # shape: [C, D]
    "count": None,     # shape: [C]
    "means": None      # shape: [C, D] (derived)
}



class PageHinkley:
    """
    Online Page-Hinkley drift detector (univariate).

    Detects a sustained *increase* in the monitored signal.
    To detect a decrease, call update() with -x instead of x.
    """
    def __init__(self, delta=0.001, lambd=0.5, min_instances=30):
        """
        delta: small tolerance for slight changes (insensitivity zone)
        lambd: threshold for raising an alarm
        min_instances: wait for this many samples before triggering
        """
        self.delta = float(delta)
        self.lambd = float(lambd)
        self.min_instances = int(min_instances)

        self.reset()

    def reset(self):
        self.t = 0
        self.mean = 0.0
        self.cum_sum = 0.0
        self.min_cum_sum = 0.0
        self.ph_stat = 0.0
        self.drift = False

    def update(self, x):
        """
        Feed one new observation x.
        Returns True if drift detected at this step, else False.
        """
        self.t += 1

        # Incremental mean
        self.mean += (x - self.mean) / self.t

        # Cumulative sum of deviations (for increase detection)
        self.cum_sum += (x - self.mean - self.delta)

        # Track minimum of cumulative sum
        self.min_cum_sum = min(self.min_cum_sum, self.cum_sum)

        # Page-Hinkley statistic
        self.ph_stat = self.cum_sum - self.min_cum_sum

        # Drift decision
        if self.t > self.min_instances and self.ph_stat > self.lambd:
            self.drift = True
            # You can either reset here or leave it accumulating
            # self.reset()
            return True

        return False


def synclass1_agent(current_agent, x_batch, y_batch, round_idx, gpu_id, return_dict, results_dict, X_test, Y_test, lr=None):
    tf.keras.backend.set_learning_phase(1)
	

    args = gv.init()
    if lr is None:
        lr = args.eta
    print('Agent %s on GPU %s' % (current_agent,gpu_id))
    # set environment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    shared_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % round_idx, allow_pickle=True)
    pre_theta = None
	
    agent_model = synclass1_model()
    x = tf.placeholder(shape=[None, gv.DATA_DIM], dtype=tf.float32, name="x")
    y = tf.placeholder(shape=[None],dtype=tf.int64, name="y")
    logits = agent_model(x)

    lr=0.001
    # Per-example loss (vector)
    per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits
    )

   # Scalar loss for training
    loss = tf.reduce_mean(per_example_loss)

    
    num_classes = gv.NUM_CLASSES
    y_int = tf.cast(y, tf.int32)

    # One-hot encode [B, C]
    one_hot = tf.one_hot(y_int, depth=num_classes, dtype=tf.float32)

   # Expand per-example loss to [B, 1]
    per_loss_col = tf.expand_dims(per_example_loss, axis=1)

    # Sum of losses contributed by each label
    loss_sum_per_label = tf.reduce_sum(one_hot * per_loss_col, axis=0)

   # Samples per label
    count_per_label = tf.reduce_sum(one_hot, axis=0)

    eps = 1e-8

# Average loss per label
    per_label_loss = tf.where(
    count_per_label > 0,
    loss_sum_per_label / (count_per_label + eps),
    tf.zeros_like(loss_sum_per_label)
    )

# Predictions for this batch
    preds = tf.argmax(logits, axis=1, output_type=tf.int32)

# Confusion matrix [C, C]
    cm = tf.math.confusion_matrix(
       y_int, preds, num_classes=num_classes, dtype=tf.float32
    )

# True Positives
    tp = tf.linalg.diag_part(cm)

    pred_pos = tf.reduce_sum(cm, axis=0)  # predicted positives
    act_pos  = tf.reduce_sum(cm, axis=1)  # actual positives

    fp = pred_pos - tp
    fn = act_pos - tp

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)

# F1 per label
    f1_per_label = 2 * precision * recall / (precision + recall + eps)

# Macro F1
    f1_macro = tf.reduce_mean(f1_per_label)



    alpha_stable = 0.8  # when no drift
    alpha_drift  = 0.2  # when drift detected (faster adaptation)

    alpha_var = tf.Variable(alpha_stable, trainable=False,
                        dtype=tf.float32, name="ema_alpha")

# Global step (optional but useful)
    global_step = tf.Variable(0, trainable=False, name="global_step")

# EMA-based update rule using alpha_var
    ema_rule = gradient_update_rule_factory(alpha_var, name_prefix="grad_ema")

# Custom optimizer
    base_lr  = 0.01
    

    if args.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr)
        train_op = optimizer.minimize(loss, global_step=global_step)
    elif args.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=lr).minimize(loss)
    elif args.optimizer == 'strobfl_learn':
        optimizer = CustomRuleSGD(learning_rate=base_lr, update_rule=ema_rule)
        train_op = optimizer.minimize(loss, global_step=global_step)
    
    
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
 	
    batch_size = len(x_batch)

    B=gv.BATCH_SIZE
    num_steps = int(batch_size/B)
    start_offset = 0
    
    num_classes = gv.NUM_CLASSES

# History (optional, for logging/plotting)
    loss_history_per_label = [[] for _ in range(num_classes)]
    f1_history_per_label   = [[] for _ in range(num_classes)]

# Page-Hinkley detectors per label
# Tune delta / lambd / min_instances as needed
    loss_ph_per_label = [
      PageHinkley(delta=0.001, lambd=0.5, min_instances=30)
      for _ in range(num_classes)
   ]

    f1_ph_per_label = [
      PageHinkley(delta=0.001, lambd=0.5, min_instances=30)
      for _ in range(num_classes)
    ]
    f1_ph_per_label = [
       PageHinkley(delta=0.001, lambd=0.5, min_instances=30)
       for _ in range(num_classes)
   ]
 	
    alpha_stable = 0.8
    alpha_drift  = 0.2
    cooldown_steps = 100  # after this many steps without drift -> go back to stable

    steps_since_drift = 0  # Python-side counter
    for step in range(num_steps):
        offset = (start_offset + step * B) 
        X_batch = x_batch[offset: (offset + B)]
        Y_batch = y_batch[offset: (offset + B)]
        fetch_ops = [
           train_op,
           loss,
           per_label_loss,
           f1_macro,
           f1_per_label
         ]

        _, loss_val, pll_val, f1m_val, f1l_val = sess.run(fetch_ops,feed_dict={x: X_batch, y: Y_batch})

        pll_val = np.asarray(pll_val, dtype=np.float32)
        f1l_val = np.asarray(f1l_val, dtype=np.float32)

        pll_val = np.nan_to_num(pll_val, nan=0.0)
        f1l_val = np.nan_to_num(f1l_val, nan=0.0)

        any_drift = False

        for c in range(num_classes):
          loss_c = float(pll_val[c])
          f1_c   = float(f1l_val[c])

          # (optional) store histories...
          loss_history_per_label[c].append(loss_c)
          f1_history_per_label[c].append(f1_c)

          # Page–Hinkley on loss ↑
          loss_drift = loss_ph_per_label[c].update(loss_c)

         # Page–Hinkley on F1 ↓ (use -F1)
          f1_drift   = f1_ph_per_label[c].update(-f1_c)

          if loss_drift or f1_drift:
            any_drift = True
            print(f"[PH] Drift detected on label {c} at step {step} "
                  f"(loss_c={loss_c:.4f}, f1_c={f1_c:.4f})")

         # ---- Adapt EMA alpha based on drift ----
        if any_drift:
           # Concept drift: reduce alpha -> less memory, more weight to current grad
           sess.run(alpha_var.assign(alpha_drift))
           steps_since_drift = 0
        else:
           steps_since_drift += 1
           if steps_since_drift >= cooldown_steps:
              # Go back to “stable” alpha when no drift for a while
              sess.run(alpha_var.assign(alpha_stable))
              steps_since_drift = 0


        start_offset = offset
        # print('Agent %s, Step %s, Loss %s, Train step %s' % (i, step, loss_val, step_val))


    local_weights = agent_model.get_weights()
    # print("Local weights shape:", local_weights[0].shape, local_weights[0])
    local_delta = local_weights - shared_weights

    # eval_success, eval_loss = eval_minimal(X_test,Y_test,x, y, sess, prediction, loss)
    # print("Y test in agents:", Y_test.shape
  
    eval_success, eval_loss = eval_minimal(X_test, Y_test, local_weights)
 	
    client_str = "client_" + str(current_agent) + "_t_" + str(round_idx)
    results_dict[client_str] = {"t": round_idx, "i": current_agent, "eval_success": eval_success, "eval_loss": eval_loss}  
    # print("Results dict:", results_dict[client_str])
    # print("Number of results_dict items - client:", len(results_dict))
 	
 	
    print('Agent {}: success {}, loss {}'.format(current_agent, eval_success, eval_loss))#  
    return_dict[str(current_agent)] = np.array(local_delta)
    return_dict["theta{}".format(current_agent)] = np.array(local_weights)
    return_dict[str(current_agent) + "_num_samples"] = batch_size

    np.save(gv.dir_name + 'ben_delta_%s_t%s.npy' % (current_agent, round_idx), local_delta)

    return


