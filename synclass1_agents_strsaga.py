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
from utils.io_utils import file_write_train_metrics
from collections import deque
import time


PER_LABEL_STATS = {
    "sum": None,       # shape: [C, D]
    "count": None,     # shape: [C]
    "means": None      # shape: [C, D] (derived)
}

class LossStabilityTest:
    def __init__(self, window=10, min_increase=0.4, std_mult=3.0):
        self.window = int(window)
        self.min_increase = float(min_increase)
        self.std_mult = float(std_mult)
        self.buf = deque(maxlen=self.window)

    def update(self, loss_val):
        self.buf.append(float(loss_val))
        if len(self.buf) < self.window:
            return False, {}

        arr = np.array(self.buf, dtype=np.float32)
        half = self.window // 2
        early = arr[:half]
        late  = arr[half:]

        early_mean, late_mean = float(early.mean()), float(late.mean())
        early_std,  late_std  = float(early.std() + 1e-8), float(late.std() + 1e-8)

        mean_up = (late_mean - early_mean) / max(early_mean, 1e-8) > self.min_increase
        std_up  = late_std > self.std_mult * early_std

        unstable = mean_up and std_up
        stats = {
            "early_mean": early_mean, "late_mean": late_mean,
            "early_std": early_std,   "late_std": late_std
        }
        return unstable, stats


class PageHinkley:
    """
    Online Page-Hinkley drift detector (univariate).

    Detects a sustained *increase* in the monitored signal.
    To detect a decrease, call update() with -x instead of x.
    """
    def __init__(self, delta=0.05, lambd=0.8, min_instances=30):
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
            self.reset()
            return True

        return False


def synclass1_agent(current_agent, x_batch, y_batch, round_idx, gpu_id, return_dict, results_dict, X_test, Y_test, lr=None):
    tf.keras.backend.set_learning_phase(1)
	

    args = gv.init()
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
    x = tf.placeholder(shape=[None, gv.DATA_DIM], dtype=tf.float32, name="x")
    y = tf.placeholder(shape=[None],dtype=tf.int64, name="y")
    logits = agent_model(x)

    num_classes = gv.NUM_CLASSES
    batch_size = len(x_batch)

    num_steps = int(batch_size/train_batchsize)
   
# Global step (optional but useful)
    global_step = tf.Variable(0, trainable=False, name="global_step")

# Custom optimizer

    lr_var = tf.Variable(1e-1, trainable=False, name="lr")
    if args.optimizer == 'strsaga':
        eps = 1e-8
        class_w_ph = tf.placeholder(tf.float32, shape=[gv.NUM_CLASSES], name="class_w")
# Per-example loss: shape [B]
        per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y,
                logits=logits
                )
# --- Per-label sums and counts ---
        y_int = tf.cast(y, tf.int32)
# sum of losses per label: shape [C]
        loss_sum_per_label = tf.math.unsorted_segment_sum(
                data=per_example_loss,
                segment_ids=y_int,
                num_segments=gv.NUM_CLASSES
                )
# count per label: shape [C]
        ones = tf.ones_like(per_example_loss, dtype=tf.float32)
        count_per_label = tf.math.unsorted_segment_sum(
            data=ones,
            segment_ids=y_int,
            num_segments=gv.NUM_CLASSES
            )
                
              
# per-label mean loss: shape [C]
        per_label_loss = tf.where(
            count_per_label > 0.0,
            loss_sum_per_label / (count_per_label + eps),
            tf.zeros_like(loss_sum_per_label)
            )
# --- If you want a scalar loss with class weights ---
# weight each example by its label's weight
        w_per_example = tf.gather(class_w_ph, y_int)  # shape [B]

        weighted_loss = tf.reduce_sum(w_per_example * per_example_loss) / (
            tf.reduce_sum(w_per_example) + eps
            )
        trainable_vars = tf.trainable_variables()
        grad_tensors = tf.gradients(weighted_loss, trainable_vars)

        # Placeholders + assign ops for manual weight updates
        new_w_ph_list = [
            tf.placeholder(v.dtype, shape=v.shape, name="strsaga_new_w_%d" % i)
            for i, v in enumerate(trainable_vars)
        ]
        assign_ops = [
            tf.assign(v, nw_ph)
            for v, nw_ph in zip(trainable_vars, new_w_ph_list)
        ]
        

        mem_size = 10
        grad_mem = [None] * mem_size  # slots, each slot = list of gradients
        grad_sum = None               # list of arrays (running sum)
        mem_count = 0
        slot_idx = 0


        num_classes = gv.NUM_CLASSES

       
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
    
    steps_since_drift = 0  # Python-side counter
    agent_drift = []
    start_offset = 0
    print("Num training steps: {}".format(num_steps))
    for step in range(num_steps):
        start_offset = start_offset
        end_offset = start_offset + train_batchsize
        X_batch = x_batch[start_offset: end_offset]
        Y_batch = y_batch[start_offset: end_offset]
        if args.optimizer == 'strsaga':
            # --- STRSAGA local step ---
          counts = np.bincount(Y_batch.astype(np.int32), minlength=gv.NUM_CLASSES)
          w = counts.sum() / np.maximum(counts, 1)
          w = np.clip(w, 0.5, 3.0).astype(np.float32)
          w = w / w.mean()

          feed = {x: X_batch, y: Y_batch, class_w_ph: w}

            # 1) raw gradients for this batch
          grad_vals = sess.run(grad_tensors, feed_dict=feed)  # list of np arrays

            # 2) STRSAGA variance-reduction
          if grad_sum is None:
                grad_sum = [np.zeros_like(g) for g in grad_vals]
       
          j = slot_idx
          old_g = grad_mem[j]

          if mem_count == 0 or old_g is None:
                # memory empty -> plain SGD
             vr_grad = [g.copy() for g in grad_vals]
             for k, g in enumerate(grad_vals):
                    grad_sum[k] += g
                    grad_mem[j] = [g.copy() for g in grad_vals]
                    mem_count += 1
             else:
                # SAGA-style correction: g_t - phi_j + phi_bar
                phi_bar = [grad_sum[k] / float(mem_count) for k in range(len(grad_vals))]
                vr_grad = [g - old_g[k] + phi_bar[k] for k, g in enumerate(grad_vals)]
                for k, g in enumerate(grad_vals):
                    grad_sum[k] += g - old_g[k]
                grad_mem[j] = [g.copy() for g in grad_vals]
                slot_idx = (slot_idx + 1) % mem_size

               # 3) apply update: w <- w - lr_stable * vr_grad
                w_vals = sess.run(trainable_vars)
                new_w_vals = [w_ - lr_stable * g_ for w_, g_ in zip(w_vals, vr_grad)]

                assign_feed = {ph: nw for ph, nw in zip(new_w_ph_list, new_w_vals)}
                sess.run(assign_ops, feed_dict=assign_feed)

                # 4) Compute loss / per-label loss / F1 / preds for monitoring
                fetch_metrics = [weighted_loss]
                loss_val, _ = sess.run(
                    fetch_metrics,
                    feed_dict=feed
                    )
 
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


