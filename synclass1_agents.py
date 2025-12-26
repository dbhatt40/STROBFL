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
    if args.optimizer == 'adam':
        per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(per_example_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr_var)
        train_op = optimizer.minimize(loss, global_step=global_step)
    elif args.optimizer == 'strobfl_learn':
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
        alpha_stable= 0.8
        alpha_var = tf.Variable(alpha_stable, trainable=False,
                        dtype=tf.float32, name="ema_alpha")
        # EMA-based update rule using alpha_var
        ema_rule = gradient_update_rule_factory(alpha_var, name_prefix="grad_ema")
     
        optimizer = CustomRuleSGD(learning_rate=lr_var, update_rule=ema_rule)
        train_op = optimizer.minimize(weighted_loss, global_step=global_step)
        reset_ema_op = ema_rule.make_reset_op()
        num_classes = gv.NUM_CLASSES

    # History (optional, for logging/plotting)
        loss_history_per_label = [[] for _ in range(num_classes)]
        f1_history_per_label   = [[] for _ in range(num_classes)]

    # Page-Hinkley detectors per label
    # Tune delta / lambd / min_instances as needed
        loss_ph_per_label = [
          PageHinkley(delta=0.08, lambd=20, min_instances=20)
          for _ in range(num_classes)
         ]

        f1_ph_per_label = [
          PageHinkley(delta=0.05, lambd=20, min_instances=30)
          for _ in range(num_classes)
          ]
  
        cooldown_steps = 10  # after this many steps without drift -> go back to stable
        stab = LossStabilityTest(window=10, min_increase=0.40, std_mult=3.0)
        
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
    loss_ema = None
    pll_ema = None
    f1l_ema = None
    ema_beta = 0.9  
 
    lr_stable = 0.1
    lr_unstable = lr_stable*0.5
    lr_lfdrift = lr_stable*0.75
    lr_ldrift = lr_stable*0.9
    
    alpha_stable = 0.8
    alpha_unstable = 0
    alpha_lfdrift = alpha_stable*0.625
    alpha_ldrift = alpha_stable*0.25
    
    steps_since_drift = 0  # Python-side counter
    print("Num steps: {}".format(num_steps))
    for step in range(num_steps):
        start_offset = start_offset
        end_offset = start_offset + train_batchsize
        X_batch = x_batch[start_offset: end_offset]
        Y_batch = y_batch[start_offset: end_offset]
        if args.optimizer == 'adam':
           _, loss_val = sess.run([train_op, loss], feed_dict={x: X_batch, y: Y_batch})	
        elif args.optimizer == 'strobfl_learn':
          counts = np.bincount(Y_batch.astype(np.int32), minlength=gv.NUM_CLASSES)
          # inverse frequency; boost rare classes strongly, but avoid huge explosions
          w = counts.sum() / np.maximum(counts, 1)
          w = np.clip(w, 0.5, 3.0).astype(np.float32)   # cap helps stability
          w = w / w.mean()
               
          # print("For step: {} X_batch: {}, Y_batch: {}".format(step, X_batch, Y_batch))
    
          pred_op = tf.argmax(logits, axis=1, output_type=tf.int32)
          fetch_ops = [train_op, weighted_loss, per_label_loss, f1_macro, f1_per_label, pred_op]
          _, loss_val, pll_val, f1m_val, f1l_val, pred_val = sess.run(
                                  fetch_ops, feed_dict={x: X_batch, y: Y_batch, class_w_ph: w})
          pll_val = np.nan_to_num(pll_val, nan=0.0)
          f1l_val = np.nan_to_num(f1l_val, nan=0.0)

# scalar loss EMA
          if loss_ema is None:
            loss_ema = loss_val
          else:
            loss_ema = ema_beta * loss_ema + (1.0 - ema_beta) * loss_val

          unstable, stats = stab.update(loss_ema)

# per-label loss EMA
          if pll_ema is None:
            pll_ema = pll_val.copy()
          else:
            pll_ema = ema_beta * pll_ema + (1.0 - ema_beta) * pll_val

# per-label F1 EMA (good since same-batch F1 is noisy)
          if f1l_ema is None:
            f1l_ema = f1l_val.copy()
          else:
            f1l_ema = ema_beta * f1l_ema + (1.0 - ema_beta) * f1l_val

# ---- Drift detection (PH per label) ----
          any_drift  = False
          loss_drift = False
          f1_drift   = False
          unstable = False

          MIN_COUNT_LOSS = int(train_batchsize/gv.NUM_CLASSES)
          MIN_COUNT_F1   = int(train_batchsize/gv.NUM_CLASSES)*2

          label_counts = np.bincount(Y_batch, minlength=gv.NUM_CLASSES)

          for c in range(num_classes):
             loss_c = float(pll_ema[c])
             f1_c   = float(f1l_ema[c])

             loss_history_per_label[c].append(loss_c)
             f1_history_per_label[c].append(f1_c)

    # loss PH
             if label_counts[c] >= MIN_COUNT_LOSS:
               ld = loss_ph_per_label[c].update(loss_c)
               loss_drift |= ld
               any_drift |= ld

    # F1 PH (use -F1, but only if enough support)
             if label_counts[c] >= MIN_COUNT_F1:
               fd = f1_ph_per_label[c].update(-f1_c)
               f1_drift  |= fd
               any_drift |= ld

# ---- Adapt EMA alpha based on drift ----

             if (unstable or any_drift):
                steps_since_drift = 0
                if unstable:
                   sess.run(reset_ema_op)
                   sess.run(alpha_var.assign(alpha_unstable))
                   sess.run(lr_var.assign(lr_unstable)) 
                elif loss_drift and f1_drift:
                   sess.run(alpha_var.assign(alpha_lfdrift))
                   sess.run(lr_var.assign(lr_lfdrift))                 
                elif loss_drift:
                  sess.run(alpha_var.assign(alpha_ldrift))
                  sess.run(lr_var.assign(lr_ldrift))                
             else:
                steps_since_drift += 1
                if steps_since_drift >= cooldown_steps:
                    sess.run(alpha_var.assign(alpha_stable))
                    sess.run(lr_var.assign(lr_stable)) 


        start_offset = end_offset

        # print('Agent %s, Step %s, Loss %s, Train step %s' % (i, step, loss_val, step_val))


    local_weights = agent_model.get_weights()
    # print("Local weights shape:", local_weights[0].shape, local_weights[0])
    local_delta = local_weights - shared_weights

    # eval_success, eval_loss = eval_minimal(X_test,Y_test,x, y, sess, prediction, loss)
    # print("Y test in agents:", Y_test.shape
  
    eval_success, eval_loss = eval_minimal(X_test, Y_test, local_weights)
    
    seed=None
    max_delay_s = 0.8 # max 2 sec delay
    rng = np.random.default_rng(seed if seed is not None else (12345 + current_agent))
    if rng.random() < 0.3:    # delay only some clients
      delay = rng.exponential(scale=0.5)   # mean 0.5s
      delay = min(delay, max_delay_s)      # cap it
      time.sleep(float(delay))	
    
    client_str = "client_" + str(current_agent) + "_t_" + str(round_idx)
    results_dict[client_str] = {"t": round_idx, "i": current_agent, "eval_success": eval_success, "eval_loss": eval_loss}  
    # print("Results dict:", results_dict[client_str])
    # print("Number of results_dict items - client:", len(results_dict))
 	
 	
    print('Agent {}: success {}, loss {}'.format(current_agent, eval_success, eval_loss))#  
    return_dict[str(current_agent)] = np.array(local_delta)
    return_dict["theta{}".format(current_agent)] = np.array(local_weights)
    return_dict[str(current_agent) + "_num_samples"] = batch_size
    return_dict[str(current_agent) + "_time"] = time.time()

    np.save(gv.dir_name + 'ben_delta_%s_t%s.npy' % (current_agent, round_idx), local_delta)

    return


