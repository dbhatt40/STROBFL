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
from utils.synclass1_utils import synclass1_model,PageHinkley, LossStabilityTest
from utils.synclass1_utils import _reset_accumulators, _update_from_minibatch, _compute_metrics_from_acc
import time


def compute_sample_weights(y_batch, class_weight_mode="balanced"):
    B = len(y_batch)

    if class_weight_mode == "none":
        return np.ones(B, dtype=np.float32)

    classes, counts = np.unique(y_batch, return_counts=True)

    if class_weight_mode == "balanced":
        # inverse-frequency
        weights = B / (len(classes) * counts)

    # map each label to its weight
    class_to_w = dict(zip(classes, weights))

    return np.array([class_to_w[y] for y in y_batch], dtype=np.float32)



def synclass1_agent(current_agent, x_batch, y_batch, round_idx, gpu_id, return_dict, results_dict, X_test, Y_test, lr=None):
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
#--------------------------------------------------------------------
   	
    if pre_theta is not None:
        theta = pre_theta - gv.moving_rate * (pre_theta - shared_weights)
    else:
        theta = shared_weights
    agent_model.set_weights(theta)
    
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
    eps = 1e-8
    class_w_ph = tf.placeholder(tf.float32, shape=[gv.NUM_CLASSES], name="class_w")
# Per-example loss: shape [B]
    per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y,
                logits=logits
                )
    model_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y,
        logits=logits,
    )
# --- Per-label sums and counts ---
    y_int = tf.cast(y, tf.int32)
# sum of losses per label: shape [C]
# =============================================================================

# --- If you want a scalar loss with class weights ---
# weight each example by its label's weight
    w_per_example = tf.gather(class_w_ph, y_int)  # shape [B]
# =============================================================================
    weighted_loss = tf.reduce_sum(w_per_example * per_example_loss) / (
            tf.reduce_sum(w_per_example) + eps
            )

    alpha_stable= 0.8
    alpha_var = tf.Variable(alpha_stable, trainable=False,
                        dtype=tf.float32, name="ema_alpha")
        # EMA-based update rule using alpha_var
    ema_rule = gradient_update_rule_factory(alpha_var, name_prefix="grad_ema")
     
    optimizer = CustomRuleSGD(learning_rate=lr_var, update_rule=ema_rule)
    train_op = optimizer.minimize(weighted_loss, global_step=global_step)
    
    reset_ema_op = ema_rule.make_reset_op()
    num_classes = gv.NUM_CLASSES      

    AGG_STEPS = 3          # 2 steps * minibatch 10 => effective metric batch 20
    MIN_LABEL_CT = 5         # require >=2 true samples of a label in the aggregated window before PH update

    # --- Per-round accumulators for metrics (reset at start of each round) ---
    cm_probe_acc = np.zeros((num_classes, num_classes), dtype=np.float64)   # aggregated confusion matrix
    loss_probe_sum_acc = np.zeros(num_classes, dtype=np.float64)            # sum of per-example loss per true label
    cnt_probe_acc = np.zeros(num_classes, dtype=np.float64)   
              # count per true label

#-------------------------------------------------------------------------------------
    x_probe_ph = tf.placeholder(tf.float32, shape=[None, gv.DATA_DIM], name="x_probe")
    y_probe_ph = tf.placeholder(tf.int32,   shape=[None],             name="y_probe")
    w_probe_ph = tf.placeholder(tf.float32, shape=[gv.NUM_CLASSES],   name="w_probe")
    logits_probe = agent_model(x_probe_ph) 

    pred_probe = tf.argmax(logits_probe, axis=1, output_type=tf.int32)
    per_ex_loss_probe = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=y_probe_ph,
    logits=logits_probe
    )
    y_probe_int = tf.cast(y_probe_ph, tf.int32)
    eps = 1e-8

    loss_sum_probe_per_label = tf.math.unsorted_segment_sum(
      per_ex_loss_probe,
      y_probe_int,
      gv.NUM_CLASSES
    )

    cnt_probe_per_label = tf.math.unsorted_segment_sum(
      tf.ones_like(per_ex_loss_probe, tf.float32),
      y_probe_int,
      gv.NUM_CLASSES
    )

    per_label_probe_loss = tf.math.divide_no_nan(
      loss_sum_probe_per_label,
      cnt_probe_per_label + eps
    )

# Confusion matrix on probe
    cm_probe = tf.math.confusion_matrix(
        y_probe_ph, pred_probe, num_classes=num_classes, dtype=tf.float32
    )

    tp_p = tf.linalg.diag_part(cm_probe)
    pred_pos_p = tf.reduce_sum(cm_probe, axis=0)   # predicted positives
    act_pos_p  = tf.reduce_sum(cm_probe, axis=1)   # actual positives

    fp_p = pred_pos_p - tp_p
    fn_p = act_pos_p - tp_p

    precision_p = tf.math.divide_no_nan(tp_p, tp_p + fp_p)
    recall_p    = tf.math.divide_no_nan(tp_p, tp_p + fn_p)

    f1_probe_per_label = tf.math.divide_no_nan(
        2.0 * precision_p * recall_p, precision_p + recall_p
    )
    f1_probe_macro = tf.reduce_mean(f1_probe_per_label)

# Reuse the same model to compute logits on probe inputs
      # however you build logits; must reuse weights

    probe_loss = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_probe_ph, logits=logits_probe) * 
            tf.gather(w_probe_ph, y_probe_ph)
            )
    counts_probe = np.bincount(Y_test.astype(np.int32), minlength=gv.NUM_CLASSES)
    w_probe = counts_probe.sum() / np.maximum(counts_probe, 1)
    w_probe = np.clip(w_probe, 0.5, 3.0).astype(np.float32)        
    w_probe = w_probe / w_probe.mean()
    
#-------------------------run with initial probe-on global model----------------------------------------------
    
    sess.run(tf.global_variables_initializer())
    probe_loss_before, pll_p_before, f1m_p_before, f1l_p_before, cnt_p_before = sess.run(
      [probe_loss,
       per_label_probe_loss,
       f1_probe_macro,
       f1_probe_per_label,
       cnt_probe_per_label],
      feed_dict={
          x_probe_ph: X_test,
          y_probe_ph: Y_test,
          w_probe_ph: w_probe
         }
       )

    loss_before = float(probe_loss_before)
#------------------------------------definitions of PH and Stab classes----------------------------
    # print('loaded shared weights')
    cooldown_steps = 10
    loss_history_per_label = [[] for _ in range(num_classes)]
    f1_history_per_label  =[[] for _ in range(num_classes)]
    loss_ph_per_label = [
      PageHinkley(current_agent,delta=0.01, lambd=0.9, min_instances=20,signal_type="loss")
      for _ in range(num_classes)
      ]
    f1_ph_per_label = [
      PageHinkley(current_agent, delta=0.02, lambd=1.0, min_instances=15, signal_type="f1-score")
      for _ in range(num_classes)
      ]
    
    stab = LossStabilityTest(window=10, min_increase=4.0, std_mult=3.5)
#--------------------------------------------------------------------------------------------------------
    for c in range(num_classes):
        # ----- loss signal: LEVEL -----
        loss_c = float(pll_p_before[c])           
        if np.isfinite(loss_c):
                      loss_history_per_label[c].append(loss_c)
                      ld = loss_ph_per_label[c].update(loss_c)   # PH will self-gate via min_instances
        f1_c = float(f1l_p_before[c])  
        if np.isfinite(f1_c):
                      err_f1 = 1.0 - f1_c                # higher = worse
                      f1_history_per_label[c].append(err_f1)
                      fd = f1_ph_per_label[c].update(err_f1)
    stab.update(loss_before)

#---------------------------------------------------------------------------------------
    start_offset = 0

    lr_stable = 0.8   
    alpha_stable = 0.05
    alpha_csdrift = alpha_stable*0.8
    alpha_cddrift = 0
    
    agg_k=0
    steps_since_drift = 0  # Python-side counter
    agent_drift = []
    _reset_accumulators(cm_probe_acc, loss_probe_sum_acc, cnt_probe_acc)
    print("Num training steps: {}".format(num_steps))
#-----------------------------------------------------training ----------------------
    for step in range(num_steps):
        start_offset = start_offset
        end_offset = start_offset + train_batchsize
        if(end_offset>batch_size):
            end_offset = batch_size

        X_batch = x_batch[start_offset: end_offset]
        Y_batch = y_batch[start_offset: end_offset]

        wb = compute_sample_weights(Y_batch, class_weight_mode="balanced")
          # print("For step: {} X_batch: {}, Y_batch: {}".format(step, X_batch, Y_batch))
                 
        train_loss_val, _ = sess.run(
                  [weighted_loss, train_op],
                  feed_dict={
                  x: X_batch,
                  y: Y_batch,
                  class_w_ph: wb
               }
          )
# 2) compute probe AFTER update
        pred_probe_after, per_ex_loss_probe_after = sess.run(
         [pred_probe, per_ex_loss_probe],
         feed_dict={x_probe_ph: X_test, y_probe_ph: Y_test, w_probe_ph: w_probe}
         )

        
        _update_from_minibatch(cm_probe_acc, loss_probe_sum_acc, cnt_probe_acc,
                        Y_test,
                        pred_probe_after,
                        per_ex_loss_probe_after,
                        num_classes)

        agg_k += 1

        if(agg_k % AGG_STEPS==0):
          act_pos, pll_val, f1l_val, f1m_val = _compute_metrics_from_acc(cm_probe_acc, loss_probe_sum_acc, cnt_probe_acc,eps)
           
          pll_val = np.nan_to_num(pll_val, nan=0.0)
          f1l_val = np.nan_to_num(f1l_val, nan=0.0)

          any_drift  = False
          loss_drift = False
          f1_drift   = False
          probe_loss_scalar = np.nanmean(pll_val)
# or (slightly more robust to imbalance)
          probe_loss_scalar = np.nansum(pll_val * cnt_probe_acc) / (np.sum(cnt_probe_acc) + eps)
          unstable, stats = stab.update(probe_loss_scalar)   

          
    #       for c in range(num_classes):
    # # ----- loss signal: LEVEL -----
    #           loss_c = float(pll_val[c])           
    #           if np.isfinite(loss_c) and act_pos[c] >= MIN_LABEL_CT:
    #               loss_history_per_label[c].append(loss_c)
    #               ld = loss_ph_per_label[c].update(loss_c)   # PH will self-gate via min_instances
    #               loss_drift |= ld
    #               any_drift  |= ld

    # # ----- F1 signal: ERROR LEVEL -----
    #                       # <-- level
    #           f1_c = float(f1l_val[c])  
    #           if np.isfinite(f1_c) and act_pos[c] >= MIN_LABEL_CT:
    #               err_f1 = 1.0 - f1_c                # higher = worse
    #               f1_history_per_label[c].append(err_f1)
    #               fd = f1_ph_per_label[c].update(err_f1)
    #               f1_drift |= fd
    #               any_drift |= fd

        
    #       if unstable and "u" not in agent_drift:
    #            agent_drift.append("u")
    #            driftstr = "-".join(agent_drift)
    #            print(f"Drift {driftstr} detected in client: {current_agent}")
    #            # lr_val = sess.run(lr_var)
    #            # sess.run(lr_var.assign(lr_val*0.9))
    #            steps_since_drift = 0
    #            unstable = False
    #       elif loss_drift and f1_drift and "cd" not in agent_drift:
    #            agent_drift.append("cd")
    #            driftstr = "-".join(agent_drift)
    #            print(f"Drift {driftstr} detected in client: {current_agent}")
    #            # sess.run(reset_ema_op)
    #            # lr_val = sess.run(lr_var)
    #            # sess.run(lr_var.assign(lr_val*0.6))              
    #            # sess.run(alpha_var.assign(alpha_cddrift))
    #            steps_since_drift = 0
    #            loss_drift = False
    #            f1_drift = False
    #       elif loss_drift and "cs" not in agent_drift:
    #            agent_drift.append("cs")
    #            driftstr = "-".join(agent_drift)
    #            print(f"Drift {driftstr} detected in client: {current_agent}")       
    #            # sess.run(alpha_var.assign(alpha_csdrift))
    #            steps_since_drift = 0   
    #            loss_drift = False
    #       else:
    #            steps_since_drift += 1
    #            if steps_since_drift >= cooldown_steps:
    #                # sess.run(alpha_var.assign(alpha_stable))
    #                # sess.run(lr_var.assign(lr_stable))
                  
          _reset_accumulators(cm_probe_acc, loss_probe_sum_acc, cnt_probe_acc)
          agg_k = 0

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


