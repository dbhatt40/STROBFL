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
import math

from customSGD import CustomRuleSGD, gradient_update_rule_factory
from utils.synclass1_utils import synclass1_model,PageHinkley, LossStabilityTest
from utils.synclass1_utils import _reset_accumulators, _update_from_minibatch, _compute_metrics_from_acc
import time

LR_STABLE = 0.1
LR_CD_DRIFT = LR_STABLE*1.5
LR_UNSTABLE = LR_STABLE*1.5

ALPHA_STABLE = 0.8
ALPHA_CS_DRIFT = ALPHA_STABLE*0.25
ALPHA_CD_DRIFT = 0
ALPHA_UNSTABLE = ALPHA_STABLE*0.25

COOLDOWN_STEPS = 2
NUM_CLASSES = 4
AGG_STEPS = 3          # 2 steps * minibatch 10 => effective metric batch 20
MIN_LABEL_CT = 5        # require >=2 true samples of a label in the aggregated window before PH update


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


#--------------------------intialize-----------------------------------

def synclass1_agent(current_agent, x_batch, y_batch, x_client_test, y_client_test, round_idx, gpu_id, return_dict, results_dict, X_test, Y_test, lr=None):
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
        lr = args.eta
    print('Agent %s on GPU %s' % (CURRENT_AGENT,gpu_id))
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

    NUM_CLASSES = gv.NUM_CLASSES
    batch_size = len(x_batch)
    num_steps = math.ceil(batch_size/train_batchsize)   
# Global step (optional but useful)
    global_step = tf.Variable(0, trainable=False, name="global_step")
# Custom optimizer
    lr_var = tf.Variable(1e-1, trainable=False, name="lr")
    eps = 1e-8
    # class_w_ph = tf.placeholder(tf.float32, shape=[gv.NUM_CLASSES], name="class_w")
    sample_w = tf.compat.v1.placeholder(tf.float32, shape=[None], name="sample_w")

# Per-example loss: shape [B]
    per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y,
                logits=logits
                )
    # model_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     labels=y,
    #     logits=logits,
    # )
# --- Per-label sums and counts ---
    y_int = tf.cast(y, tf.int32)
# sum of losses per label: shape [C]
# =============================================================================

# --- If you want a scalar loss with class weights ---
# weight each example by its label's weight
    w_per_example = tf.gather(sample_w, y_int)  # shape [B]
# =============================================================================
    weighted_loss = tf.reduce_sum(w_per_example * per_example_loss) / (
            tf.reduce_sum(w_per_example) + eps
            )


    alpha_var = tf.Variable(ALPHA_STABLE, trainable=False,
                        dtype=tf.float32, name="ema_alpha")
        # EMA-based update rule using alpha_var
    ema_rule = gradient_update_rule_factory(alpha_var, name_prefix="grad_ema")
     
    optimizer = CustomRuleSGD(learning_rate=lr_var, update_rule=ema_rule)
    train_op = optimizer.minimize(weighted_loss, global_step=global_step)
    
    reset_ema_op = ema_rule.make_reset_op()
    NUM_CLASSES = gv.NUM_CLASSES      



    # --- Per-round accumulators for metrics (reset at start of each round) ---
    cm_probe_acc = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)   # aggregated confusion matrix
    loss_probe_sum_acc = np.zeros(NUM_CLASSES, dtype=np.float64)            # sum of per-example loss per true label
    cnt_probe_acc = np.zeros(NUM_CLASSES, dtype=np.float64)   
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
        y_probe_ph, pred_probe, num_classes=NUM_CLASSES, dtype=tf.float32
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
    counts_probe = np.bincount(y_client_test.astype(np.int32), minlength=gv.NUM_CLASSES)
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
          x_probe_ph: x_client_test,
          y_probe_ph: y_client_test,
          w_probe_ph: w_probe
         }
       )

#------------------------------------definitions of PH and Stab classes----------------------------
    # print('loaded shared weights')
    loss_history_per_label = [[] for _ in range(NUM_CLASSES)]
    f1_history_per_label  =[[] for _ in range(NUM_CLASSES)]
    loss_ph_per_label = [
      PageHinkley(CURRENT_AGENT,delta=0.002, lambd=0.08, min_instances=10,signal_type="loss")
      for _ in range(NUM_CLASSES)
      ]
    f1_ph_per_label = [
      PageHinkley(CURRENT_AGENT, delta=0.01, lambd=0.1, min_instances=10, signal_type="f1-score")
      for _ in range(NUM_CLASSES)
      ]
    
    stability = LossStabilityTest(window=10, min_increase=0.05, std_mult=2.0)
#--------------------------------------------------------------------------------------------------------

    start_offset = 0
    
    agg_k=0
    agsteps_since_drift = 0  # Python-side counter
    agent_drift = []
    _reset_accumulators(cm_probe_acc, loss_probe_sum_acc, cnt_probe_acc)
    print("Num training steps: {}".format(num_steps))
    DRIFT_FLAG = False
#-----------------------------------------------------training ----------------------
    for step in range(num_steps):
        if start_offset >= batch_size:
          break
        end_offset = min(start_offset + train_batchsize, batch_size)

        X_batch = x_batch[start_offset: end_offset]
        Y_batch = y_batch[start_offset: end_offset]

        wb = compute_sample_weights(Y_batch, class_weight_mode="balanced")
          # print("For step: {} X_batch: {}, Y_batch: {}".format(step, X_batch, Y_batch))
                 
        train_loss_val, _ = sess.run(
                  [weighted_loss, train_op],
                  feed_dict={
                  x: X_batch,
                  y: Y_batch,
                  sample_w: wb
               }
          )
# 2) compute probe AFTER update
        pred_probe_after, per_ex_loss_probe_after = sess.run(
         [pred_probe, per_ex_loss_probe],
         feed_dict={x_probe_ph: x_client_test, y_probe_ph: y_client_test, w_probe_ph: w_probe}
         )
        
        _update_from_minibatch(cm_probe_acc, loss_probe_sum_acc, cnt_probe_acc,
                        y_client_test,
                        pred_probe_after,
                        per_ex_loss_probe_after,
                        NUM_CLASSES)
        agg_k += 1

        if(agg_k % AGG_STEPS==0):
            
           if ((DRIFT_FLAG==False) or (DRIFT_FLAG==True and agsteps_since_drift >= COOLDOWN_STEPS)):
              
              DRIFT_FLAG = detect_drift(cm_probe_acc, loss_probe_sum_acc, cnt_probe_acc, eps, NUM_CLASSES, loss_history_per_label, loss_ph_per_label, 
                                        f1_history_per_label, f1_ph_per_label, stability, agent_drift, agsteps_since_drift,
                                        reset_ema_op, sess, lr_var, alpha_var, CURRENT_AGENT)
              if(DRIFT_FLAG==True):
                  agsteps_since_drift = 0
           else:
             agsteps_since_drift += 1
             if(agsteps_since_drift==COOLDOWN_STEPS):
                 sess.run(lr_var.assign(LR_STABLE))              
                 sess.run(alpha_var.assign(ALPHA_STABLE))
    
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
    rng = np.random.default_rng(seed if seed is not None else (12345 + CURRENT_AGENT))
    if rng.random() < 0.3:    # delay only some clients
      delay = rng.exponential(scale=0.05)   # mean 0.05s
      delay = min(delay, max_delay_s)      # cap it
      time.sleep(float(delay))	
      delayedclient="true"
    
    client_str = "client_" + str(CURRENT_AGENT) + "_t_" + str(round_idx)
    driftstr = "-".join(agent_drift)
    delayedstr = delayedclient
    results_dict[client_str] = {"t": round_idx, "i": CURRENT_AGENT, "eval_success": eval_success, "eval_loss": eval_loss, "drift": driftstr, "delayed":delayedstr}  
    # print("Results dict:", results_dict[client_str])
    # print("Number of results_dict items - client:", len(results_dict))
 	
 	
    print('Agent {}: success {}, loss {}'.format(CURRENT_AGENT, eval_success, eval_loss))#  
    return_dict[str(CURRENT_AGENT)] = np.array(local_delta)
    return_dict["theta{}".format(CURRENT_AGENT)] = np.array(local_weights)
    return_dict[str(CURRENT_AGENT) + "_num_samples"] = batch_size
    return_dict[str(CURRENT_AGENT) + "_time"] = time.time()

    np.save(gv.dir_name + 'ben_delta_%s_t%s.npy' % (CURRENT_AGENT, round_idx), local_delta)


    return


def detect_drift(cm_probe_acc, loss_probe_sum_acc, cnt_probe_acc,eps, num_classes,loss_history_per_label,
                 loss_ph_per_label,f1_history_per_label, f1_ph_per_label, stability, agent_drift, steps_since_drift, 
                 reset_ema_op, sess, lr_var, alpha_var, CURRENT_AGENT):
    
   act_pos, pll_val, f1l_val, f1m_val = _compute_metrics_from_acc(cm_probe_acc, loss_probe_sum_acc, cnt_probe_acc,eps)
           
   pll_val = np.nan_to_num(pll_val, nan=0.0)
   f1l_val = np.nan_to_num(f1l_val, nan=0.0)

   any_drift  = False
   loss_drift = False
   f1_drift   = False
          # probe_loss_scalar = np.nanmean(pll_val)
# or (slightly more robust to imbalance)
   probe_loss_scalar = np.nansum(pll_val * cnt_probe_acc) / (np.sum(cnt_probe_acc) + eps)
   unstable, stats = stability.update(probe_loss_scalar)   

          
   for c in range(num_classes):
    # ----- loss signal: LEVEL -----
              loss_c = float(pll_val[c])  

              if np.isfinite(loss_c) and cnt_probe_acc[c]  >= MIN_LABEL_CT:
                 

                  loss_history_per_label[c].append(loss_c)
                  ld = loss_ph_per_label[c].update(loss_c)   # PH will self-gate via min_instances
                  if ld:
                   loss_ph_per_label[c].reset()
                  loss_drift |= ld
                  any_drift  |= ld

    # ----- F1 signal: ERROR LEVEL -----
                          # <-- level
              f1_c = float(f1l_val[c])  

              if np.isfinite(f1_c) and cnt_probe_acc[c] >= MIN_LABEL_CT:

                  f1_c = min(max(f1_c, 0.0), 1.0)
                  err_f1 = 1.0 - f1_c                # higher = worse
                  f1_history_per_label[c].append(err_f1)
                  fd = f1_ph_per_label[c].update(err_f1)
                  if fd:
                   f1_ph_per_label[c].reset()
                  f1_drift |= fd
                  any_drift |= fd

        
   if unstable and "u" not in agent_drift:
                agent_drift.append("u")
                driftstr = "-".join(agent_drift)
                print(f"Drift {driftstr} detected in client: {CURRENT_AGENT}")
                sess.run(lr_var.assign(LR_UNSTABLE))
                drift_flag = True
                unstable = False
            
   elif loss_drift and f1_drift and "cd" not in agent_drift:
                agent_drift.append("cd")
                driftstr = "-".join(agent_drift)
                print(f"Drift {driftstr} detected in client: {CURRENT_AGENT}")
                sess.run(reset_ema_op)
                sess.run(lr_var.assign(LR_CD_DRIFT))              
                sess.run(alpha_var.assign(ALPHA_CD_DRIFT))
                drift_flag = True
                loss_drift = False
                f1_drift = False
                any_drift = False
   elif loss_drift and "cs" not in agent_drift:
                agent_drift.append("cs")
                driftstr = "-".join(agent_drift)
                print(f"Drift {driftstr} detected in client: {CURRENT_AGENT}")       
                sess.run(alpha_var.assign(ALPHA_CS_DRIFT))
                drift_flag = True
                loss_drift = False
                f1_drift = False
                any_drift=False
   else:
               drift_flag = False
               loss_drift = False
               f1_drift = False
               any_drift = False
               
   return drift_flag

              