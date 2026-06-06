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
from utils.synclass1_utils import build_2step_accumulators
import time

LR_STABLE = 0.1
# LR_CD_DRIFT = LR_STABLE*1.05
# LR_UNSTABLE = LR_STABLE*1.1
LR_CD_DRIFT = LR_STABLE
LR_UNSTABLE = LR_STABLE*0.6

ALPHA_STABLE = 0.9
# ALPHA_CD_DRIFT = ALPHA_STABLE*0.8
# ALPHA_UNSTABLE = ALPHA_STABLE*0.6
ALPHA_CD_DRIFT = ALPHA_STABLE*0.9
ALPHA_UNSTABLE = ALPHA_STABLE*0.8

# COOLDOWN_STEPS = 4
COOLDOWN_STEPS = 6
WARMUP_STEPS = 2

NUM_CLASSES = 4
AGG_STEPS = 2          # 2 steps * minibatch 10 => effective metric batch 20
MIN_LABEL_CT = 5        # require >=2 true samples of a label in the aggregated window before PH update
LR_SUM=0

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

def detect_drift(
    agg,
    eps,
    num_classes,
    min_label_ct,
    stability,
    loss_ph_per_label,
    f1_ph_per_label,
    loss_history_per_label,
    f1_history_per_label,
    agent_drift,
    reset_ema_op,
    sess,
    lr_var,
    alpha_var,
    CURRENT_AGENT,
):
    """
    Uses aggregated stats in `agg` (over your AGG_STEPS window) to detect drift.

    agg keys expected:
      - "label_counts"  [C] int
      - "loss_per_label" [C] float
      - "f1_per_label"   [C] float
      - "f1_macro"       scalar
      - "loss"           scalar
    """

    lbl_counts = np.asarray(agg["label_counts"], dtype=np.float32)         # [C]
    pll_val    = np.asarray(agg["loss_per_label"], dtype=np.float32)       # [C]
    f1l_val    = np.asarray(agg["f1_per_label"], dtype=np.float32)         # [C]

    # Optional, if you want them:
    # f1m_val    = float(agg["f1_macro"])
    # lossm_val  = float(agg["loss"])

    pll_val = np.nan_to_num(pll_val, nan=0.0, posinf=0.0, neginf=0.0)
    f1l_val = np.nan_to_num(f1l_val, nan=0.0, posinf=0.0, neginf=0.0)

    # Stability test on a scalar overall loss (count-weighted per-label loss)
    probe_loss_scalar = float(np.nansum(pll_val * lbl_counts) / (np.sum(lbl_counts) + eps))
    unstable, stab_stats = stability.update(probe_loss_scalar)

    loss_drift = False
    f1_drift   = False

    for c in range(num_classes):
        if lbl_counts[c] < min_label_ct:
            continue

        # ---- loss PH (level) ----
        loss_c = float(pll_val[c])
        if np.isfinite(loss_c):
            loss_history_per_label[c].append(loss_c)
            ld = bool(loss_ph_per_label[c].update(loss_c))
            loss_drift |= ld


        # ---- F1 PH on error level (1 - f1) ----
        f1_c = float(f1l_val[c])
        if np.isfinite(f1_c):
            f1_c = min(max(f1_c, 0.0), 1.0)
            err_f1 = 1.0 - f1_c
            f1_history_per_label[c].append(err_f1)
            fd = bool(f1_ph_per_label[c].update(err_f1))
            f1_drift |= fd

    # Decision logic (your original ordering)
    drift_flag = False
    
    if loss_drift and f1_drift and "cd" not in agent_drift:
        agent_drift.append("cd")
        print(f"Drift {'-'.join(agent_drift)} detected in client: {CURRENT_AGENT}")
        old_lr, old_alpha = sess.run([lr_var, alpha_var])
        sess.run(lr_var.assign(LR_CD_DRIFT))
        sess.run(alpha_var.assign(ALPHA_CD_DRIFT))
        drift_flag = True
    elif unstable and "u" not in agent_drift:
        agent_drift.append("u")
        print(f"Drift {'-'.join(agent_drift)} detected in client: {CURRENT_AGENT}")
        sess.run(reset_ema_op)
        # old_lr, old_alpha = sess.run([lr_var, alpha_var])
        # sess.run(lr_var.assign(LR_UNSTABLE))
        # sess.run(alpha_var.assign(ALPHA_UNSTABLE))
        drift_flag = True           
 



    return drift_flag

              
#--------------------------intialize-----------------------------------

def synclass1_agent(current_agent, x_batch, y_batch, round_idx, gpu_id, return_dict, results_dict, X_test, Y_test, client_seed, lr=None):
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
   	
    if pre_theta is not None:
        theta = pre_theta - gv.moving_rate * (pre_theta - shared_weights)
    else:
        theta = shared_weights
   
    
    NUM_CLASSES = gv.NUM_CLASSES
    batch_size = len(x_batch)
    num_steps = math.ceil(batch_size/train_batchsize)   
    
  #  ----------------------------------------------------------------------------
    
    x = tf.placeholder(shape=[None, gv.DATA_DIM], dtype=tf.float32, name="x")
    y = tf.placeholder(shape=[None],dtype=tf.int32, name="y")
    logits = agent_model(x, training=True)
    global_step = tf.Variable(0, trainable=False, name="global_step")

    lr_var = tf.Variable(LR_STABLE, trainable=False, name="lr")
    alpha_var = tf.Variable(ALPHA_STABLE, trainable=False,
                        dtype=tf.float32, name="ema_alpha")
    eps = 1e-8
    sample_w = tf.compat.v1.placeholder(tf.float32, shape=[None], name="sample_w")  # [B]
    w_per_example = sample_w  # [B]

    per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y,
                logits=logits
                )
    y_int = tf.cast(y, tf.int32)

    weighted_loss = tf.reduce_sum(w_per_example * per_example_loss) / (
                    tf.reduce_sum(w_per_example) + eps
                    )
    ema_rule = gradient_update_rule_factory(alpha_var, name_prefix="grad_ema")
     
    optimizer = CustomRuleSGD(learning_rate=lr_var, update_rule=ema_rule)
    train_op = optimizer.minimize(weighted_loss, global_step=global_step)
    
    reset_ema_op = ema_rule.make_reset_op()

#------------------------------------definitions of PH and Stab classes----------------------------
    # print('loaded shared weights')
    loss_history_per_label = [[] for _ in range(NUM_CLASSES)]
    f1_history_per_label  =[[] for _ in range(NUM_CLASSES)]
    loss_ph_per_label = [
      PageHinkley(CURRENT_AGENT,delta=0.1, lambd=0.1, min_instances=15,signal_type="loss")
      for _ in range(NUM_CLASSES)
      ]
    f1_ph_per_label = [
      PageHinkley(CURRENT_AGENT, delta=0.05, lambd=0.02, min_instances=12, signal_type="f1-score")
      for _ in range(NUM_CLASSES)
      ]
    
    stability = LossStabilityTest(window=10, min_increase=0.1, std_mult=3.5)
#--------------------------------------------------------------------------------------------------------
    logits_post = agent_model(x, training=False)
    per_ex_loss_post = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_int, logits=logits_post)

    update_accum_op, read_agg, reset_accum_op = build_2step_accumulators(
      logits=logits_post,
      y_int=y_int,
      num_classes=NUM_CLASSES,
      per_example_loss=per_ex_loss_post,
      scope="train_accum2"
     )
    
#----------------------------------------------------------------------------------------------------
 
    print("Num training steps: {}".format(num_steps))
    start_offset = 0
    agg_k=0
    agsteps_since_drift = 0  # Python-side counter
    agent_drift = []
    DRIFT_FLAG = False
#-----------------------------------------------------training ----------------------
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(reset_accum_op)
    agent_model.set_weights(theta)
    LR_SUM = 0
    
    for step in range(num_steps):
      if start_offset >= batch_size:
         break
      end_offset = min(start_offset + train_batchsize, batch_size)

      X_batch = x_batch[start_offset:end_offset]
      Y_batch = y_batch[start_offset:end_offset]

      wb = compute_sample_weights(Y_batch, class_weight_mode="balanced")
      lr_value = sess.run(lr_var)
      LR_SUM += float(lr_value)

    # Step 1: training update (pre-update loss not needed unless you want it)
      sess.run(train_op, feed_dict={x: X_batch, y: Y_batch, sample_w: wb})

    # Step 2: post-update stats accumulation
      sess.run(update_accum_op, feed_dict={x: X_batch, y: Y_batch})

      # print("******LR VAR: ", lr_value)
      # LR_SUM += num_steps

    # Move to next batch  ✅
      start_offset = end_offset
      
      

      agg_k += 1
      if (agg_k % AGG_STEPS == 0) and (step >= WARMUP_STEPS) :

          agg = sess.run(read_agg)          # agg["loss"], agg["loss_per_label"], agg["f1_per_label"], agg["f1_macro"], agg["label_counts"]
          sess.run(reset_accum_op)
          if ((DRIFT_FLAG==False) or ((DRIFT_FLAG==True) and (agsteps_since_drift>=COOLDOWN_STEPS))):
                 
                  DRIFT_FLAG = detect_drift(agg, eps, NUM_CLASSES, MIN_LABEL_CT,stability, loss_ph_per_label,f1_ph_per_label,
                                       loss_history_per_label,f1_history_per_label,agent_drift, reset_ema_op,sess, lr_var, alpha_var, CURRENT_AGENT)
                  if(DRIFT_FLAG==True):
                    agsteps_since_drift = 0

          elif (DRIFT_FLAG==True) and (agsteps_since_drift<COOLDOWN_STEPS):
                  agsteps_since_drift += 1
                  if(agsteps_since_drift==COOLDOWN_STEPS):
                      sess.run(lr_var.assign(LR_STABLE))              
                      sess.run(alpha_var.assign(ALPHA_STABLE))
                      DRIFT_FLAG = False
                      agsteps_since_drift = 0
            
        # print('Agent %s, Step %s, Loss %s, Train step %s' % (i, step, loss_val, step_val))



    local_weights = agent_model.get_weights()
    # print("Local weights shape:", local_weights[0].shape, local_weights[0])
    local_delta = local_weights - shared_weights
    local_delta = local_delta

    # eval_success, eval_loss = eval_minimal(X_test,Y_test,x, y, sess, prediction, loss)
    # print("Y test in agents:", Y_test.shape
  
    eval_success, eval_loss = eval_minimal(X_test, Y_test, local_weights)
    
    seed=None
    delayedclient = "false"
    # max_delay_s = 0.1 # max .1 sec delay
    # rng = np.random.default_rng(seed if seed is not None else (12345 + CURRENT_AGENT))
    # if rng.random() < 0.3:    # delay only some clients
    #   delay = rng.exponential(scale=0.05)   # mean 0.05s
    #   delay = min(delay, max_delay_s)      # cap it
    #   time.sleep(float(delay))	
    #   delayedclient="true"
    
    client_str = "client_" + str(CURRENT_AGENT) + "_t_" + str(round_idx)
    driftstr = "-".join(agent_drift)
    delayedstr = delayedclient
    results_dict[client_str] = {"t": round_idx, "i": CURRENT_AGENT, "eval_success": eval_success, "eval_loss": eval_loss, "drift": driftstr, "delayed":delayedstr}  

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
    return_dict[str(CURRENT_AGENT) + "_lrsum"] = LR_SUM
    return_dict[f"{CURRENT_AGENT}_r{round_idx}_round_created"] = round_idx
    return_dict[f"{CURRENT_AGENT}_r{round_idx}_round_arrived"] = round_idx + delay
    print(
      f"Added a delay for {CURRENT_AGENT} at round {round_idx} "
      f"to round_arrived {round_idx + delay}, delay={delay}"
    )

    np.save(gv.dir_name + 'ben_delta_%s_t%s.npy' % (CURRENT_AGENT, round_idx), local_delta)


    return



