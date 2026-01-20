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
import time

from synclass1_utils import PageHinkley, LossStabilityTest
from customSGD import CustomRuleSGD, gradient_update_rule_factory
from utils.synclass1_utils import build_2step_accumulators

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gv.mem_frac)

LR_STABLE = 0.1
LR_CD_DRIFT = LR_STABLE*1.5
LR_UNSTABLE = LR_STABLE*1.5

ALPHA_STABLE = 0.8
ALPHA_CS_DRIFT = ALPHA_STABLE*0.25
ALPHA_CD_DRIFT = 0
ALPHA_UNSTABLE = ALPHA_STABLE*0.25

COOLDOWN_STEPS = 2
WARMUP_STEPS = 4
NUM_CLASSES = 4
AGG_STEPS = 2          # 2 steps * minibatch 10 => effective metric batch 20
MIN_LABEL_CT = 5        # require >=2 true samples of a label in the aggregated window before PH update


def detect_drift(
    agg,
    eps,
    stability,
    loss_ph,
    mseloss,
    loss_history,
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
      - "mseloss" [C] float
    """

    msel_val    = np.asarray(agg["mseloss"], dtype=np.float32)       
    mse_val = np.nan_to_num(mseloss, nan=0.0, posinf=0.0, neginf=0.0)
    unstable, stab_stats = stability.update(mse_val)

    loss_drift = False
    loss_c = float(mse_val)
    if np.isfinite(loss_c):
       loss_history.append(loss_c)
       ld = bool(loss_ph.update(loss_c))
       loss_drift |= ld


    # Decision logic (your original ordering)
    drift_flag = False
    
    if loss_drift and "cd" not in agent_drift:
        agent_drift.append("cd")
        print(f"Drift {'-'.join(agent_drift)} detected in client: {CURRENT_AGENT}")
        old_lr, old_alpha = sess.run([lr_var, alpha_var])
        sess.run(reset_ema_op)
        sess.run(lr_var.assign(old_lr*1.5))
        sess.run(alpha_var.assign(0))
        drift_flag = True
    elif unstable and "u" not in agent_drift:
        agent_drift.append("u")
        print(f"Drift {'-'.join(agent_drift)} detected in client: {CURRENT_AGENT}")
        old_lr, old_alpha = sess.run([lr_var, alpha_var])
        sess.run(lr_var.assign(old_lr*1.5))
        drift_flag = True           
 
    return drift_flag

def aq_agent(i, x_batch, y_batch, t, gpu_id, return_dict, results_dict, X_test, Y_test, y_scaler):
    CURRENT_AGENT = i
    tf.reset_default_graph()
    tf.keras.backend.set_learning_phase(1)        

    # set environment
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    args = gv.init()
    print('Agent %s on GPU %s' % (i,gpu_id))
    shared_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t, allow_pickle=True)

    if args.k > 1:
        config = tf.ConfigProto(gpu_options=gv.gpu_options)
        config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = 0.05
        sess = tf.Session(config=config)
    else:
        sess = tf.Session()

    tf.compat.v1.keras.backend.set_session(sess)

#----------build model
    
    agent_model = airquality_model()
    x = tf.placeholder(shape=(None, gv.DATA_DIM), dtype=tf.float32)
    y = tf.placeholder(shape=(None, 1), dtype=tf.float32, name="y")
    logits = agent_model(x)

    loss = tf.reduce_mean(tf.losses.mean_squared_error(y, logits))
    if args.optimizer == 'adam':
        lr=1e-3
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr).minimize(loss)
    elif args.optimizer == 'strobfl_learn':
       lr=3e-2
       alpha = 0.5
       alpha_var = tf.Variable(alpha, trainable=False, dtype=tf.float32, name="ema_alpha")
       ema_rule = gradient_update_rule_factory(alpha_var, name_prefix="grad_ema")
       optimizer = CustomRuleSGD(learning_rate=lr, update_rule=ema_rule).minimize(loss)
    
    sess.run(tf.global_variables_initializer())
    agent_model.set_weights(shared_weights)
    # update_accum_op, read_agg, reset_accum_op = build_2step_accumulators(
    #   logits=logits,
    #   y_int=y,
    #   per_example_loss=mse_loss,
    #   scope="train_accum2"
    #  )
  		
   	# print('loaded shared weights')
    # loss_history = []
    # loss_ph = PageHinkley(CURRENT_AGENT,delta=0.1, lambd=0.1, min_instances=15,signal_type="loss")   
    # stability = LossStabilityTest(window=10, min_increase=0.1, std_mult=3.5)

    # start_offset = 0
    # agg_k=0
    # agsteps_since_drift = 0  # Python-side counter
    # agent_drift = []
    # DRIFT_FLAG = False
#---------------------------------------------training


    batch_size = x_batch.shape[0]
    train_size = args.B
    num_steps = 0
   	
    for start in range(0,batch_size,train_size):
        num_steps += 1
        end = min(start + train_size, batch_size)
        X_batch = x_batch[start:end].astype(np.float32)
        Y_batch = y_batch[start:end]
        _, loss_val = sess.run([optimizer, loss], feed_dict={x: X_batch, y: Y_batch})	

        # print('Agent %s, Step %s, Loss %s, Train step %s' % (i, step, loss_val, step_val))
        
        # agg_k += 1
        # if (agg_k % AGG_STEPS == 0) and (step >= WARMUP_STEPS) :

        #     agg = sess.run(read_agg)          # agg["loss"], agg["loss_per_label"], agg["f1_per_label"], agg["f1_macro"], agg["label_counts"]
        #     sess.run(reset_accum_op)
        #     if ((DRIFT_FLAG==False) or ((DRIFT_FLAG==True) and (agsteps_since_drift>=COOLDOWN_STEPS))):
                   
        #             DRIFT_FLAG = detect_drift(agg, eps, NUM_CLASSES, MIN_LABEL_CT,stability, loss_ph_per_label,f1_ph_per_label,
        #                                  loss_history_per_label,f1_history_per_label,agent_drift, reset_ema_op,sess, lr_var, alpha_var, CURRENT_AGENT)
        #             if(DRIFT_FLAG==True):
        #               agsteps_since_drift = 0

        #     elif (DRIFT_FLAG==True) and (agsteps_since_drift<COOLDOWN_STEPS):
        #             agsteps_since_drift += 1
        #             if(agsteps_since_drift==COOLDOWN_STEPS):
        #                 sess.run(lr_var.assign(LR_STABLE))              
        #                 sess.run(alpha_var.assign(ALPHA_STABLE))
        #                 DRIFT_FLAG = False
        #                 agsteps_since_drift = 0

    local_weights = agent_model.get_weights()
    # print("Local weights shape:", local_weights[0].shape, local_weights[0])
    local_delta = local_weights - shared_weights
    
    # w0 = shared_weights
    # w1 = agent_model.get_weights()
    # mean_abs_delta = np.mean([np.mean(np.abs(a-b)) for a,b in zip(w1, w0)])


    # eval_success, eval_loss = eval_minimal(X_test,Y_test,x, y, sess, prediction, loss)
    # print("Y test in agents:", Y_test.shape
  

    eval_success, eval_loss = eval_minimal(X_test, Y_test, local_weights, y_scaler=y_scaler)
    seed=None
    delayedclient = "false"
    max_delay_s = 0.1 # max .1 sec delay
    rng = np.random.default_rng(seed if seed is not None else (12345 + CURRENT_AGENT))
    if rng.random() < 0.3:    # delay only some clients
      delay = rng.exponential(scale=0.05)   # mean 0.05s
      delay = min(delay, max_delay_s)      # cap it
      time.sleep(float(delay))	
      delayedclient="true"
 	
    client_str = "client_" + str(i) + "_t_" + str(t)

    results_dict[client_str] = {"t": t, "i": i, "eval_success": eval_success, "eval_loss": eval_loss}  
    # print("Results dict:", results_dict[client_str])
    # print("Number of results_dict items - client:", len(results_dict))
 	
 	
    print('Agent {}: success {}, loss {}'.format(i, eval_success, eval_loss))#  
    return_dict[str(i)] = np.array(local_delta)
    return_dict["theta{}".format(i)] = np.array(local_weights)
    return_dict[str(CURRENT_AGENT) + "_num_samples"] = batch_size
    return_dict[str(CURRENT_AGENT) + "_time"] = time.time()

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
