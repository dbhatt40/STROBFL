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

from utils.synclass1_utils import ZPageHinkley, LossStabilityTest
from customSGD import CustomRuleSGD, gradient_update_rule_factory
from utils.synclass1_utils import build_2step_accumulators

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gv.mem_frac)

LR_STABLE = 0.1
LR_CD_DRIFT = LR_STABLE*1.1
LR_UNSTABLE = LR_STABLE*1.1

ALPHA_STABLE = 0.8
ALPHA_CD_DRIFT = ALPHA_STABLE*0.9
ALPHA_UNSTABLE = ALPHA_STABLE*0.8

COOLDOWN_STEPS = 4
WARMUP_STEPS = 5



def detect_drift(
    eps,
    mse_loss,
    stability,
    loss_ph,
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

    
    mse_val = np.nan_to_num(mse_loss, nan=0.0, posinf=0.0, neginf=0.0)
    unstable, stab_stats = stability.update(mse_val)

    loss_drift = False
    loss_c = float(mse_val)
    if np.isfinite(loss_c):
       ld = bool(loss_ph.update(loss_c))
       loss_drift |= ld


    # Decision logic (your original ordering)
    drift_flag = False
    
    if loss_drift and "cd" not in agent_drift:
        agent_drift.append("cd")
        print(f"Drift {'-'.join(agent_drift)} detected in client: {CURRENT_AGENT}")
        old_lr, old_alpha = sess.run([lr_var, alpha_var])
        sess.run(reset_ema_op)
        # sess.run(lr_var.assign(old_lr*1.5))
        # sess.run(alpha_var.assign(0))
        drift_flag = True
    elif unstable and "u" not in agent_drift:
        agent_drift.append("u")
        print(f"Drift {'-'.join(agent_drift)} detected in client: {CURRENT_AGENT}")
        old_lr, old_alpha = sess.run([lr_var, alpha_var])
        sess.run(lr_var.assign(old_lr*1.1))
        sess.run(alpha_var.assign(old_alpha*0.8))
        drift_flag = True           
 
    return drift_flag

def aq_agent(i, x_batch, y_batch, t, gpu_id, return_dict, results_dict, X_test, Y_test, y_scaler, client_seed):
    CURRENT_AGENT = i
    round_idx = t
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

    mse_loss = tf.reduce_mean(tf.losses.mean_squared_error(y, logits))
    if args.optimizer == 'adam':
        lr=3e-3
        optimizer = tf.train.AdamOptimizer(
            learning_rate=lr).minimize(mse_loss)
    elif args.optimizer == 'strobfl_learn':
       lr=3e-2
       alpha = 0.5
       alpha_var = tf.Variable(alpha, trainable=False, dtype=tf.float32, name="ema_alpha")
       ema_rule = gradient_update_rule_factory(alpha_var, name_prefix="grad_ema")
       optimizer = CustomRuleSGD(learning_rate=lr, update_rule=ema_rule).minimize(mse_loss)
    
    lr_var = tf.Variable(LR_STABLE, trainable=False, name="lr")
    alpha_var = tf.Variable(ALPHA_STABLE, trainable=False,  dtype=tf.float32, name="ema_alpha")
    sess.run(tf.global_variables_initializer())
    agent_model.set_weights(shared_weights)
   
    reset_ema_op = ema_rule.make_reset_op()
    eps = 1e-8
    batch_size = x_batch.shape[0]
    train_size = args.B

    loss_ph = ZPageHinkley(CURRENT_AGENT,alpha=0.02, delta_z=0.5, lambd_z=20, min_instances=30,signal_type="loss")   
    stability = LossStabilityTest(window=20, min_increase=10.0, std_mult=15)

    steps_since_drift = 0  # Python-side counter
    agent_drift = []
    DRIFT_FLAG = False
    
#---------------------------------------------training
    LR_SUM=0

    num_steps = 0
   	
    for start in range(0,batch_size,train_size):
       
        end = min(start + train_size, batch_size)
        X_batch = x_batch[start:end].astype(np.float32)
        Y_batch = y_batch[start:end]
        _, loss_val = sess.run([optimizer, mse_loss], feed_dict={x: X_batch, y: Y_batch})	

        # print('Agent %s, Step %s, Loss %s, Train step %s' % (i, step, loss_val, step_val))

        if(num_steps >= WARMUP_STEPS) :


            if ((DRIFT_FLAG==False) or ((DRIFT_FLAG==True) and (steps_since_drift>=COOLDOWN_STEPS))):
                    
                    DRIFT_FLAG = detect_drift(eps, loss_val,stability, loss_ph, agent_drift, 
                                              reset_ema_op,sess, lr_var, alpha_var, CURRENT_AGENT)
                    if(DRIFT_FLAG==True):
                      steps_since_drift = 0

            elif (DRIFT_FLAG==True) and (steps_since_drift<COOLDOWN_STEPS):
                    steps_since_drift += 1

                    if(steps_since_drift==COOLDOWN_STEPS):
                        sess.run(lr_var.assign(LR_STABLE))              
                        sess.run(alpha_var.assign(ALPHA_STABLE))
                        DRIFT_FLAG = False
                        steps_since_drift = 0
                        
        num_steps += 1
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
    
    client_str = "client_" + str(CURRENT_AGENT) + "_t_" + str(round_idx)
    driftstr = "-".join(agent_drift)
    delayedstr = delayedclient
    results_dict[client_str] = {"t": round_idx, "i": CURRENT_AGENT, "eval_success": eval_success, "eval_loss": eval_loss, "drift": driftstr, "delayed":delayedstr}  
    
    
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
