# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 18:49:22 2026

@author: Divya
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 13:40:33 2025

@author: Divya
"""

#########################
# Purpose: Mimics a benign agent in the federated learning setting and sets up the master agent
# Replaced with CDA-Fed client-side logic based on Algorithms 4, 5, 6 from:
# Casado et al., "Concept drift detection and adaptation for federated and continual learning"
########################

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
import math
import pickle
import random
import time

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.get_logger().setLevel(logging.ERROR)
tf.set_random_seed(99)
np.random.seed(99)
random.seed(99)

from utils.eval_utils import eval_minimal
import global_vars as gv
from utils.synclass1_utils import synclass1_model


# =========================================================
# CDA-Fed hyperparameters
# =========================================================
# These are paper-inspired, but scaled down for your client-round setting.
# Tune for your stream and per-round sample counts.
DEFAULT_LR = 0.1
Q_MAX = 200                 # Nmax: max size of sliding confidence window Q
DELTA_CDA = 40             # Δ: minimum sub-window size for detection
LAMBDA_CDA = 0.05          # λ: sensitivity to change
MIN_TRAIN_DATA = 40        # L: minimum amount of data before training a concept
LOCAL_ROUNDS_PER_CHANGE = 2  # R
LOCAL_EPOCHS_PER_ROUND = 1   # E
NUM_CLASSES_DEFAULT = 4
MIN_SAMPLES_BEFORE_DETECTION = 90


# =========================================================
# Utility helpers
# =========================================================
def compute_sample_weights(y_batch, class_weight_mode="balanced"):
    B = len(y_batch)

    if B == 0:
        return np.array([], dtype=np.float32)

    if class_weight_mode == "none":
        return np.ones(B, dtype=np.float32)

    classes, counts = np.unique(y_batch, return_counts=True)

    if class_weight_mode == "balanced":
        weights = B / (len(classes) * counts)
    else:
        weights = np.ones_like(counts, dtype=np.float32)

    class_to_w = dict(zip(classes, weights))
    return np.array([class_to_w[y] for y in y_batch], dtype=np.float32)


def batch_iter(X, Y, batch_size, shuffle=True):
    n = len(X)
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        ids = idx[s:e]
        yield X[ids], Y[ids]


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_cda_state_path(client_id):
    return os.path.join(gv.dir_name, "cda_state_client_{}.pkl".format(client_id))


def load_cda_state(client_id):
    """
    Persistent per-client CDA state across rounds.
    """
    path = get_cda_state_path(client_id)
    if not os.path.exists(path):
        return {
            "initialized": False,
            "Q_conf": [],
            "Q_x": [],
            "Q_y": [],
            "L_x": [],
            "L_y": [],
            "pending_x": [],
            "pending_y": [],
            "drift_events": 0,
        }

    with open(path, "rb") as f:
        return pickle.load(f)


def save_cda_state(client_id, state):
    path = get_cda_state_path(client_id)
    with open(path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def min_per_class_required(L_min, num_classes):
    return int(math.ceil(float(L_min) / float(2 * num_classes)))


def class_counts(y, num_classes):
    y = np.asarray(y, dtype=np.int32)
    counts = np.zeros(num_classes, dtype=np.int32)
    if len(y) > 0:
        uniq, cnt = np.unique(y, return_counts=True)
        counts[uniq] = cnt
    return counts


def balanced_enough(y, L_min, num_classes):
    """
    Paper heuristic:
    at least L/(2M) examples from each class for the concept memory,
    where M is the number of classes. :contentReference[oaicite:2]{index=2}
    """
    req = min_per_class_required(L_min, num_classes)
    counts = class_counts(y, num_classes)
    return np.all(counts >= req)


def append_q_sample(state, q_i, x_i, y_i):
    state["Q_conf"].append(float(q_i))
    state["Q_x"].append(np.array(x_i, copy=True))
    state["Q_y"].append(int(y_i))

    if len(state["Q_conf"]) > Q_MAX:
        state["Q_conf"].pop(0)
        state["Q_x"].pop(0)
        state["Q_y"].pop(0)


def clear_q(state):
    state["Q_conf"] = []
    state["Q_x"] = []
    state["Q_y"] = []


def beta_moment_params(q, eps=1e-6):
    """
    Estimate Beta(alpha, beta) by method of moments.
    """
    q = np.asarray(q, dtype=np.float64)
    m = float(np.mean(q))
    v = float(np.var(q))

    m = min(max(m, eps), 1.0 - eps)
    v = max(v, eps)

    common = (m * (1.0 - m) / v) - 1.0
    if common <= 0:
        alpha = max(m * 10.0, eps)
        beta = max((1.0 - m) * 10.0, eps)
    else:
        alpha = max(m * common, eps)
        beta = max((1.0 - m) * common, eps)

    return alpha, beta


def log_beta_pdf(x, a, b, eps=1e-8):
    x = np.clip(x, eps, 1.0 - eps)
    return (
        (a - 1.0) * np.log(x)
        + (b - 1.0) * np.log(1.0 - x)
        - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    )


def drift_detection(Q_conf, lam=LAMBDA_CDA, delta=DELTA_CDA):
    """
    CDA-Fed drift detection (Algorithm 5 approximation faithful to pseudocode):
      1. split Q at each k in [Δ, N-Δ]
      2. require a negative mean change ma <= (1-lambda)*mb
      3. estimate Beta params for older/newer subwindows
      4. sum log-likelihood ratios
      5. detect drift if sf > Th = -log(lambda)

    Returns:
        drift_found (bool), k_max (int or None), sf_best (float)
    """
    Q_conf = np.asarray(Q_conf, dtype=np.float64)
    N = len(Q_conf)

    if N < 2 * delta:
        return False, None, 0.0

    sf = 0.0
    k_max = None
    Th = -math.log(lam)

    for k in range(delta, N - delta + 1):
        Qb = Q_conf[:k]      # older
        Qa = Q_conf[k:]      # newer

        mb = float(np.mean(Qb))
        ma = float(np.mean(Qa))

        # Only negative-direction changes:
        # ma <= (1-lambda) * mb  :contentReference[oaicite:3]{index=3}
        if ma <= (1.0 - lam) * mb:
            sk = 0.0
            a_b, b_b = beta_moment_params(Qb)
            a_a, b_a = beta_moment_params(Qa)

            for q_i in Qa:
                sk += log_beta_pdf(q_i, a_b, b_b) - log_beta_pdf(q_i, a_a, b_a)

            if sk > sf:
                sf = sk
                k_max = k

    drift_found = sf > Th and k_max is not None
    return drift_found, k_max, sf


def merge_new_concept_into_long_memory(state, X_new, Y_new):
    """
    Long-term memory L <- L union Lnew
    """
    for x_i, y_i in zip(X_new, Y_new):
        state["L_x"].append(np.array(x_i, copy=True))
        state["L_y"].append(int(y_i))


def local_train_on_memory(
    sess,
    train_op,
    x_ph,
    y_ph,
    sample_w_ph,
    X_train,
    Y_train,
    batch_size,
    local_rounds,
    local_epochs,
    class_weight_mode="balanced",
):
    """
    Implements local training over the long-term memory L
    for R rounds and E epochs per round (Algorithm 6 style). :contentReference[oaicite:4]{index=4}
    """
    X_train = np.asarray(X_train, dtype=np.float32)
    Y_train = np.asarray(Y_train, dtype=np.int32)

    if len(X_train) == 0:
        return

    for _ in range(local_rounds):
        for _ in range(local_epochs):
            for xb, yb in batch_iter(X_train, Y_train, batch_size, shuffle=True):
                wb = compute_sample_weights(yb, class_weight_mode=class_weight_mode)
                sess.run(train_op, feed_dict={x_ph: xb, y_ph: yb, sample_w_ph: wb})


def drift_adaptation(
    state,
    sess,
    train_op,
    x_ph,
    y_ph,
    sample_w_ph,
    train_batchsize,
    num_classes,
    local_rounds,
    local_epochs,
):
    """
    Algorithm 6:
      1) collect enough new data on new concept
      2) L <- L union Lnew
      3) train locally on L

    Here, "collectData()" is implemented via state["pending_*"] accumulated
    from the stream after drift / startup. :contentReference[oaicite:5]{index=5}
    """
    pending_y = np.asarray(state["pending_y"], dtype=np.int32)

    if len(pending_y) == 0:
        return False

    if not balanced_enough(pending_y, MIN_TRAIN_DATA, num_classes):
        return False

    X_new = np.asarray(state["pending_x"], dtype=np.float32)
    Y_new = pending_y

    merge_new_concept_into_long_memory(state, X_new, Y_new)

    X_mem = np.asarray(state["L_x"], dtype=np.float32)
    Y_mem = np.asarray(state["L_y"], dtype=np.int32)

    local_train_on_memory(
        sess=sess,
        train_op=train_op,
        x_ph=x_ph,
        y_ph=y_ph,
        sample_w_ph=sample_w_ph,
        X_train=X_mem,
        Y_train=Y_mem,
        batch_size=train_batchsize,
        local_rounds=local_rounds,
        local_epochs=local_epochs,
        class_weight_mode="balanced",
    )

    state["pending_x"] = []
    state["pending_y"] = []
    state["initialized"] = True
    return True


# =========================================================
# Main client
# =========================================================
def synclass1_agent_cdafed(
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
    lr=None
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
    NUM_CLASSES = getattr(gv, "NUM_CLASSES", NUM_CLASSES_DEFAULT)

    if lr is None:
        lr = getattr(args, "eta", DEFAULT_LR)

    print('Agent %s on GPU %s' % (CURRENT_AGENT, gpu_id))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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
    num_steps = int(math.ceil(float(batch_size) / float(train_batchsize)))

    # ---------------------------------------------------------
    # Graph
    # ---------------------------------------------------------
    x = tf.placeholder(shape=[None, gv.DATA_DIM], dtype=tf.float32, name="x")
    y = tf.placeholder(shape=[None], dtype=tf.int32, name="y")
    sample_w = tf.placeholder(tf.float32, shape=[None], name="sample_w")
    global_step = tf.Variable(0, trainable=False, name="global_step")

    logits_train = agent_model(x, training=True)
    logits_eval  = agent_model(x, training=False)

    probs_eval = tf.nn.softmax(logits_eval, axis=1)
    max_conf = tf.reduce_max(probs_eval, axis=1)

    per_example_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=y,
      logits=logits_train
    )


    eps = 1e-8
    weighted_loss = tf.reduce_sum(sample_w * per_example_loss) / (
        tf.reduce_sum(sample_w) + eps
    )

    # CDA-FedAvg uses plain SGD-style local updates, not EMA-reset logic. :contentReference[oaicite:6]{index=6}
    lr_var = tf.Variable(float(DEFAULT_LR), trainable=False, dtype=tf.float32, name="lr")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr_var)
    train_op = optimizer.minimize(weighted_loss, global_step=global_step)

    sess.run(tf.compat.v1.global_variables_initializer())
    agent_model.set_weights(theta)

    # ---------------------------------------------------------
    # Persistent CDA state
    # ---------------------------------------------------------
    state = load_cda_state(CURRENT_AGENT)
    state["Q_conf"].clear()
    state["Q_x"].clear()
    state["Q_y"].clear()


    print("Num training steps: {}".format(num_steps))

    start_offset = 0
    cda_drift_this_round = False

    # ---------------------------------------------------------
    # Stream processing within this client round
    # ---------------------------------------------------------
    for step in range(num_steps):
        if start_offset >= batch_size:
            break

        end_offset = min(start_offset + train_batchsize, batch_size)
        X_batch = np.asarray(x_batch[start_offset:end_offset], dtype=np.float32)
        Y_batch = np.asarray(y_batch[start_offset:end_offset], dtype=np.int32)
        wb = compute_sample_weights(Y_batch, class_weight_mode="balanced")
        sess.run(train_op, feed_dict={x: X_batch, y: Y_batch, sample_w: wb})
        # 1) Observe new instances and obtain confidences before any update on them
        conf_vals = sess.run(max_conf, feed_dict={x: X_batch})

        for i in range(len(X_batch)):
            x_i = X_batch[i]
            y_i = int(Y_batch[i])
            q_i = float(conf_vals[i])
            

            # Algorithm 4: predict -> add confidence to Q -> maybe run detection
            append_q_sample(state, q_i, x_i, y_i)

            # Keep candidate new-concept data
            state["pending_x"].append(np.array(x_i, copy=True))
            state["pending_y"].append(y_i)

            # Run drift detection with probability exp(-2 q_i)  :contentReference[oaicite:7]{index=7}
            r = random.random()
            if len(state["Q_conf"]) >= MIN_SAMPLES_BEFORE_DETECTION:
             if r < math.exp(-2.0 * q_i):
                drift_found, k_max, sf = drift_detection(
                    state["Q_conf"],
                    lam=LAMBDA_CDA,
                    delta=DELTA_CDA
                )

                if drift_found and not cda_drift_this_round:
                    cda_drift_this_round = True
                    state["drift_events"] += 1

                    print(
                        "CDA drift detected in client {} at local step {}, "
                        "k_max={}, score={:.4f}".format(
                            CURRENT_AGENT, step, k_max, sf
                        )
                    )

                    # All data after the cut-off can immediately belong to the new concept. :contentReference[oaicite:8]{index=8}
                    q_tail_x = state["Q_x"][k_max:]
                    q_tail_y = state["Q_y"][k_max:]

                    # Reset pending to focus on the new concept after detected cut-off
                    state["pending_x"] = [np.array(v, copy=True) for v in q_tail_x]
                    state["pending_y"] = [int(v) for v in q_tail_y]

                    # Clear Q after drift event to start tracking the new regime
                    clear_q(state)

                    # Try adaptation immediately if enough balanced data exists
                    drift_adaptation(
                        state=state,
                        sess=sess,
                        train_op=train_op,
                        x_ph=x,
                        y_ph=y,
                        sample_w_ph=sample_w,
                        train_batchsize=train_batchsize,
                        num_classes=NUM_CLASSES,
                        local_rounds=LOCAL_ROUNDS_PER_CHANGE,
                        local_epochs=LOCAL_EPOCHS_PER_ROUND,
                    )

        # 2) Startup case:
        # learn the first concept once enough balanced data has been collected
        if not state["initialized"]:
            drift_adaptation(
                state=state,
                sess=sess,
                train_op=train_op,
                x_ph=x,
                y_ph=y,
                sample_w_ph=sample_w,
                train_batchsize=train_batchsize,
                num_classes=NUM_CLASSES,
                local_rounds=LOCAL_ROUNDS_PER_CHANGE,
                local_epochs=LOCAL_EPOCHS_PER_ROUND,
            )

        start_offset = end_offset

    # Optional safeguard:
    # if already initialized and enough fresh pending data exists, you may adapt again.
    # This is useful in short rounds where a drift is detected late.
    if len(state["pending_y"]) > 0 and balanced_enough(
        np.asarray(state["pending_y"], dtype=np.int32),
        MIN_TRAIN_DATA,
        NUM_CLASSES
    ):
        drift_adaptation(
            state=state,
            sess=sess,
            train_op=train_op,
            x_ph=x,
            y_ph=y,
            sample_w_ph=sample_w,
            train_batchsize=train_batchsize,
            num_classes=NUM_CLASSES,
            local_rounds=LOCAL_ROUNDS_PER_CHANGE,
            local_epochs=LOCAL_EPOCHS_PER_ROUND,
        )

    # ---------------------------------------------------------
    # Final weights, eval, and bookkeeping
    # ---------------------------------------------------------
    local_weights = agent_model.get_weights()
    local_delta = local_weights - shared_weights

    eval_success, eval_loss = eval_minimal(X_test, Y_test, local_weights)

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
    driftstr = "cda_{}".format(state["drift_events"]) if state["drift_events"] > 0 else ""

    results_dict[client_str] = {
        "t": round_idx,
        "i": CURRENT_AGENT,
        "eval_success": eval_success,
        "eval_loss": eval_loss,
        "drift": driftstr,
        "delayed": delayedclient
    }

    print('Agent {}: success {}, loss {}'.format(CURRENT_AGENT, eval_success, eval_loss))

    return_dict[str(CURRENT_AGENT)] = np.array(local_delta)
    return_dict["theta{}".format(CURRENT_AGENT)] = np.array(local_weights)
    return_dict[str(CURRENT_AGENT) + "_num_samples"] = batch_size
    return_dict[str(CURRENT_AGENT) + "_time"] = time.time()

    np.save(gv.dir_name + 'ben_delta_%s_t%s.npy' % (CURRENT_AGENT, round_idx), local_delta)

    # Persist CDA state for this client across rounds
    save_cda_state(CURRENT_AGENT, state)

    return