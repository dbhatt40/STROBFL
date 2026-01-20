
#########################
# Purpose: Useful functions for evaluating a model on test data
########################
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
# tf.set_random_seed(777)
# np.random.seed(777)
# import keras.backend as K
from keras.utils import np_utils


from .census_utils import census_model_1
from .gas_sensor_utils import uci_sensor_model
from .air_quality_utils import airquality_model
import global_vars as gv

from .io_utils import file_write
from collections import OrderedDict
from .synclass1_utils import synclass1_model
from sklearn.preprocessing import StandardScaler

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)

def eval_setup(global_weights):
    args = gv.args

 
    # global_weights_np = np.load(gv.dir_name + 'global_weights_t%s.npy' % t)
    global_weights_np = global_weights

    if args.dataset == 'synthetic-class1':
        global_model = synclass1_model()
        x = tf.placeholder(shape=[None, gv.DATA_DIM], dtype=tf.float32, name="x")
        y = tf.placeholder(shape=[None],dtype=tf.int64, name="y")
        logits = global_model(x)
        prediction = tf.nn.softmax(logits)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits))
    elif args.dataset == 'air-quality':
        global_model = airquality_model()
        x = tf.placeholder(shape=(None, gv.DATA_DIM), dtype=tf.float32)
        y = tf.placeholder(shape=(None,gv.NUM_CLASSES), dtype=tf.float32)
        logits = global_model(x)
        prediction = logits
        # print("In Eval functionLabels shape and Logits", logits.shape, y.shape)
        loss = tf.reduce_mean(tf.losses.mean_squared_error(y, logits))

    if args.k > 1:
        config = tf.ConfigProto(gpu_options=gv.gpu_options)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        sess = tf.Session(config=config)
    elif args.k == 1:
        sess = tf.Session()
    
    tf.keras.backend.set_session(sess)
    sess.run(tf.global_variables_initializer())

    global_model.set_weights(global_weights_np)
    has_nan = any(np.isnan(a).any() for a in global_weights_np)
    has_inf = any(np.isinf(a).any() for a in global_weights_np)


    return x, y, sess, prediction, loss

def eval_minimal(X_test, Y_test, global_weights, return_dict=None, y_scaler=None):
    args = gv.args
    # print("[agent] y_scaler is None?", y_scaler is None)

    x, y, sess, prediction, loss = eval_setup(global_weights)

    num_samples = len(X_test)
    num_batches = int(np.ceil(num_samples / gv.BATCH_SIZE))
    total_count = 0
    eval_loss = 0.0

    # allocate predictions
    if args.dataset == "air-quality":
        pred_np = np.zeros((num_samples, 1), dtype=np.float32)   # regression
    else:
        pred_np = np.zeros((num_samples, gv.NUM_CLASSES), dtype=np.float32)  # classification

    for i in range(num_batches):
        start = i * gv.BATCH_SIZE
        end   = min((i + 1) * gv.BATCH_SIZE, num_samples)

        Xs = X_test[start:end]
        Ys = Y_test[start:end]

        # finite mask
        x_ok = np.isfinite(Xs).all(axis=1)
        y_ok = np.isfinite(Ys.reshape(-1))   # works for (B,) or (B,1)
        ok = x_ok & y_ok

        if not ok.any():
            continue

        if args.dataset == "air-quality":
            X_feed = Xs[ok].astype(np.float32)
            Y_feed = Ys[ok].astype(np.float32).reshape(-1, 1)
        else:
            X_feed = Xs[ok].astype(np.float32)
            Y_feed = Ys.reshape(-1)[ok].astype(np.int64)

        loss_val, pred_i = sess.run(
            [loss, prediction],
            feed_dict={x: X_feed, y: Y_feed}
        )

        valid_bs = X_feed.shape[0]
        total_count += valid_bs
        eval_loss += float(loss_val) * valid_bs

        # store preds
        if args.dataset == "air-quality":
            pred_np[start:end, 0][ok] = pred_i.reshape(-1).astype(np.float32)
        else:
            pred_np[start:end, :][ok, :] = pred_i.astype(np.float32)

    eval_loss = eval_loss / total_count if total_count > 0 else float("nan")
    # print(f"Eval loss {eval_loss} total_count {total_count}")
    sess.close()

    # ---- compute success metric ----
    if args.dataset == "air-quality":
        y_true_s = np.asarray(Y_test, dtype=np.float32).reshape(-1)
        y_pred_s = np.asarray(pred_np, dtype=np.float32).reshape(-1)

        x_ok = np.isfinite(X_test).all(axis=1)
        mask = x_ok & np.isfinite(y_true_s) & np.isfinite(y_pred_s)

        if mask.sum() == 0:
            mse = float("nan")
            sse = float("nan")
            sst = float("nan")
            r2  = float("nan")
        else:
            if y_scaler is not None:
                y_v = y_scaler.inverse_transform(y_true_s[mask].reshape(-1, 1)).reshape(-1)
                p_v = y_scaler.inverse_transform(y_pred_s[mask].reshape(-1, 1)).reshape(-1)
            else:
                y_v = y_true_s[mask]
                p_v = y_pred_s[mask]

            diff = p_v - y_v
            mse = float(np.mean(diff * diff))
            sse = float(np.sum(diff * diff))

            y_mean = float(np.mean(y_v))
            sst = float(np.sum((y_v - y_mean) ** 2))
            r2 = 1.0 - sse / sst if sst > 0 else float("nan")

        # print(f"Valid eval rows: {int(mask.sum())}/{len(mask)}")
        print(f"MSE {mse}; SSE {sse}; R2 {r2};")

        eval_success = r2  # ✅ IMPORTANT
    else:
        y_true = np.asarray(Y_test, dtype=np.int64).reshape(-1)
        pred_logits = np.asarray(pred_np, dtype=np.float32)
        y_pred = np.argmax(pred_logits, axis=1)
        eval_success = 100.0 * float(np.mean(y_pred == y_true))

    if return_dict is not None:
        return_dict["success_thresh"] = eval_success

    return eval_success, eval_loss


def eval_func(X_test, Y_test, t, return_dict, y_scaler=None, global_weights=None):

    # if global_weights is None:
    #     global_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t)

    eval_success, eval_loss = eval_minimal(X_test, Y_test, global_weights, return_dict=None, y_scaler=y_scaler)

    print('*****Iteration {}: validation accuracy {}, loss {} ******'.format(t, eval_success, eval_loss))
    write_dict = OrderedDict()
    write_dict['t'] = t
    write_dict['eval_success'] = eval_success
    write_dict['eval_loss'] = eval_loss
    file_write(write_dict)

    return_dict['eval_success'] = eval_success
    return_dict['eval_loss'] = eval_loss

   

    return
