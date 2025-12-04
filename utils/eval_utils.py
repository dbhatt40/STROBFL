
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
from synthetic_class1_utils import synthetic_class1_model

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)

def eval_setup(global_weights):
    args = gv.args

 
    # global_weights_np = np.load(gv.dir_name + 'global_weights_t%s.npy' % t)
    global_weights_np = global_weights

    if args.dataset == 'synthetic-class1':
        global_model = synthetic_class1_model()
        x = tf.placeholder(shape=[None, gv.DATA_DIM], dtype=tf.float32, name="x")
        y = tf.placeholder(shape=[None],dtype=tf.int64, name="y")
        logits = global_model(x)
    elif args.dataset == 'air-quality':
        global_model = airquality_model()
        x = tf.placeholder(shape=(None, gv.DATA_DIM), dtype=tf.float32)
        y = tf.placeholder(shape=(None,gv.NUM_CLASSES), dtype=tf.float32)
        logits = global_model(x)


	
    if args.dataset == 'synthetic-class1':
       prediction = tf.nn.softmax(logits)
       loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
           labels=y, logits=logits))
    elif args.dataset == 'air-quality':
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

    return x, y, sess, prediction, loss


def eval_minimal(X_test, Y_test, global_weights, return_dict=None):
    args = gv.args
    # args = gv.args
    print("Shape of x, y test slice:", X_test.shape, Y_test.shape)
    x, y, sess, prediction, loss = eval_setup(global_weights)

    pred_np = np.zeros((len(X_test), gv.NUM_CLASSES))
    eval_loss = 0.0
	
    num_samples = len(X_test)
    num_batches = int(np.ceil(num_samples / gv.BATCH_SIZE))
    total_count = 0
    eval_loss = 0.0    
    for i in range(num_batches):
        start = i * gv.BATCH_SIZE
        end   = min((i + 1) * gv.BATCH_SIZE, num_samples)
        X_test_slice = X_test[start:end]
        Y_test_slice = Y_test[start:end]
        # Y_test_cat_slice = np_utils.to_categorical(Y_test_slice)
        if(args.dataset=='air-quality'):
          Y_test_slice = Y_test_slice.astype('float32').reshape(-1, 1) 
          loss_val, pred_np_i = sess.run([loss,prediction], feed_dict={x: X_test_slice, y: Y_test_slice})
                  # print("Shape of prediction", pred_np_i.shape)
          batch_size = X_test_slice.shape[0]
          total_count += batch_size
          eval_loss += loss_val*batch_size 
          pred_np[start:end,:] = pred_np_i
  
# =============================================================================
#         else:
#          pred_np_i = sess.run(prediction, feed_dict={x: X_test_slice})
#          # print("Shape of predictioni", pred_np_i.shape)       
#          # print("Shape of x, y test slice:", X_test_slice.shape, Y_test_slice.shape)		
#          eval_loss += sess.run(loss,
#                               feed_dict={x: X_test_slice, y: Y_test_slice})
#          pred_np[i * gv.BATCH_SIZE:(i + 1) * gv.BATCH_SIZE, :] = pred_np_i		
#         # print("Shape of prediction", pred_np_i.shape)
#          eval_loss = eval_loss / (len(X_test) / gv.BATCH_SIZE)   
# =============================================================================
    
    eval_loss = eval_loss / total_count if total_count > 0 else float('nan')
    sess.close()

    if(args.dataset=='air-quality'):
         Y_true = Y_test.reshape(-1,1)
         mse = np.mean((pred_np - Y_true)**2)
         y_mean = np.mean(Y_true)
         sse = np.sum((pred_np - Y_true)**2)
         sst = np.sum((Y_true - y_mean) **2)
         r2 = 1.0 - sse/sst if sst>0 else float('nan') 		    
         eval_success = 100.0 * r2            
    else:
         eval_success = 100.0 * \
              np.sum(np.argmax(pred_np, 1) ==Y_test) / len(Y_test)
			  
    if return_dict is not None:
        return_dict['success_thresh'] = eval_success

    return eval_success, eval_loss


def eval_func(X_test, Y_test, t, return_dict, mal_data_X=None, mal_data_Y=None, global_weights=None):
    args = gv.args 

    # if global_weights is None:
    #     global_weights = np.load(gv.dir_name + 'global_weights_t%s.npy' % t)

    eval_success, eval_loss = eval_minimal(X_test, Y_test, global_weights)

    print('*****Iteration {}: validation accuracy {}, loss {} ******'.format(t, eval_success, eval_loss))
    write_dict = OrderedDict()
    write_dict['t'] = t
    write_dict['eval_success'] = eval_success
    write_dict['eval_loss'] = eval_loss
    file_write(write_dict)

    return_dict['eval_success'] = eval_success
    return_dict['eval_loss'] = eval_loss

   

    return
