# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 12:02:15 2026

@author: Divya
"""
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()


def strsaga_client_learn_tf1(
    sess,
    model,
    X_batch,
    Y_batch,
    *,
    data_dim,
    num_classes,
    lr=1e-3,
    memory_size=2048,
    reset_state=False,
    return_f1=True,
):
    """
    STRSAGA client learning (TF1 sess.run style) for synclass1_model-like MLP.

    - Builds graph & state (gradient memory) on first call and caches them.
    - Each call performs ONE STRSAGA update using (X_batch, Y_batch).
    - Uses a bounded ring buffer of stored per-example gradients (size = memory_size).

    Args:
      sess: tf.compat.v1.Session
      X_batch: np.ndarray [B, data_dim], float32/float64
      Y_batch: np.ndarray [B], int (0..num_classes-1)
      data_dim: gv.DATA_DIM
      num_classes: gv.NUM_CLASSES
      lr: learning rate
      memory_size: SAGA memory size M
      reset_state: if True, zero out SAGA memory (g_mem, g_sum, ptr, count)
      return_f1: if True, returns batch macro-F1 (noisy; logging only)

    Returns:
      loss_val: float
      f1_val: float (if return_f1=True else None)
    """

    # ---------- Build once & cache ----------
    if not hasattr(strsaga_client_learn_tf1, "_cache"):
        g = tf.compat.v1.get_default_graph()

        with g.as_default():
            x_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, data_dim], name="x")
            y_ph = tf.compat.v1.placeholder(tf.int32,   shape=[None],          name="y")
            lr_ph = tf.compat.v1.placeholder(tf.float32, shape=[],             name="lr")

            # ----- Model: synclass1_model() -----
            inp = tf.keras.layers.Input(tensor=x_ph, name="main_input")
            x = tf.keras.layers.Dense(32, activation="relu")(inp)
            x = tf.keras.layers.Dense(32, activation="relu")(x)
            logits = tf.keras.layers.Dense(num_classes)(x)  # logits

            model = tf.keras.Model(inputs=inp, outputs=logits)
            vars_ = model.trainable_variables  # list of tf.Variable

            # ----- Helpers: pack/unpack variable vectors -----
            var_shapes = [v.shape.as_list() for v in vars_]
            var_sizes = [int(np.prod(s)) for s in var_shapes]
            P = int(np.sum(var_sizes))

            def pack(grads_list):
                flat = []
                for g_i, v in zip(grads_list, vars_):
                    if g_i is None:
                        g_i = tf.zeros_like(v)
                    flat.append(tf.reshape(g_i, [-1]))
                return tf.concat(flat, axis=0)  # [P]

            def unpack(vec):
                outs = []
                offset = 0
                for sz, shp in zip(var_sizes, var_shapes):
                    outs.append(tf.reshape(vec[offset:offset + sz], shp))
                    offset += sz
                return outs

            # ----- Per-example loss & per-example grads (exact) -----
            # loss_i = sparse_softmax_xent for each example
            per_ex_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y_ph, logits=logits
            )  # [B]
            batch_loss = tf.reduce_mean(per_ex_loss, name="train_loss")

            # Build per-example grad vectors: G is [B, P]
            # map_fn body: takes (xi, yi) -> grad_vec_i
            def grad_for_one(ex):
                xi, yi = ex  # xi: [data_dim], yi: []
                xi = tf.expand_dims(xi, 0)  # [1, data_dim]
                yi = tf.expand_dims(yi, 0)  # [1]
                li = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=yi, logits=model(xi, training=True)
                )  # [1]
                li = tf.reshape(li[0], [])  # scalar
                gi = tf.gradients(li, vars_)  # list of grads
                return pack(gi)  # [P]

            G = tf.map_fn(
                grad_for_one,
                (x_ph, y_ph),
                dtype=tf.float32,
                name="per_example_grad_vecs",
            )  # [B, P]

            B = tf.shape(x_ph)[0]

            # ----- STRSAGA state: ring buffer memory -----
            M = int(memory_size)
            g_mem = tf.compat.v1.get_variable(
                "saga_g_mem", shape=[M, P], dtype=tf.float32,
                initializer=tf.zeros_initializer(), trainable=False
            )
            g_sum = tf.compat.v1.get_variable(
                "saga_g_sum", shape=[P], dtype=tf.float32,
                initializer=tf.zeros_initializer(), trainable=False
            )
            ptr = tf.compat.v1.get_variable(
                "saga_ptr", shape=[], dtype=tf.int32,
                initializer=tf.zeros_initializer(), trainable=False
            )
            cnt = tf.compat.v1.get_variable(
                "saga_cnt", shape=[], dtype=tf.int32,
                initializer=tf.zeros_initializer(), trainable=False
            )

            eps = tf.constant(1e-8, tf.float32)
            cnt_f = tf.cast(tf.maximum(cnt, 1), tf.float32)
            alpha_bar = g_sum / cnt_f  # [P] (0 if cnt==0, because g_sum==0)

            # ----- One STRSAGA step over the batch using a while_loop -----
            def body(i, vr_acc, ptr_t, cnt_t, g_sum_t, g_mem_t):
                g_i = G[i]  # [P]
                old = g_mem_t[ptr_t]  # [P]

                # v_i = g_i - old + alpha_bar
                vr_acc = vr_acc + (g_i - old + alpha_bar)

                # Update g_sum and g_mem at ptr_t
                g_sum_t = g_sum_t + (g_i - old)

                # Scatter update g_mem[ptr_t] = g_i
                g_mem_t = tf.tensor_scatter_nd_update(
                    g_mem_t,
                    indices=tf.reshape(ptr_t, [1, 1]),
                    updates=tf.reshape(g_i, [1, P]),
                )

                # Advance ring pointer + count
                ptr_t = tf.math.floormod(ptr_t + 1, M)
                cnt_t = tf.minimum(cnt_t + 1, M)
                return i + 1, vr_acc, ptr_t, cnt_t, g_sum_t, g_mem_t

            def cond(i, *_):
                return i < B

            i0 = tf.constant(0, tf.int32)
            vr0 = tf.zeros([P], tf.float32)
            # We run loop in "functional" style then assign back to variables
            _, vr_vec, ptr_new, cnt_new, g_sum_new, g_mem_new = tf.while_loop(
                cond, body,
                loop_vars=[i0, vr0, ptr, cnt, g_sum, g_mem],
                parallel_iterations=1,
                back_prop=False,
                name="saga_batch_loop",
            )

            vr_vec = vr_vec / tf.cast(tf.maximum(B, 1), tf.float32)  # [P]

            # Optional clip (helps with stability; safe for training-only usage)
            vr_vec, _ = tf.clip_by_global_norm([vr_vec], 5.0)
            vr_vec = vr_vec[0]

            # Apply parameter update: w <- w - lr * vr
            theta = pack([v for v in vars_])  # pack current vars
            theta_new = theta - lr_ph * vr_vec
            new_vars = unpack(theta_new)

            assign_ops = [v.assign(nv) for v, nv in zip(vars_, new_vars)]
            state_assign = tf.group(
                g_mem.assign(g_mem_new),
                g_sum.assign(g_sum_new),
                ptr.assign(ptr_new),
                cnt.assign(cnt_new),
                name="saga_state_assign",
            )

            train_op = tf.group(*(assign_ops + [state_assign]), name="train_op_strsaga")

            # Batch macro-F1 for logging only (noisy)
            if return_f1:
                pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
                f1s = []
                for c in range(num_classes):
                    yc = tf.equal(y_ph, c)
                    pc = tf.equal(pred, c)
                    tp = tf.reduce_sum(tf.cast(yc & pc, tf.float32))
                    fp = tf.reduce_sum(tf.cast(~yc & pc, tf.float32))
                    fn = tf.reduce_sum(tf.cast(yc & ~pc, tf.float32))
                    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + eps)
                    f1s.append(f1)
                f1_macro = tf.reduce_mean(tf.stack(f1s), name="f1_macro")
            else:
                f1_macro = None

            reset_op = tf.group(
                g_mem.assign(tf.zeros_like(g_mem)),
                g_sum.assign(tf.zeros_like(g_sum)),
                ptr.assign(tf.zeros_like(ptr)),
                cnt.assign(tf.zeros_like(cnt)),
                name="saga_reset_op",
            )

            strsaga_client_learn_tf1._cache = {
                "x": x_ph, "y": y_ph, "lr": lr_ph,
                "train_op": train_op,
                "loss": batch_loss,
                "f1": f1_macro,
                "reset": reset_op,
                "vars": vars_,
            }

        # IMPORTANT: initialize variables created by this builder
        sess.run(tf.compat.v1.variables_initializer(
            tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
        ))

    # ---------- Run ----------
    h = strsaga_client_learn_tf1._cache

    if reset_state:
        sess.run(h["reset"])

    feed = {h["x"]: X_batch, h["y"]: Y_batch, h["lr"]: float(lr)}

    if h["f1"] is not None:
        loss_val, f1_val, _ = sess.run([h["loss"], h["f1"], h["train_op"]], feed_dict=feed)
        return float(loss_val), float(f1_val)
    else:
        loss_val, _ = sess.run([h["loss"], h["train_op"]], feed_dict=feed)
        return float(loss_val), None
