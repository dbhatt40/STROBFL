# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 12:02:15 2026

@author: Divya
"""
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
    agent_model,
    X_batch,
    Y_batch,
    *,
    data_dim,
    num_classes,
    lr=1e-1,
    memory_size=2048,
    reset_state=False,
    return_f1=True,
    class_weights=None,          # NEW: None or np.ndarray/list shape [num_classes]
    batch_weighted=False,        # NEW: if True and class_weights is None, compute weights from batch
    weight_power=1.0,            # NEW: only used for batch_weighted; 1.0=inv-freq, 0.5=inv-sqrt
):
    """
    STRSAGA client learning (TF1 sess.run style) for synclass1_model-like MLP.

    - Builds graph & state (gradient memory) on first call and caches them.
    - Each call performs ONE STRSAGA update using (X_batch, Y_batch).
    - Uses a bounded ring buffer of stored per-example gradients (size = memory_size).

    Class-weighted extension:
      - If class_weights is provided: uses those fixed weights (recommended).
      - Else if batch_weighted=True: computes inverse-frequency weights from current batch.
      - Else: unweighted (original behavior).

    Args:
      sess: tf.compat.v1.Session
      agent_model: callable Keras model, called as agent_model(x, training=True/False)
      X_batch: np.ndarray [B, data_dim], float32/float64
      Y_batch: np.ndarray [B], int (0..num_classes-1)
      data_dim: feature dimension
      num_classes: number of classes
      lr: learning rate
      memory_size: SAGA memory size M
      reset_state: if True, zero out SAGA memory (g_mem, g_sum, ptr, count)
      return_f1: if True, returns batch macro-F1 (noisy; logging only)
      class_weights: None or array-like [C] float; fixed weights to apply
      batch_weighted: if True and class_weights is None, compute inv-freq weights from the batch
      weight_power: power for inv-freq: w = 1 / freq^power; 1.0 inv-freq, 0.5 inv-sqrt

    Returns:
      loss_val: float
      f1_val: float (if return_f1=True else None)
    """

    # ---------- Build once & cache ----------
    if not hasattr(strsaga_client_learn_tf1, "_cache"):

        with tf.compat.v1.name_scope("strsaga"):
            x_ph = tf.compat.v1.placeholder(tf.float32, shape=[None, data_dim], name="x")
            y_ph = tf.compat.v1.placeholder(tf.int32,   shape=[None],          name="y")
            lr_ph = tf.compat.v1.placeholder(tf.float32, shape=[],             name="lr")

            # fixed class weights placeholder (optional feed)
            cw_ph = tf.compat.v1.placeholder_with_default(
                                  tf.ones([num_classes], dtype=tf.float32),
                                  shape=[num_classes],
                                  name="class_weights",
                         )
            use_cw_ph = tf.compat.v1.placeholder_with_default(False, shape=[], name="use_class_weights")

            # choose whether to compute batch weights
            use_batch_w_ph = tf.compat.v1.placeholder_with_default(False, shape=[], name="use_batch_weights")
            w_power_ph = tf.compat.v1.placeholder_with_default(1.0, shape=[], name="weight_power")

            logits = agent_model(x_ph, training=True)
            vars_ = agent_model.trainable_variables  # list of tf.Variable

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

            eps = tf.constant(1e-8, tf.float32)

            # ----- Per-example loss -----
            per_ex_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y_ph, logits=logits
            )  # [B]

            # ----- Build example weights (either fixed class weights, or batch-derived, or all-ones) -----
            def _batch_class_weights():
                # inverse frequency from current batch, with power
                counts = tf.math.bincount(
                    y_ph, minlength=num_classes, maxlength=num_classes, dtype=tf.float32
                )  # [C]
                freqs = counts / (tf.reduce_sum(counts) + eps)
                inv = 1.0 / (tf.pow(freqs + eps, w_power_ph))
                inv = inv / (tf.reduce_mean(inv) + eps)  # normalize mean ~ 1
                return inv

            # Choose class-weight vector:
            #   if use_cw_ph: cw_ph
            #   elif use_batch_w_ph: batch-derived
            #   else: ones
            ones_cw = tf.ones([num_classes], tf.float32)

            cw_vec = tf.cond(
                use_cw_ph,
                lambda: cw_ph,
                lambda: tf.cond(use_batch_w_ph, _batch_class_weights, lambda: ones_cw),
            )  # [C]

            ex_w = tf.gather(cw_vec, y_ph)  # [B]

            # Weighted mean loss (stable scaling)
            batch_loss = tf.reduce_sum(per_ex_loss * ex_w) / (tf.reduce_sum(ex_w) + eps)
            batch_loss = tf.identity(batch_loss, name="train_loss")

            # ----- Per-example grads (weighted): grad_i = ex_w[i] * grad(loss_i) -----
            def grad_for_one(ex):
                xi, yi, wi = ex  # xi: [data_dim], yi: [], wi: []
                xi = tf.expand_dims(xi, 0)  # [1, data_dim]
                yi = tf.expand_dims(yi, 0)  # [1]

                li = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=yi, logits=agent_model(xi, training=True)
                )  # [1]
                li = tf.reshape(li[0], [])  # scalar

                gi = tf.gradients(li, vars_)  # list of grads
                gvec = pack(gi)               # [P]
                gvec = tf.cast(wi, tf.float32) * gvec  # weight this example's gradient
                return gvec

            G = tf.map_fn(
                grad_for_one,
                (x_ph, y_ph, ex_w),
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

            cnt_f = tf.cast(tf.maximum(cnt, 1), tf.float32)
            alpha_bar = g_sum / cnt_f  # [P] (0 if cnt==0, because g_sum==0)

            # ----- One STRSAGA step over the batch using a while_loop -----
            def body(i, vr_acc, ptr_t, cnt_t, g_sum_t, g_mem_t):
                g_i = G[i]              # [P] (already weighted)
                old = g_mem_t[ptr_t]    # [P]

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

            # Functional loop then assign back to variables
            _, vr_vec, ptr_new, cnt_new, g_sum_new, g_mem_new = tf.while_loop(
                cond, body,
                loop_vars=[i0, vr0, ptr, cnt, g_sum, g_mem],
                parallel_iterations=1,
                back_prop=False,
                name="saga_batch_loop",
            )

            vr_vec = vr_vec / tf.cast(tf.maximum(B, 1), tf.float32)  # [P]

            # Optional clip (helps stability)
            vr_vec, _ = tf.clip_by_global_norm([vr_vec], 5.0)
            vr_vec = vr_vec[0]

            # Apply parameter update: w <- w - lr * vr
           # Split vr_vec into per-variable shapes
            vr_list = unpack(vr_vec)  # list with same shapes as vars_

            assign_ops = [v.assign_sub(lr_ph * ghat) for v, ghat in zip(vars_, vr_list)]
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
                "cw": cw_ph, "use_cw": use_cw_ph,
                "use_batch_w": use_batch_w_ph, "w_power": w_power_ph,
                "train_op": train_op,
                "loss": batch_loss,
                "f1": f1_macro,
                "reset": reset_op,
                "vars": vars_,
            }

        # Initialize variables created by this builder (ONLY those uninitialized)
        uninit = sess.run(tf.compat.v1.report_uninitialized_variables())
        if len(uninit) > 0:
            all_vars = tf.compat.v1.global_variables()
            name_to_var = {v.name.split(":")[0]: v for v in all_vars}
            to_init = [name_to_var[n.decode("utf-8")] for n in uninit if n.decode("utf-8") in name_to_var]
            if to_init:
                sess.run(tf.compat.v1.variables_initializer(to_init))

    # ---------- Run ----------
    h = strsaga_client_learn_tf1._cache

    if reset_state:
        sess.run(h["reset"])

    # ensure correct dtypes
    Xb = X_batch.astype(np.float32, copy=False)
    Yb = Y_batch.astype(np.int32, copy=False)

    feed = {
        h["x"]: Xb,
        h["y"]: Yb,
        h["lr"]: float(lr),
    }

    if class_weights is not None:
        cw = np.asarray(class_weights, dtype=np.float32).reshape((num_classes,))
        feed[h["cw"]] = cw
        feed[h["use_cw"]] = True
        feed[h["use_batch_w"]] = False
    else:
        feed[h["use_cw"]] = False
        feed[h["use_batch_w"]] = bool(batch_weighted)
        feed[h["w_power"]] = float(weight_power)

    if h["f1"] is not None:
        loss_val, f1_val, _ = sess.run([h["loss"], h["f1"], h["train_op"]], feed_dict=feed)
        return float(loss_val), float(f1_val)
    else:
        loss_val, _ = sess.run([h["loss"], h["train_op"]], feed_dict=feed)
        return float(loss_val), None
    
    
    
def strsaga_client_learn_tf1_regression(
    sess,
    agent_model,
    X_batch,
    Y_batch,
    *,
    data_dim,
    lr=1e-3,
    memory_size=2048,
    reset_state=False,
    sample_weights=None,      # None or np.ndarray shape [B] (or [B,1])
    loss_type="mse",          # "mse" or "huber"
    huber_delta=1.0,
    clip_norm=5.0,            # global-norm clip on VR gradient vector
):
    """
    STRSAGA client learning (TF1 sess.run style) for REGRESSION.

    - Builds graph & STRSAGA state (gradient memory) on first call and caches them.
    - Each call performs ONE STRSAGA update using (X_batch, Y_batch).
    - Uses a bounded ring buffer of stored per-example gradients (size = memory_size).
    - Supports optional sample weights (regression analog of class weights).
    - Supports MSE or Huber per-example loss.

    Args:
      sess: tf.compat.v1.Session
      agent_model: callable Keras model, called as agent_model(x, training=True/False)
                  Must output shape [None, 1] for regression.
      X_batch: np.ndarray [B, data_dim], float32/float64
      Y_batch: np.ndarray [B] or [B,1], float32/float64
      data_dim: feature dimension
      lr: learning rate
      memory_size: SAGA memory size M
      reset_state: if True, zero out SAGA memory (g_mem, g_sum, ptr, cnt)
      sample_weights: None or array-like [B] or [B,1] float; per-example weights
      loss_type: "mse" or "huber"
      huber_delta: delta for huber loss (only if loss_type="huber")
      clip_norm: clip norm for VR vector (helps stability)

    Returns:
      loss_val: float (weighted mean batch loss)
    """
    import numpy as np
    import tensorflow as tf

    # ---------- Build once & cache ----------
    if not hasattr(strsaga_client_learn_tf1_regression, "_cache"):
        with tf.compat.v1.name_scope("strsaga_reg"):
            x_ph  = tf.compat.v1.placeholder(tf.float32, shape=[None, data_dim], name="x")
            y_ph  = tf.compat.v1.placeholder(tf.float32, shape=[None, 1],        name="y")
            w_ph  = tf.compat.v1.placeholder_with_default(
                tf.ones([tf.shape(x_ph)[0]], dtype=tf.float32),
                shape=[None],
                name="sample_w"
            )
            lr_ph = tf.compat.v1.placeholder(tf.float32, shape=[], name="lr")

            # Model forward
            pred = agent_model(x_ph, training=True)  # expected [B,1]
            pred = tf.cast(pred, tf.float32)

            # Trainable vars
            vars_ = agent_model.trainable_variables

            # ----- Helpers: pack/unpack variable vectors -----
            var_shapes = [v.shape.as_list() for v in vars_]
            var_sizes  = [int(np.prod(s)) for s in var_shapes]
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

            eps = tf.constant(1e-8, tf.float32)

            # ----- Per-example loss -----
            # per_ex_loss: [B]
            if str(loss_type).lower() == "huber":
                # Huber on residuals
                r = pred - y_ph                              # [B,1]
                r = tf.squeeze(r, axis=1)                    # [B]
                abs_r = tf.abs(r)
                delta = tf.constant(float(huber_delta), tf.float32)
                quad = tf.minimum(abs_r, delta)
                lin  = abs_r - quad
                per_ex_loss = 0.5 * quad * quad + delta * lin
            else:
                # MSE per-example
                r = pred - y_ph                               # [B,1]
                per_ex_loss = tf.squeeze(r * r, axis=1)       # [B]

            # Weighted mean loss (stable scaling)
            w_sum = tf.reduce_sum(w_ph) + eps
            batch_loss = tf.reduce_sum(per_ex_loss * w_ph) / w_sum
            batch_loss = tf.identity(batch_loss, name="train_loss")

            # ----- Per-example grads (weighted): grad_i = w[i] * grad(loss_i) -----
            def grad_for_one(ex):
                xi, yi, wi = ex
                xi = tf.expand_dims(xi, 0)  # [1, data_dim]
                yi = tf.expand_dims(yi, 0)  # [1, 1]

                pi = agent_model(xi, training=True)  # [1,1]
                pi = tf.cast(pi, tf.float32)

                ri = tf.squeeze(pi - yi, axis=[0, 1])  # scalar
                if str(loss_type).lower() == "huber":
                    abs_ri = tf.abs(ri)
                    delta = tf.constant(float(huber_delta), tf.float32)
                    quad = tf.minimum(abs_ri, delta)
                    lin  = abs_ri - quad
                    li = 0.5 * quad * quad + delta * lin
                else:
                    li = ri * ri

                gi = tf.gradients(li, vars_)  # list
                gvec = pack(gi)               # [P]
                gvec = tf.cast(wi, tf.float32) * gvec
                return gvec

            # Map: returns [B, P]
            # Need y as [B,1] to pair with x; w as [B]
            G = tf.map_fn(
                grad_for_one,
                (x_ph, y_ph, w_ph),
                dtype=tf.float32,
                name="per_example_grad_vecs",
            )

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

            cnt_f = tf.cast(tf.maximum(cnt, 1), tf.float32)
            alpha_bar = g_sum / cnt_f  # [P]

            # ----- One STRSAGA step over the batch using a while_loop -----
            def body(i, vr_acc, ptr_t, cnt_t, g_sum_t, g_mem_t):
                g_i = G[i]              # [P] (already weighted)
                old = g_mem_t[ptr_t]    # [P]

                # v_i = g_i - old + alpha_bar
                vr_acc = vr_acc + (g_i - old + alpha_bar)

                # Update g_sum and g_mem at ptr_t
                g_sum_t = g_sum_t + (g_i - old)

                # Update g_mem[ptr_t] = g_i
                g_mem_t = tf.tensor_scatter_nd_update(
                    g_mem_t,
                    indices=tf.reshape(ptr_t, [1, 1]),
                    updates=tf.reshape(g_i, [1, P]),
                )

                ptr_t = tf.math.floormod(ptr_t + 1, M)
                cnt_t = tf.minimum(cnt_t + 1, M)
                return i + 1, vr_acc, ptr_t, cnt_t, g_sum_t, g_mem_t

            def cond(i, *_):
                return i < B

            i0  = tf.constant(0, tf.int32)
            vr0 = tf.zeros([P], tf.float32)

            _, vr_vec, ptr_new, cnt_new, g_sum_new, g_mem_new = tf.while_loop(
                cond, body,
                loop_vars=[i0, vr0, ptr, cnt, g_sum, g_mem],
                parallel_iterations=1,
                back_prop=False,
                name="saga_batch_loop",
            )

            vr_vec = vr_vec / tf.cast(tf.maximum(B, 1), tf.float32)  # average over batch

            # Optional clip (helps stability)
            if clip_norm is not None and float(clip_norm) > 0:
                vr_vec_list, _ = tf.clip_by_global_norm([vr_vec], float(clip_norm))
                vr_vec = vr_vec_list[0]

            # Apply parameter update: w <- w - lr * vr
            vr_list = unpack(vr_vec)
            assign_ops = [v.assign_sub(lr_ph * ghat) for v, ghat in zip(vars_, vr_list)]

            state_assign = tf.group(
                g_mem.assign(g_mem_new),
                g_sum.assign(g_sum_new),
                ptr.assign(ptr_new),
                cnt.assign(cnt_new),
                name="saga_state_assign",
            )

            train_op = tf.group(*(assign_ops + [state_assign]), name="train_op_strsaga_reg")

            reset_op = tf.group(
                g_mem.assign(tf.zeros_like(g_mem)),
                g_sum.assign(tf.zeros_like(g_sum)),
                ptr.assign(tf.zeros_like(ptr)),
                cnt.assign(tf.zeros_like(cnt)),
                name="saga_reset_op",
            )

            strsaga_client_learn_tf1_regression._cache = {
                "x": x_ph, "y": y_ph, "w": w_ph, "lr": lr_ph,
                "train_op": train_op,
                "loss": batch_loss,
                "reset": reset_op,
                "vars": vars_,
            }

        # Initialize variables created by this builder (ONLY those uninitialized)
        uninit = sess.run(tf.compat.v1.report_uninitialized_variables())
        if len(uninit) > 0:
            all_vars = tf.compat.v1.global_variables()
            name_to_var = {v.name.split(":")[0]: v for v in all_vars}
            to_init = []
            for n in uninit:
                name = n.decode("utf-8")
                if name in name_to_var:
                    to_init.append(name_to_var[name])
            if to_init:
                sess.run(tf.compat.v1.variables_initializer(to_init))

    # ---------- Run ----------
    h = strsaga_client_learn_tf1_regression._cache

    if reset_state:
        sess.run(h["reset"])

    Xb = np.asarray(X_batch, dtype=np.float32)
    yb = np.asarray(Y_batch, dtype=np.float32).reshape(-1, 1)

    feed = {h["x"]: Xb, h["y"]: yb, h["lr"]: float(lr)}

    if sample_weights is not None:
        wb = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
        # Safety: match batch size
        if wb.shape[0] != Xb.shape[0]:
            raise ValueError(f"sample_weights has length {wb.shape[0]} but batch has {Xb.shape[0]}")
        feed[h["w"]] = wb
    else:
        # default handled by placeholder_with_default (all ones)
        pass

    loss_val, _ = sess.run([h["loss"], h["train_op"]], feed_dict=feed)
    return float(loss_val)

