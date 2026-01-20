# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 16:06:49 2026

@author: Divya
"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior()

def svrg_client_learn_tf1(
    sess,
    agent_model,
    X_batch,
    Y_batch,
    *,
    data_dim,
    num_classes,
    lr=1e-1,
    buffer_size=2048,
    refresh_every=50,
    mu_batch_size=256,
    clip_norm=1.0,
    reset_state=False,
    return_f1=True,
    # ---- class-weight options ----
    class_weight_mode="inv_sqrt_freq",   # "inv_freq" | "inv_sqrt_freq" | "effective_num"
    effective_beta=0.999,                # for "effective_num"
    weight_clip=(0.5, 5.0),              # clip class weights for stability
    weight_smoothing=0.7,                # EMA smoothing on class weights (0..1)
):
    """
    SVRG (streaming/buffered) for TF1 + tf.keras.Model that outputs logits.

    Implements true SVRG control variate:
        v = g_cur(w) - g_snap(w_tilde) + mu(w_tilde)
        w <- w - lr * v

    And uses class-weighted cross-entropy CONSISTENTLY for:
        - g_cur
        - g_snap
        - mu

    Buffer:
        Maintains a ring buffer of (x,y) samples.
        mu is computed over the buffer at the snapshot weights w_tilde.

    Returns:
        (loss_val, f1_val_or_None) measured on the CURRENT input batch (before update).
    """

    # ---------------- helpers ----------------
    def _compute_class_weights_from_counts(counts: np.ndarray) -> np.ndarray:
        
      if class_weight_mode == "None":
            cw = np.ones((num_classes,), dtype=np.float32)
      else:
         counts = counts.astype(np.float32)
         counts = np.maximum(counts, 1.0)

         if class_weight_mode == "inv_freq":
             cw = 1.0 / counts
         elif class_weight_mode == "inv_sqrt_freq":
             cw = 1.0 / np.sqrt(counts)
         elif class_weight_mode == "effective_num":
             beta = float(effective_beta)
             cw = (1.0 - beta) / (1.0 - np.power(beta, counts))
         else:
            raise ValueError(f"Unknown class_weight_mode: {class_weight_mode}")

      cw = cw / np.mean(cw)  # normalize mean weight ~ 1
      lo, hi = weight_clip
      cw = np.clip(cw, float(lo), float(hi)).astype(np.float32)
      return cw

    # ---------------- build graph ops once per (graph, model) ----------------
    g = tf.get_default_graph()
    cache = getattr(svrg_client_learn_tf1, "_cache", {})
    key = (id(g), id(agent_model))

    if key not in cache:
        # Placeholders
        x_ph  = tf.placeholder(tf.float32, shape=[None, data_dim],   name="svrg_x")
        y_ph  = tf.placeholder(tf.int32,   shape=[None],            name="svrg_y")
        lr_ph = tf.placeholder(tf.float32, shape=[],                name="svrg_lr")
        cw_ph = tf.placeholder(tf.float32, shape=[num_classes],     name="svrg_class_w")

        # Forward pass on PASSED model
        logits = agent_model(x_ph, training=True)

        # Weighted loss
        per_ex = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_ph, logits=logits)  # [B]
        w_per_ex = tf.gather(cw_ph, y_ph)  # [B]
        eps = tf.constant(1e-8, tf.float32)
        loss = tf.reduce_sum(w_per_ex * per_ex) / (tf.reduce_sum(w_per_ex) + eps)

        # Gradients wrt model vars
        vars_cur = agent_model.trainable_variables
        grads = tf.gradients(loss, vars_cur)
        grads = [tf.zeros_like(v) if gg is None else gg for gg, v in zip(grads, vars_cur)]

        # Pack/unpack gradient vectors
        shapes = [v.shape.as_list() for v in vars_cur]
        sizes  = [int(np.prod(s)) for s in shapes]
        P = int(np.sum(sizes))

        def pack(ts):
            return tf.concat([tf.reshape(t, [-1]) for t in ts], axis=0)  # [P]

        def unpack(vec):
            outs, off = [], 0
            for sz, shp in zip(sizes, shapes):
                outs.append(tf.reshape(vec[off:off+sz], shp))
                off += sz
            return outs

        g_vec = pack(grads)  # [P]

        # Placeholder for SVRG direction (computed in python)
        vr_ph = tf.placeholder(tf.float32, shape=[P], name="svrg_vr")

        vr_use = vr_ph
        if clip_norm is not None:
            vr_use, _ = tf.clip_by_global_norm([vr_use], clip_norm)
            vr_use = vr_use[0]

        vr_unpacked = unpack(vr_use)

        # Apply update directly to model variables
        apply_op = tf.group(*[
            v.assign_sub(lr_ph * dv) for v, dv in zip(vars_cur, vr_unpacked)
        ])

        # Optional macro-F1 (for logging only)
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
            f1_macro = tf.reduce_mean(tf.stack(f1s))
        else:
            f1_macro = None

        cache[key] = {
            "x": x_ph, "y": y_ph, "lr": lr_ph, "cw": cw_ph,
            "loss": loss, "f1": f1_macro,
            "g_vec": g_vec, "P": P,
            "vr": vr_ph, "apply": apply_op,
        }
        svrg_client_learn_tf1._cache = cache

        # Python-side state (buffer + snapshot + mu + cw EMA)
        state = getattr(svrg_client_learn_tf1, "_state", {})
        state[key] = {
            "step": 0,
            "buf_x": np.zeros((buffer_size, data_dim), dtype=np.float32),
            "buf_y": np.zeros((buffer_size,), dtype=np.int32),
            "buf_ptr": 0,
            "buf_count": 0,
            "w_tilde": None,                    # snapshot weights (list of arrays)
            "mu": None,                         # np array [P]
            "cw_ema": np.ones((num_classes,), dtype=np.float32),
        }
        svrg_client_learn_tf1._state = state

    h = svrg_client_learn_tf1._cache[key]
    st = svrg_client_learn_tf1._state[key]

    # ---------------- reset state ----------------
    if reset_state:
        st["step"] = 0
        st["buf_ptr"] = 0
        st["buf_count"] = 0
        st["w_tilde"] = None
        st["mu"] = None
        st["cw_ema"] = np.ones((num_classes,), dtype=np.float32)

    # ---------------- ingest batch into buffer ----------------
    Xb = np.asarray(X_batch, dtype=np.float32)
    yb = np.asarray(Y_batch, dtype=np.int32).reshape(-1)

    # (optional) guard: ensure labels in range
    if yb.size > 0:
        if yb.min() < 0 or yb.max() >= num_classes:
            raise ValueError(f"Labels out of range: min={yb.min()} max={yb.max()} num_classes={num_classes}")

    for i in range(Xb.shape[0]):
        st["buf_x"][st["buf_ptr"]] = Xb[i]
        st["buf_y"][st["buf_ptr"]] = yb[i]
        st["buf_ptr"] = (st["buf_ptr"] + 1) % st["buf_x"].shape[0]
        st["buf_count"] = min(st["buf_count"] + 1, st["buf_x"].shape[0])

    # ---------------- compute class weights from buffer (stable) ----------------
    nbuf = st["buf_count"]
    if nbuf > 0:
        counts = np.bincount(st["buf_y"][:nbuf].astype(np.int32), minlength=num_classes)
        cw = _compute_class_weights_from_counts(counts)
    else:
        cw = np.ones((num_classes,), dtype=np.float32)

    # smooth class weights (EMA)
    alpha = float(weight_smoothing)
    st["cw_ema"] = (1.0 - alpha) * cw + alpha * st["cw_ema"]
    cw_use = st["cw_ema"].astype(np.float32)

    # ---------------- snapshot refresh + mu computation ----------------
    do_refresh = (st["w_tilde"] is None) or (st["step"] % int(refresh_every) == 0)

    if do_refresh:
        # snapshot weights
        st["w_tilde"] = agent_model.get_weights()

        # compute mu at snapshot weights over buffer
        P = h["P"]
        mu_acc = np.zeros((P,), dtype=np.float32)

        if nbuf > 0:
            # Save current weights, switch to snapshot, compute grads, restore
            w_cur_save = agent_model.get_weights()
            agent_model.set_weights(st["w_tilde"])

            X_buf = st["buf_x"][:nbuf]
            y_buf = st["buf_y"][:nbuf]

            for start in range(0, nbuf, int(mu_batch_size)):
                end = min(start + int(mu_batch_size), nbuf)
                g_chunk = sess.run(
                    h["g_vec"],
                    feed_dict={h["x"]: X_buf[start:end], h["y"]: y_buf[start:end], h["cw"]: cw_use}
                ).astype(np.float32)
                mu_acc += g_chunk * float(end - start)

            agent_model.set_weights(w_cur_save)
            st["mu"] = mu_acc / float(nbuf)
        else:
            st["mu"] = mu_acc

    # ---------------- SVRG step: g_cur, g_snap, v, apply ----------------
    # g_cur at current weights on current batch
    g_cur = sess.run(
        h["g_vec"],
        feed_dict={h["x"]: Xb, h["y"]: yb, h["cw"]: cw_use}
    ).astype(np.float32)

    # g_snap at snapshot weights on same batch
    w_cur_save = agent_model.get_weights()
    agent_model.set_weights(st["w_tilde"])
    g_snap = sess.run(
        h["g_vec"],
        feed_dict={h["x"]: Xb, h["y"]: yb, h["cw"]: cw_use}
    ).astype(np.float32)
    agent_model.set_weights(w_cur_save)

    vr = g_cur - g_snap + st["mu"]

    # logging on current batch (before update)
    if h["f1"] is not None:
        loss_val, f1_val = sess.run(
            [h["loss"], h["f1"]],
            feed_dict={h["x"]: Xb, h["y"]: yb, h["cw"]: cw_use}
        )
    else:
        loss_val = sess.run(h["loss"], feed_dict={h["x"]: Xb, h["y"]: yb, h["cw"]: cw_use})
        f1_val = None

    # apply update
    sess.run(
        h["apply"],
        feed_dict={h["vr"]: vr, h["lr"]: float(lr)}
    )

    st["step"] += 1
    return float(loss_val), (float(f1_val) if f1_val is not None else None)


def svrg_client_learn_tf1_regression(
    sess,
    agent_model,
    X_batch,
    Y_batch,
    *,
    data_dim,
    lr=1e-3,
    buffer_size=2048,
    refresh_every=50,
    mu_batch_size=256,
    clip_norm=1.0,
    reset_state=False,
    loss_type="mse",          # "mse" | "huber"
    huber_delta=1.0,
    sample_weights=None,      # None or np.ndarray shape [B] or [B,1]
):
    """
    SVRG (streaming/buffered) for TF1 + tf.keras.Model for REGRESSION.

    True SVRG control variate:
        v = g_cur(w) - g_snap(w_tilde) + mu(w_tilde)
        w <- w - lr * v

    Buffer:
        Maintains a ring buffer of (x,y[,w]) samples.
        mu is computed over the buffer at snapshot weights w_tilde.

    Loss:
        - MSE or Huber, optionally sample-weighted (regression analog of class weights).

    Returns:
        (loss_val, mse_val) measured on the CURRENT input batch (before update).
    """
    import numpy as np
    import tensorflow as tf

    # ---------------- build graph ops once per (graph, model) ----------------
    g = tf.compat.v1.get_default_graph()
    cache = getattr(svrg_client_learn_tf1_regression, "_cache", {})
    key = (id(g), id(agent_model), str(loss_type).lower())

    def _per_example_loss_tf(pred_2d, y_2d):
        # pred_2d, y_2d: [B,1]
        if str(loss_type).lower() == "huber":
            r = tf.squeeze(pred_2d - y_2d, axis=1)  # [B]
            abs_r = tf.abs(r)
            delta = tf.constant(float(huber_delta), tf.float32)
            quad = tf.minimum(abs_r, delta)
            lin  = abs_r - quad
            return 0.5 * quad * quad + delta * lin  # [B]
        else:
            r = tf.squeeze(pred_2d - y_2d, axis=1)  # [B]
            return r * r  # [B]

    if key not in cache:
        # Placeholders
        x_ph  = tf.compat.v1.placeholder(tf.float32, shape=[None, data_dim], name="svrg_x")
        y_ph  = tf.compat.v1.placeholder(tf.float32, shape=[None, 1],        name="svrg_y")
        w_ph  = tf.compat.v1.placeholder(tf.float32, shape=[None],           name="svrg_w")  # sample weights
        lr_ph = tf.compat.v1.placeholder(tf.float32, shape=[],               name="svrg_lr")

        # Forward pass on PASSED model
        pred = agent_model(x_ph, training=True)  # expected [B,1]
        pred = tf.cast(pred, tf.float32)

        # Per-example loss
        per_ex = _per_example_loss_tf(pred, y_ph)     # [B]
        eps = tf.constant(1e-8, tf.float32)

        # Weighted mean loss (stable)
        w_sum = tf.reduce_sum(w_ph) + eps
        loss = tf.reduce_sum(w_ph * per_ex) / w_sum

        # Also compute plain MSE for logging (unweighted)
        mse = tf.reduce_mean(tf.square(tf.squeeze(pred - y_ph, axis=1)))

        # Gradients wrt model vars
        vars_cur = agent_model.trainable_variables
        grads = tf.gradients(loss, vars_cur)
        grads = [tf.zeros_like(v) if gg is None else gg for gg, v in zip(grads, vars_cur)]

        # Pack/unpack gradient vectors
        shapes = [v.shape.as_list() for v in vars_cur]
        sizes  = [int(np.prod(s)) for s in shapes]
        P = int(np.sum(sizes))

        def pack(ts):
            return tf.concat([tf.reshape(t, [-1]) for t in ts], axis=0)  # [P]

        def unpack(vec):
            outs, off = [], 0
            for sz, shp in zip(sizes, shapes):
                outs.append(tf.reshape(vec[off:off+sz], shp))
                off += sz
            return outs

        g_vec = pack(grads)  # [P]

        # Placeholder for SVRG direction (computed in python)
        vr_ph = tf.compat.v1.placeholder(tf.float32, shape=[P], name="svrg_vr")

        vr_use = vr_ph
        if clip_norm is not None and float(clip_norm) > 0:
            vr_use_list, _ = tf.clip_by_global_norm([vr_use], float(clip_norm))
            vr_use = vr_use_list[0]

        vr_unpacked = unpack(vr_use)

        # Apply update directly to model variables
        apply_op = tf.group(*[
            v.assign_sub(lr_ph * dv) for v, dv in zip(vars_cur, vr_unpacked)
        ])

        cache[key] = {
            "x": x_ph, "y": y_ph, "w": w_ph, "lr": lr_ph,
            "loss": loss, "mse": mse,
            "g_vec": g_vec, "P": P,
            "vr": vr_ph, "apply": apply_op,
        }
        svrg_client_learn_tf1_regression._cache = cache

        # Python-side state (buffer + snapshot + mu)
        state = getattr(svrg_client_learn_tf1_regression, "_state", {})
        state[key] = {
            "step": 0,
            "buf_x": np.zeros((buffer_size, data_dim), dtype=np.float32),
            "buf_y": np.zeros((buffer_size, 1),       dtype=np.float32),
            "buf_w": np.ones((buffer_size,),          dtype=np.float32),
            "buf_ptr": 0,
            "buf_count": 0,
            "w_tilde": None,   # snapshot weights (list of arrays)
            "mu": None,        # np array [P]
        }
        svrg_client_learn_tf1_regression._state = state

    h = svrg_client_learn_tf1_regression._cache[key]
    st = svrg_client_learn_tf1_regression._state[key]

    # ---------------- reset state ----------------
    if reset_state:
        st["step"] = 0
        st["buf_ptr"] = 0
        st["buf_count"] = 0
        st["w_tilde"] = None
        st["mu"] = None
        st["buf_w"].fill(1.0)

    # ---------------- prepare batch ----------------
    Xb = np.asarray(X_batch, dtype=np.float32)
    yb = np.asarray(Y_batch, dtype=np.float32).reshape(-1, 1)

    if sample_weights is None:
        wb = np.ones((Xb.shape[0],), dtype=np.float32)
    else:
        wb = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
        if wb.shape[0] != Xb.shape[0]:
            raise ValueError(f"sample_weights length {wb.shape[0]} != batch size {Xb.shape[0]}")
        # avoid all-zero weights
        if not np.isfinite(wb).all():
            wb = np.where(np.isfinite(wb), wb, 0.0).astype(np.float32)
        if wb.sum() <= 0:
            wb = np.ones((Xb.shape[0],), dtype=np.float32)

    # ---------------- ingest batch into buffer ----------------
    for i in range(Xb.shape[0]):
        st["buf_x"][st["buf_ptr"]] = Xb[i]
        st["buf_y"][st["buf_ptr"], 0] = yb[i, 0]
        st["buf_w"][st["buf_ptr"]] = wb[i]
        st["buf_ptr"] = (st["buf_ptr"] + 1) % st["buf_x"].shape[0]
        st["buf_count"] = min(st["buf_count"] + 1, st["buf_x"].shape[0])

    nbuf = st["buf_count"]

    # ---------------- snapshot refresh + mu computation ----------------
    do_refresh = (st["w_tilde"] is None) or (st["step"] % int(refresh_every) == 0)

    if do_refresh:
        st["w_tilde"] = agent_model.get_weights()

        P = h["P"]
        mu_acc = np.zeros((P,), dtype=np.float32)

        if nbuf > 0:
            w_cur_save = agent_model.get_weights()
            agent_model.set_weights(st["w_tilde"])

            X_buf = st["buf_x"][:nbuf]
            y_buf = st["buf_y"][:nbuf]
            w_buf = st["buf_w"][:nbuf]

            total_w = float(np.sum(w_buf) + 1e-8)

            for start in range(0, nbuf, int(mu_batch_size)):
                end = min(start + int(mu_batch_size), nbuf)
                g_chunk = sess.run(
                    h["g_vec"],
                    feed_dict={
                        h["x"]: X_buf[start:end],
                        h["y"]: y_buf[start:end],
                        h["w"]: w_buf[start:end],
                    }
                ).astype(np.float32)

                # weight chunk contribution by sum of weights in chunk
                mu_acc += g_chunk * float(np.sum(w_buf[start:end]))

            agent_model.set_weights(w_cur_save)
            st["mu"] = mu_acc / total_w
        else:
            st["mu"] = mu_acc

    # ---------------- SVRG step: g_cur, g_snap, v, apply ----------------
    # g_cur at current weights on current batch
    g_cur = sess.run(
        h["g_vec"],
        feed_dict={h["x"]: Xb, h["y"]: yb, h["w"]: wb}
    ).astype(np.float32)

    # g_snap at snapshot weights on same batch
    w_cur_save = agent_model.get_weights()
    agent_model.set_weights(st["w_tilde"])
    g_snap = sess.run(
        h["g_vec"],
        feed_dict={h["x"]: Xb, h["y"]: yb, h["w"]: wb}
    ).astype(np.float32)
    agent_model.set_weights(w_cur_save)

    vr = g_cur - g_snap + st["mu"]

    # logging on current batch (before update)
    loss_val, mse_val = sess.run(
        [h["loss"], h["mse"]],
        feed_dict={h["x"]: Xb, h["y"]: yb, h["w"]: wb}
    )

    # apply update
    sess.run(
        h["apply"],
        feed_dict={h["vr"]: vr, h["lr"]: float(lr)}
    )

    st["step"] += 1
    return float(loss_val), float(mse_val)
