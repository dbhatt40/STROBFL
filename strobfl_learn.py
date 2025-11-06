# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 13:19:29 2025

@author: Divya
"""
import tensorflow as tf
import numpy as np

class Strobfl_learn(tf.compat.v1.train.GradientDescentOptimizer):
    def __init__(self, learning_rate=0.01, update_rule=None, **kw):
        super(Strobfl_learn, self).__init__(learning_rate=learning_rate, **kw)
        self._update_rule = update_rule

    def _lr_for(self, var):
        # Ensure the parent prepared tensors
        if not hasattr(self, "_learning_rate_tensor") or self._learning_rate_tensor is None:
            self._prepare()  # creates self._learning_rate_tensor
        lr_t = self._learning_rate_tensor
        # Match variable dtype (avoids dtype errors on mixed-precision graphs)
        return tf.cast(lr_t, var.dtype.base_dtype)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
		
        pairs = [(g, v) for g, v in grads_and_vars if g is not None]
        if not pairs:
            return tf.no_op(name or "custom_sgd_noop")

        update_ops = []
        for grad, var in pairs:
            if isinstance(grad, tf.IndexedSlices):
                grad = tf.convert_to_tensor(grad)  # densify unless you implement sparse path
            lr_t = self._lr_for(var)

            # Default delta if no custom rule provided
            delta = lr_t * grad if self._update_rule is None \
                    else self._update_rule(grad, var, lr_t, global_step)
     
            update_ops.append(var.assign_sub(delta, use_locking=self._use_locking))
 
        train_op = tf.group(*update_ops, name=(name or "custom_sgd_apply"))
        if global_step is not None:
            with tf.control_dependencies([train_op]):
                return tf.compat.v1.assign_add(global_step, 1)
        return train_op
	
    def minimize(self, loss, global_step=None, var_list=None, name=None):
        grads_and_vars = self.compute_gradients(loss, var_list=var_list)
        return self.apply_gradients(grads_and_vars, global_step=global_step, name=name)


	
def shrink_rule(grad, var, lr_t, global_step):
      wd = 1e-4
      return lr_t * (grad + wd * var)

def gradient_update_rule_factory1(alpha=0.2, name_prefix="grad_avg"):
    """
    Returns an update_rule(grad, var, lr_t, global_step) such that:
      avg_grad_t = ((t-1)*avg_grad_{t-1} + grad) / t
      new_grad   = (1 - alpha)*grad + alpha*avg_grad_t
      delta      = lr_t * new_grad
    """
    slots = {}          # holds average gradient slot per variable
    steps = tf.Variable(0, trainable=False, dtype=tf.int64, name="grad_avg_step")

    def _slot_for(var):
        key = var.ref()
        if key not in slots:
            slots[key] = tf.Variable(
                tf.zeros_like(var),
                trainable=False,
                name=f"{name_prefix}/{var.op.name.replace(':','_')}"
            )
        return slots[key]

    def update_rule(grad, var, lr_t, global_step):
        avg_slot = _slot_for(var)

        # Increment step counter (shared)
        new_step = tf.compat.v1.assign_add(steps, 1)
        step_f = tf.cast(new_step, var.dtype.base_dtype)

        # Running average of all past gradients
        avg_new = tf.compat.v1.assign(
                     avg_slot,
                     ((step_f - 1.0) / step_f) * avg_slot +
                     (1.0 / step_f) * grad
                   )

        # Blended gradient
        with tf.control_dependencies([avg_new]):
            new_grad = (1.0 - alpha) * grad + alpha * avg_new
            delta = lr_t * new_grad
            return tf.identity(delta, name="grad_blend_delta")

    update_rule.avg_slots = slots
    update_rule.step_var = steps
    return update_rule


def gradient_update_rule_factory2(alpha=0.2, name_prefix="grad_ema"):
    """
    Returns an update_rule(grad, var, lr_t, global_step) that:
      m_t = alpha_t * m_{t-1} + (1 - alpha) * grad
      u_t = (1 - mix) * grad + mix * m_t
      delta = lr_t * u_t
   
    """
    # One non-trainable slot per variable, created lazily on first use.
    slots = {}  # maps var.ref() -> tf.Variable (EMA slot)

    def _slot_for(var):
        key = var.ref()
        if key not in slots:
            slots[key] = tf.Variable(
                tf.zeros_like(var), trainable=False,
                name=f"{name_prefix}/{var.op.name.replace(':','_')}"
            )
        return slots[key]

    def update_rule(grad, var, lr_t, global_step):
        # Handle IndexedSlices (sparse) the simple way by densifying.
        # If you have huge embeddings, implement a scatter version instead.
        if isinstance(grad, tf.IndexedSlices):
            grad = tf.convert_to_tensor(grad)

        m = _slot_for(var)
        alpha_t = tf.convert_to_tensor(alpha, dtype=var.dtype.base_dtype)
       

        # m_t = alpha*m + (1-alpha)*grad
        m_t = m.assign(alpha_t * m + (1.0 - alpha_t) * grad)

        # use control dependency so the EMA update happens before using m_t
        with tf.control_dependencies([m_t]):
            upd = (1.0 - alpha_t) * grad + alpha_t * m_t
            delta = lr_t * upd         # THIS is the amount to subtract from var
           
           # print("In gradient update rule - gradient")
           # print(grad)
            #print("In gradient update rule - delta")
            #print(delta)
            
            return tf.identity(delta, name="ema_blend_delta")
  
    # expose slots if you want to read them later (optional)
    update_rule.ema_slots = slots
    return update_rule


def exp_decay_weights(n, *, alpha=None, half_life=None, rate=None,
                      newest='last', normalize=True, dtype=float):
   #Returns length-n weights w where newer samples get larger weight.
   # newest: 'last' -> newest is index n-1   | 'first' -> newest is index 0
   # Only one of alpha, half_life, rate should be given.
     if alpha is None and half_life is None and rate is None:
        alpha = 0.9  # default
     if half_life is not None:
        alpha = 2.0 ** (-1.0 / float(half_life))
     if rate is not None:
        # exact exp form; ignore alpha
        k = np.arange(n, dtype=dtype)
        w = np.exp(-float(rate) * k)
     else:
        k = np.arange(n, dtype=dtype)
        w = np.power(float(alpha), k)

     if newest == 'last':      # make the last item the newest
        w = w[::-1]
     if normalize:
        print("w", w)
        s = w.sum()
        if s != 0:
            w = w / s
     return w.astype(dtype)


