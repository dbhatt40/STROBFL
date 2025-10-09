# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 13:19:29 2025

@author: Divya
"""
import tensorflow as tf


class CustomRuleSGD(tf.compat.v1.train.GradientDescentOptimizer):
    def __init__(self, learning_rate=0.01, update_rule=None, **kw):
        super(CustomRuleSGD, self).__init__(learning_rate=learning_rate, **kw)
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
        print("Training step")
        return train_op


	
def shrink_rule(grad, var, lr_t, global_step):
      wd = 1e-4
      return lr_t * (grad + wd * var)

# opt = CustomRuleSGD(learning_rate=0.05, update_rule=shrink_rule)