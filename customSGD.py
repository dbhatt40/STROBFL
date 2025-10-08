# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 13:19:29 2025

@author: Divya
"""

import tensorflow as tf
from tensorflow import keras

class CustomSGD(keras.optimizers.Optimizer):
    """SGD with optional momentum, Nesterov, and decoupled weight decay."""
    def __init__(
        self,
        learning_rate=0.01,
        momentum=0.0,
        nesterov=False,
        weight_decay=0.0,   # decoupled weight decay (like AdamW-style)
        name="custom_sgd",		
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", learning_rate)
        self._set_hyper("weight_decay", weight_decay)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)

    def build(self, var_list):
        # Create slot variables (e.g., velocity) for each trainable variable.
        if self.momentum > 0.0:
            for var in var_list:
                self.add_slot(var, "velocity")

    def update_step(self, grad, var):
        # Convert IndexedSlices -> dense if needed (simple, robust default)
        if isinstance(grad, tf.IndexedSlices):
            grad = tf.convert_to_tensor(grad)

        lr = tf.cast(self._decayed_lr(var.dtype), var.dtype)
        wd = tf.cast(self._get_hyper("weight_decay", var.dtype), var.dtype)

        # Decoupled weight decay: w <- w - lr * wd * w
        if wd > 0:
            var.assign_sub(lr * wd * var)

        if self.momentum > 0.0:
            v = self.get_slot(var, "velocity")
            # v <- m * v + grad
            v.assign(self.momentum * v + grad)
            # Nesterov: grad_hat = grad + m * v
            update = grad + self.momentum * v if self.nesterov else v
        else:
            update = grad

        # w <- w - lr * update
        var.assign_sub(lr * update)

    def get_config(self):
        base = super().get_config()
        return {
            **base,
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "weight_decay": self._serialize_hyperparameter("weight_decay"),
            "momentum": self.momentum,
            "nesterov": self.nesterov,
        }

# ---- Example usage ----
# A tiny model to show it working end-to-end.
# model = keras.Sequential([
#     keras.layers.Dense(16, activation="relu", input_shape=(10,)),
#     keras.layers.Dense(1)
# ])

# opt = CustomSGD(learning_rate=0.01, momentum=0.9, nesterov=True, weight_decay=1e-4)
# model.compile(optimizer=opt, loss="mse")

# import numpy as np
# x = np.random.randn(128, 10).astype("float32")
# y = np.random.randn(128, 1).astype("float32")
# model.fit(x, y, epochs=3, verbose=0)
