# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 19:38:45 2025

@author: Divya
"""

import numpy as np

class SGD:
    """
    Stochastic Gradient Descent with optional Momentum and Nesterov.
    Args
    ----
    lr : float
        Learning rate.
    momentum : float
        0.0 means vanilla SGD. Typical values: 0.8–0.99.
    nesterov : bool
        Use Nesterov accelerated gradient if True (requires momentum > 0).
    weight_decay : float
        L2 regularization coefficient (applied as wd * w).
    """
    def __init__(self, lr=1e-2, momentum=0.0, nesterov=False, weight_decay=0.0):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        self._vel = {}  # parameter -> velocity buffer

    def step(self, params, grads):
        """
        Perform one SGD update.
        params: list of np.ndarray (parameters)
        grads : list of np.ndarray (gradients, same shapes as params)
        Returns updated params (in-place also).
        """
        for i, (w, g) in enumerate(zip(params, grads)):
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * w

            if self.momentum > 0.0:
                v = self._vel.get(i, np.zeros_like(w))
                v = self.momentum * v - self.lr * g
                self._vel[i] = v
                if self.nesterov:
                    # Nesterov uses the look-ahead gradient step
                    w += self.momentum * v - self.lr * g
                else:
                    w += v
            else:
                w -= self.lr * g
        return params


# --- Tiny demo on linear regression y = x @ w + b ---
# Generate toy data
rng = np.random.default_rng(0)
X = rng.normal(size=(256, 5))
true_w = rng.normal(size=(5, 1))
true_b = 0.7
y = X @ true_w + true_b + 0.1 * rng.normal(size=(256, 1))

# Initialize params
w = rng.normal(scale=0.1, size=(5, 1))
b = np.array([[0.0]])

# opt = SGD(lr=0.05, momentum=0.9, nesterov=True, weight_decay=1e-4)

opt = SGD(lr=0.05, momentum=0.0, nesterov=False, weight_decay=1e-4)

def mse_and_grads(X, y, w, b):
    # preds and loss
    yhat = X @ w + b
    diff = yhat - y
    loss = (diff**2).mean()

    # grads
    N = X.shape[0]
    gw = (2.0 / N) * X.T @ diff       # d/dw MSE
    gb = (2.0 / N) * diff.sum(0, keepdims=True)  # d/db MSE
    return loss, [gw, gb]

for epoch in range(200):
    print('Epoch %d' % epoch)
    loss, grads = mse_and_grads(X, y, w, b)
    print('Loss %f' % loss)
    opt.step([w, b], grads)
# After training, w ~ true_w and b ~ true_b
