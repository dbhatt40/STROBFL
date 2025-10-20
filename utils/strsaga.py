# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 20:47:58 2025

@author: Divya
"""

# STRSAGA: Streaming SAGA-style variance-reduced SGD
# -----------------------------------------------
# Supports:
#   * Logistic loss (binary, y in {0,1} or {-1,+1})
#   * Squared loss (regression)
#   * True SAGA control variates with a bounded LRU cache if sample IDs are provided
#   * EMA-based surrogate control variates if no IDs are provided
#   * Sliding window or EMA of gradient means
#   * L2 regularization and FedProx-style proximal term
#
# Author: ChatGPT (reference implementation)
# License: MIT

from __future__ import annotations
from collections import OrderedDict, deque
import numpy as np

def _sigmoid(z):
    # numerically stable
    pos = z >= 0
    neg = ~pos
    out = np.empty_like(z, dtype=float)
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out

class LRUGradTable:
    """Bounded LRU cache for per-sample stored gradients (SAGA table)."""
    def __init__(self, capacity: int, d: int):
        self.capacity = int(capacity)
        self.d = int(d)
        self._store = OrderedDict()  # key -> grad vector

    def get(self, key):
        if key in self._store:
            g = self._store.pop(key)
            self._store[key] = g
            return g
        return None

    def put(self, key, grad_vec):
        if key in self._store:
            self._store.pop(key)
            self._store[key] = grad_vec
        else:
            if len(self._store) >= self.capacity:
                self._store.popitem(last=False)  # evict LRU
            self._store[key] = grad_vec

    def mean(self):
        if not self._store:
            return np.zeros(self.d, dtype=float)
        # arithmetic mean of stored grads
        return np.mean(np.stack(list(self._store.values()), axis=0), axis=0)

    def __len__(self):
        return len(self._store)


class STRSAGA:
    """
    Streaming SAGA-style optimizer for linear models.

    Model:
        - Logistic: p(y=1|x) = sigmoid(w^T x + b)
        - Squared:  y_hat = w^T x + b

    Loss (per-sample):
        logistic:  - y*log(sigmoid(z)) - (1-y)*log(1-sigmoid(z))
        squared:   0.5 * (y - z)^2
        with L2:   + (lambda_/2)*||w||^2
        with FedProx-like proximal: + (mu/2)*||w - w_ref||^2

    Control variates:
        - If sample_ids are provided: true SAGA with a bounded LRU grad table.
        - Else: EMA surrogate: use g_t - ema_last + ema_mean.

    Parameters
    ----------
    n_features : int
        Dimension of x.
    loss : {"logistic", "squared"}
    step_size : float
        Base learning rate (η).
    lambda_l2 : float
        L2 regularization coefficient.
    mu_prox : float
        Proximal (FedProx-like) coefficient. If >0, pass w_ref to fit/partial_fit.
    batch_size : int
        Minibatch size per step.
    grad_table_capacity : int
        Capacity of the per-sample gradient table when IDs are provided.
    ema_beta : float
        EMA decay for surrogate control variate when no IDs are present (0 < beta < 1).
    mean_mode : {"ema", "window"}
        How to maintain running mean gradient for the control variate.
    mean_window : int
        Window size if mean_mode == "window".
    clip_norm : float or None
        If set, clip gradients to this L2 norm.
    random_state : int or None
        RNG seed for shuffling in offline mini-epochs.

    Notes
    -----
    * For streaming with no stable IDs, keep `grad_table_capacity=0`, rely on EMA surrogate.
    * For true SAGA on a bounded replay set, pass `sample_ids` and set a positive capacity.
    """
    def __init__(self,
                 n_features: int,
                 loss: str = "logistic",
                 step_size: float = 1e-2,
                 lambda_l2: float = 0.0,
                 mu_prox: float = 0.0,
                 batch_size: int = 64,
                 grad_table_capacity: int = 0,
                 ema_beta: float = 0.98,
                 mean_mode: str = "ema",
                 mean_window: int = 256,
                 clip_norm: float | None = None,
                 random_state: int | None = 42):
        self.d = int(n_features)
        self.loss = loss.lower()
        assert self.loss in {"logistic", "squared"}
        self.eta = float(step_size)
        self.lambda_l2 = float(lambda_l2)
        self.mu_prox = float(mu_prox)
        self.batch_size = int(batch_size)
        self.clip_norm = clip_norm

        self.grad_table = None
        if grad_table_capacity and grad_table_capacity > 0:
            self.grad_table = LRUGradTable(grad_table_capacity, self.d)

        # running mean (for control variate global mean)
        self.mean_mode = mean_mode
        assert mean_mode in {"ema", "window"}
        self.ema_beta = float(ema_beta)
        self.grad_mean = np.zeros(self.d, dtype=float)
        self._window = deque(maxlen=int(mean_window)) if mean_mode == "window" else None

        self.rng = np.random.default_rng(random_state)

        # parameters
        self.w = np.zeros(self.d, dtype=float)
        self.b = 0.0

        # EMA surrogate (when no IDs)
        self.ema_last = np.zeros(self.d, dtype=float)

    # ---- Utility ----
    def _update_mean(self, g_batch):
        """Update running mean gradient (control variate global mean)."""
        g_mean = g_batch.mean(axis=0)
        if self.mean_mode == "ema":
            self.grad_mean = self.ema_beta * self.grad_mean + (1 - self.ema_beta) * g_mean
        else:
            self._window.append(g_mean)
            self.grad_mean = np.mean(np.stack(self._window, axis=0), axis=0)
        return g_mean

    def _clip(self, g):
        if self.clip_norm is None:
            return g
        n = np.linalg.norm(g)
        if n > self.clip_norm and n > 0:
            return g * (self.clip_norm / n)
        return g

    # ---- Gradients ----
    def _grad_logistic(self, X, y):
        """
        y: can be {0,1} or {-1,+1}. Internally convert to {0,1}.
        Returns gradient wrt w (batch_sum), and grad per-sample for table.
        """
        y01 = ((y + 1) / 2) if np.array_equal(np.unique(y), np.array([-1, 1])) else y
        y01 = y01.astype(float)
        z = X @ self.w + self.b
        p = _sigmoid(z)
        # per-sample gradient wrt w: (p - y) * x
        err = (p - y01)
        gw_samples = (err[:, None] * X)  # shape (m, d)
        gb = np.sum(err)                 # scalar
        return gw_samples, gb

    def _grad_squared(self, X, y):
        """
        Squared loss: 0.5*(y - z)^2, z = Xw + b
        grad w per-sample: (z - y)*x
        """
        z = X @ self.w + self.b
        err = (z - y)
        gw_samples = (err[:, None] * X)
        gb = np.sum(err)
        return gw_samples, gb

    def _grad(self, X, y):
        if self.loss == "logistic":
            return self._grad_logistic(X, y)
        else:
            return self._grad_squared(X, y)

    # ---- Single optimization step ----
    def _step_minibatch(self, Xb, yb, ids=None, w_ref=None):
        """
        Perform one STRSAGA minibatch update.
        """
        m = Xb.shape[0]
        gw_samp, gb = self._grad(Xb, yb)              # (m,d), scalar
        g_mean_batch = self._update_mean(gw_samp)     # (d,)

        # Build control variates and (optionally) update table
        if (ids is not None) and (self.grad_table is not None):
            # true SAGA:
            table_mean_before = self.grad_table.mean()  # ḡ(old)
            ctrl = np.zeros(self.d, dtype=float)
            for i in range(m):
                key = ids[i]
                g_i = gw_samp[i]
                g_old = self.grad_table.get(key)  # None if absent
                if g_old is None:
                    g_old = np.zeros(self.d, dtype=float)
                # SAGA control variate: g_i - g_old + ḡ(old)
                ctrl += (g_i - g_old + table_mean_before)
                # update table with new gradient
                self.grad_table.put(key, g_i)
            ctrl /= m
            # update global mean *after* table update for next step
            # (we already updated self.grad_mean from g_mean_batch)
        else:
            # EMA surrogate: g_i - ema_last + grad_mean
            # aggregate over minibatch using their average
            ctrl = g_mean_batch - self.ema_last + self.grad_mean
            # update ema_last toward current batch mean
            self.ema_last = self.ema_beta * self.ema_last + (1 - self.ema_beta) * g_mean_batch

        # Add regularization terms (w-only)
        reg = self.lambda_l2 * self.w
        if (w_ref is not None) and (self.mu_prox > 0.0):
            reg = reg + self.mu_prox * (self.w - w_ref)

        # Compose full gradient
        gw_full = ctrl + reg
        gb_full = gb / m  # bias does not include reg

        # Optional clipping
        gw_full = self._clip(gw_full)
        if self.clip_norm is not None:
            if abs(gb_full) > self.clip_norm:
                gb_full = np.sign(gb_full) * self.clip_norm

        # Parameter update (SGD step)
        self.w = self.w - self.eta * gw_full
        self.b = self.b - self.eta * gb_full

    # ---- Public APIs ----
    def partial_fit(self, X, y, sample_ids=None, w_ref=None):
        """
        Run one pass over (X,y) in minibatches. Suitable for streaming chunks.
        X: (n,d), y: (n,)
        sample_ids: (n,) if available (ints/strings); else None
        w_ref: reference vector for proximal term if mu_prox>0
        """
        n = X.shape[0]
        idx = np.arange(n)
        # for streaming, you may choose NOT to shuffle; here we keep order
        for i in range(0, n, self.batch_size):
            sl = slice(i, min(i + self.batch_size, n))
            Xb, yb = X[sl], y[sl]
            ids = sample_ids[sl] if (sample_ids is not None) else None
            self._step_minibatch(Xb, yb, ids=ids, w_ref=w_ref)
        return self

    def fit_stream(self, iterator, steps: int, id_iterator=None, w_ref=None):
        """
        Consume 'steps' chunks from iterator, each yielding (X_chunk, y_chunk).
        id_iterator (optional) should yield arrays of sample_ids aligned to chunks.
        """
        for t in range(steps):
            Xc, yc = next(iterator)
            ids = next(id_iterator) if id_iterator is not None else None
            self.partial_fit(Xc, yc, sample_ids=ids, w_ref=w_ref)
        return self

    def predict_proba(self, X):
        z = X @ self.w + self.b
        if self.loss == "logistic":
            p1 = _sigmoid(z)
            return np.c_[1 - p1, p1]
        else:
            raise ValueError("predict_proba is only for logistic loss.")

    def predict(self, X, threshold=0.5):
        z = X @ self.w + self.b
        if self.loss == "logistic":
            return ( _sigmoid(z) >= threshold ).astype(int)
        else:
            return z  # regression

    def params(self):
        return {"w": self.w.copy(), "b": float(self.b)}

    def set_params(self, w, b=0.0):
        self.w = np.array(w, dtype=float).reshape(-1)
        self.b = float(b)
        return self
