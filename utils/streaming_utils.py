# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 16:31:28 2025

@author: Divya

"""

from dataclasses import dataclass
import numpy as np
from collections import defaultdict

@dataclass

class LabelStats:
    count: int
    mean: np.ndarray

def make_label_stats(d):
    return LabelStats(count=0, mean=np.zeros(d, dtype=float))
# --------------------------

def update_labels_batch_onehot(stats_dict, X, Y_onehot):
    """
    Batch update of per-label running means (Y is one-hot).
    
    Parameters
    ----------
    stats_dict : defaultdict(label -> LabelStats)
        storage for running stats
    X : ndarray of shape (n_samples, n_features)
        feature vectors
    Y_onehot : ndarray of shape (n_samples, n_classes)
        one-hot encoded labels
    """
    d = X.shape[1]
    y_labels = np.argmax(Y_onehot, axis=1)   # convert one-hot → class index
    labels = np.unique(y_labels)

    for lbl in labels:
        X_lbl = X[y_labels == lbl]
        n_new = X_lbl.shape[0]
        if n_new == 0:
            continue

        if lbl not in stats_dict:
            stats_dict[lbl] = make_label_stats(d)

        s = stats_dict[lbl]
        n_old = s.count
        mu_old = s.mean

        # pooled mean update
        n_tot = n_old + n_new
        mu_new = (mu_old * n_old + X_lbl.sum(axis=0)) / n_tot

        s.count = n_tot
        s.mean = mu_new


def compute_label_weights_from_stats(label_stats, normalize=True, eps=1e-8):
    """
    Compute per-label weights from a LabelStats dictionary.
    Each label's weight = N / (K * n_c), where n_c is that label's count.

    Parameters
    ----------
    label_stats : dict[label -> LabelStats]
        Dictionary where each value has a 'count' attribute.
    normalize : bool, optional
        If True, normalize weights so their mean = 1 (recommended).
    eps : float, optional
        Small value to prevent division by zero for unseen labels.

    Returns
    -------
    label_weights : dict[label -> float]
        Mapping of label -> computed class weight.
    """
    # Extract label counts
    labels = list(label_stats.keys())
    counts = np.array([max(label_stats[l].count, eps) for l in labels], dtype=float)

    total = np.sum(counts)
    K = len(counts)

    # Balanced inverse-frequency weight
    weights = total / (K * counts)

    # Normalize so mean weight = 1 (stable loss scaling)
    if normalize:
        weights = weights / np.mean(weights)

    # Return dict for easy lookup
    return {l: float(w) for l, w in zip(labels, weights)}


def aging_weights_by_label_from_index(
    Y,                      # (n, K) one-hot OR (n,) int labels
    *,
    one_hot=True,
    half_life_steps=10,     # half-life in batch index steps
    newest_heavier=True,    # True: newer has larger weight; False: older has larger weight
    normalize=True,         # normalize to mean ≈ 1
    dtype=np.float32
):
    """
    Returns per-sample aging weights (n,) based purely on within-batch indices per label.
    """
    # 1) Get integer labels per sample
    if one_hot:
        y_idx = np.argmax(Y, axis=1).astype(int)
    else:
        y_idx = np.asarray(Y, dtype=int)

    n = len(y_idx)
    idx = np.arange(n)

    # 2) For each label, compute a per-sample 'age' based on index
    age = np.zeros(n, dtype=float)

    if newest_heavier:
        # newer (larger index) -> weight closer to 1
        # age = last_pos[label] - current_pos
        for lbl in np.unique(y_idx):
            pos_lbl = idx[y_idx == lbl]
            last = pos_lbl.max()
            age[y_idx == lbl] = last - pos_lbl
    else:
        # older (smaller index) -> weight closer to 1
        # age = current_pos - first_pos[label]
        for lbl in np.unique(y_idx):
            pos_lbl = idx[y_idx == lbl]
            first = pos_lbl.min()
            age[y_idx == lbl] = pos_lbl - first

    # 3) Exponential decay by age: w = 0.5 ** (age / half_life_steps)
    #    If you prefer rate λ instead of half-life, use: np.exp(-lambda_ * age)
    half_life_steps = float(half_life_steps)
    half_life_steps = max(half_life_steps, 1e-6)  # avoid /0
    w = np.power(0.5, age / half_life_steps).astype(dtype)

    # 4) Normalize to mean ≈ 1 (keeps loss scale stable)
    if normalize and w.mean() > 0:
        w = (w / w.mean()).astype(dtype)

    return w


def tumbling_window_xy(X, y, window_size):
    """
    Generate non-overlapping (X_window, y_window) pairs — i.e., a tumbling window.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    window_size : int
        Number of samples per window (batch size per iteration).

    Yields
    ------
    (X_window, y_window) : tuple of arrays
        Consecutive non-overlapping windows from X, y.
    """
    n = len(X)
    for start in range(0, n, window_size):
        end = min(start + window_size, n)
        X_batch = X[start:end]
        y_batch = y[start:end]
        print("BATCH END:", end)
        yield X_batch, y_batch