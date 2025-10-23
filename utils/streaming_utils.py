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