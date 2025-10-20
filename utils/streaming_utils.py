# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 16:31:28 2025

@author: Divya
"""

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