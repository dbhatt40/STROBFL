# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 22:08:53 2026

@author: Divya
"""

import numpy as np

def aggregate_with_fednova(
    global_weights,
    num_clients,
    client_dict,
    agent_list,
    client_num_samples,
    client_local_steps,           # <-- REQUIRED for FedNova normalization (list length K)
    eps=1e-12,
    client_is_delta=True,         # True if client_dict holds DELTAS; False if it holds FULL weights
    normalize_by="steps",         # "steps" (common) or "lr_steps" if you want (η*τ)
    client_lrs=None,              # optional list length K, only used if normalize_by="lr_steps"
    clip_norm=None,               # optional: clip each client's normalized update (robustness)
):
    """
    FedNova aggregation (drop-in style) for heterogeneous local computation.

    Assumptions:
      - If client_is_delta=True:
          client_dict[str(cid)] is a list of numpy arrays representing Δw_i (same shapes as global_weights)
          and server applies: w_new = w_global + Σ_k α_k * (Δw_k / a_k)
      - If client_is_delta=False:
          client_dict[str(cid)] is full weights w_i, we internally convert to Δw_i = w_i - w_global.

    Normalizer a_k:
      - normalize_by="steps": a_k = max(τ_k, 1)
      - normalize_by="lr_steps": a_k = max(η_k * τ_k, eps) (requires client_lrs)

    Weighting α_k:
      - sample-weighted like FedAvg: α_k = n_k / Σ n_k

    Args:
      global_weights: list of numpy arrays (global model weights)
      num_clients: number of participating clients K
      client_dict: dict mapping str(agent_id) -> client payload (delta or weights)
      agent_list: list/array of selected agent ids length K
      client_num_samples: list/array length K
      client_local_steps: list/array length K (τ_k)
      eps: small constant
      client_is_delta: True if payload is delta; False if payload is full weights
      normalize_by: "steps" or "lr_steps"
      client_lrs: optional list length K for normalize_by="lr_steps"
      clip_norm: optional float; if set, clips each client's normalized update by global norm

    Returns:
      new_global_weights: list of numpy arrays
    """
    if num_clients == 0:
        return [w.copy() for w in global_weights]

    K = num_clients

    # ----- 1) FedAvg-style sample weights α_k -----
    alpha = np.asarray(client_num_samples, dtype=np.float64)
    alpha = np.maximum(alpha, 0.0)
    if alpha.sum() <= 0:
        alpha = np.ones_like(alpha)
    alpha = alpha / (alpha.sum() + eps)

    # ----- 2) Build normalizers a_k -----
    tau = np.asarray(client_local_steps, dtype=np.float64)
    tau = np.maximum(tau, 1.0)

    if normalize_by == "steps":
        a = tau
    elif normalize_by == "lr_steps":
        if client_lrs is None:
            raise ValueError("client_lrs must be provided when normalize_by='lr_steps'")
        lrs = np.asarray(client_lrs, dtype=np.float64)
        a = np.maximum(lrs * tau, eps)
    else:
        raise ValueError(f"Unknown normalize_by: {normalize_by}")

    # ----- 3) Collect deltas and normalize: Δw_k_norm = Δw_k / a_k -----
    deltas_norm = []
    for k in range(K):
        cid = agent_list[k]
        payload = client_dict[str(cid)]

        if client_is_delta:
            delta = payload
        else:
            # payload is full weights => delta = w_k - w_global
            delta = [payload[i] - global_weights[i] for i in range(len(global_weights))]

        inv_a = 1.0 / float(a[k])

        # Normalize delta by a_k
        dnorm = [inv_a * d.astype(np.float64) for d in delta]

        # Optional clip normalized update (helps if a client goes wild)
        if clip_norm is not None:
            # compute global norm across layers
            sq = 0.0
            for arr in dnorm:
                sq += float(np.sum(arr * arr))
            gnorm = np.sqrt(sq + eps)
            if gnorm > float(clip_norm):
                scale = float(clip_norm) / (gnorm + eps)
                dnorm = [scale * arr for arr in dnorm]

        deltas_norm.append(dnorm)

    # ----- 4) Aggregate normalized deltas with α_k -----
    new_global_weights = []
    for layer_idx in range(len(global_weights)):
        agg = np.zeros_like(global_weights[layer_idx], dtype=np.float64)
        for k in range(K):
            agg += alpha[k] * deltas_norm[k][layer_idx]
        new_global_weights.append((global_weights[layer_idx].astype(np.float64) + agg).astype(global_weights[layer_idx].dtype, copy=False))

    print("FedNova alpha(sample weights):", alpha)
    print("FedNova a(normalizers):", a)
    return new_global_weights
