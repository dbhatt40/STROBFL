# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 15:01:55 2025

@author: Divya
"""

import numpy as np
from tensorflow.keras import layers, Model
import global_vars as gv
import time
from collections import deque

import tensorflow as tf
import math
import re




class DriftStream4Class:
    """
    4-class synthetic data generator with:
      - Covariate drift (Gaussian mean/cov change by phase)
      - Concept drift (W,b change by phase)
      - Abrupt + gradual drift transitions
      - Class imbalance control
      - Gaussian noise
      - Infinite stream with time/phase cycling
    """

    def __init__(
        self,
        noise_std=0.2,
        imbalance_factor=0.0,
        samples_per_cycle=10000,
        random_state=None,
        initial_step=0
    ):
        self.rng = np.random.default_rng(random_state)
        self.noise_std = float(noise_std)
        self.imbalance_factor = float(imbalance_factor)
        self.samples_per_cycle = int(samples_per_cycle)

        # Keep a global counter for "time"
        self.global_step = initial_step

        # Class prior bias (same as stationary stream)
        self.class_prior_bias = np.array([2.0, 0.5, -0.5, -1.0])

        # Define several regimes (A,B,C,D) for concept + covariate drift
        self.mu_list, self.cov_list, self.W_list, self.b_list = self._build_regimes()

    def _build_regimes(self):
        """
        Define 4 different regimes for covariate + concept drift.
        You can tweak these if you want stronger/weaker drift.
        """
        # Means
        drift_scale_mu_a = 3.0
        drift_scale_mu_b = 4.0
        drift_scale_mu_c = 4.0
        drift_scale_mu_d = 4.0
        mu_A = np.array([0.0, 0.0])*drift_scale_mu_a
        mu_B = np.array([2.0, 0.0])*drift_scale_mu_b
        mu_C = np.array([0.0, 2.0])*drift_scale_mu_c
        mu_D = np.array([-2.0, -1.0])*drift_scale_mu_d
        
        # Covariances
        cov_A = np.array([[1.0, 0.2],
                          [0.2, 1.0]])
        cov_B = np.array([[1.5, -0.3],
                          [-0.3, 1.0]])
        cov_C = np.array([[0.8, 0.0],
                          [0.0, 1.2]])
        cov_D = np.array([[1.0, 0.5],
                          [0.5, 1.5]])
        
        W_scale_A = 1.5
        W_scale_B = 2.5
        W_scale_C = 0.8
        W_scale_D = 3.0
        # Linear classifiers (W,b) for each regime
        W_A = np.array([
            [1.0,  0.6],   # class 0
            [-1.0, 0.3],   # class 1
            [0.4, -1.0],   # class 2
            [0.6,  1.0],   # class 3
        ])
        b_A = np.array([0.0, -0.5, 0.1, 0.2])

        W_B = np.array([
            [0.5,  1.0],
            [-0.8, 0.2],
            [1.0, -0.5],
            [-0.3, -1.0],
        ])
        b_B = np.array([0.2, 0.0, -0.3, 0.4])

        W_C = np.array([
            [1.2, -0.2],
            [-0.5, 1.0],
            [0.2, -1.2],
            [0.8,  0.8],
        ])
        b_C = np.array([-0.2, 0.3, 0.0, 0.1])

        W_D = np.array([
            [0.8,  0.8],
            [-1.0, -0.1],
            [0.1, -0.8],
            [0.3,  1.2],
        ])
        b_D = np.array([0.3, -0.4, 0.2, 0.0])
        
        W_A = W_scale_A * W_A
        W_B = W_scale_B * W_B
        W_C = W_scale_C * W_C
        W_D = W_scale_D * W_D

        mu_list  = [mu_A,  mu_B,  mu_C,  mu_D]
        cov_list = [cov_A, cov_B, cov_C, cov_D]
        W_list   = [W_A,   W_B,   W_C,   W_D]
        b_list   = [b_A,   b_B,   b_C,   b_D]

        return mu_list, cov_list, W_list, b_list

    def _phase_and_mix(self, t):
        """
        Map global step t into:
          - a base regime index (0..3)
          - a mixing coefficient lambda in [0,1] for gradual transitions.
        """
        cycle_pos = (t % self.samples_per_cycle) / float(self.samples_per_cycle)

        # 4 equal segments: [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1.0)
        quarter = 0.25
        idx = int(cycle_pos // quarter)  # 0,1,2,3
        idx_next = (idx + 1) % 4

        local_pos = (cycle_pos - idx * quarter) / quarter  # in [0,1)
        transition_width = 0.20  # fraction of quarter used for transition

        if local_pos < (1.0 - transition_width):
            mix = 0.0
        else:
            mix = (local_pos - (1.0 - transition_width)) / transition_width
            mix = float(np.clip(mix, 0.0, 1.0))

        return idx, idx_next, mix

    def _current_params(self, t):
        """
        Get (mu, cov, W, b) at time t, with possible interpolation between regimes.
        """
        idx, idx_next, mix = self._phase_and_mix(t)

        mu0,  mu1  = self.mu_list[idx],  self.mu_list[idx_next]
        cov0, cov1 = self.cov_list[idx], self.cov_list[idx_next]
        W0,   W1   = self.W_list[idx],   self.W_list[idx_next]
        b0,   b1   = self.b_list[idx],   self.b_list[idx_next]

        if mix == 0.0:
            return mu0, cov0, W0, b0

        mu = (1.0 - mix) * mu0  + mix * mu1
        cov = (1.0 - mix) * cov0 + mix * cov1
        W = (1.0 - mix) * W0    + mix * W1
        b = (1.0 - mix) * b0    + mix * b1

        return mu, cov, W, b

    def sample_one_linear(self):
        """
        Draw a single (x, y, t_global) sample from the drifting process.
        """
        t_global = float(self.global_step)
        self.global_step += 1

        mu, cov, W, b = self._current_params(t_global)

        # Covariate drift: Gaussian parameters change over time
        x = self.rng.multivariate_normal(mu, cov)

        # Concept drift: classifier parameters W,b change over time
        logits = W @ x + b

        # Add noise and imbalance
        logits = logits + self.rng.normal(0.0, self.noise_std, size=4)
        logits = logits + self.imbalance_factor * self.class_prior_bias

        # Class label
        y = int(np.argmax(logits))

        return x.astype(np.float32), y, t_global
    
    def sample_one_nonlinear(self):
        """
        Draw a single (x, y, t_global) sample from the drifting process.
               
        """
        
        
        t_global = float(self.global_step)
        self.global_step += 1

        mu, cov, W, b = self._current_params(t_global)

        # Covariate drift: Gaussian parameters change over time
        x = self.rng.multivariate_normal(mu, cov)
        x1, x2 = x
        # Nonlinear features
        phi = np.array([
          np.sin(x1),
          np.cos(x2),
          x1 * x2,
          x1**2 - x2**2
        ])
        logits = W @ phi[:2] + b
        # Additional nonlinear class structure
        logits[0] += 0.8 * np.sin(x1)
        logits[1] += 0.8 * np.cos(x2)
        logits[2] += 0.4 * x1 * x2
        logits[3] += 0.2 * (x1**2 - x2**2)
        
        # Noise
        logits += self.rng.normal(
            0.0,
            self.noise_std,
            size=4
        )
                
        # Imbalance
        logits += (
            self.imbalance_factor
            * self.class_prior_bias
        )
        y = int(np.argmax(logits))
      
        return x.astype(np.float32), y, t_global
      

    def sample_batch(self, batch_size):
        """
        Draw a batch of samples: X shape (batch_size, 2), y shape (batch_size,), t shape (batch_size,)
        """
        X = np.zeros((batch_size, 2), dtype=np.float32)
        y = np.zeros(batch_size, dtype=np.int64)
        t = np.zeros(batch_size, dtype=np.float32)

        
        for i in range(batch_size):
            xi, yi, ti = self.sample_one_nonlinear()
            X[i] = xi
            y[i] = yi
            t[i] = ti

        return X, y, t
    
  

    def sample_test_batch(self, batch_size, train_batchsize):
        """
        Draw a batch of samples: X shape (batch_size, 2), y shape (batch_size,), t shape (batch_size,)
        """
        X = np.zeros((batch_size, 2), dtype=np.float32)
        y = np.zeros(batch_size, dtype=np.int64)
        t = np.zeros(batch_size, dtype=np.float32)

        for i in range(batch_size):
            xi, yi, ti = self.sample_one_nonlinear()
            X[i] = xi
            y[i] = yi
            t[i] = ti
        synchronize_steps = train_batchsize - batch_size
        self.global_step += synchronize_steps
        return X, y, t


class StationaryStream4Class:
    """
    Stationary 4-class generator:
      - Same interface as DriftStream4Class (sample_one_linear, sample_batch)
      - No drift: fixed Gaussian + fixed classifier (similar to regime A).
    """

    def __init__(self, noise_std=0.2, imbalance_factor=0.0, random_state=None):
        self.rng = np.random.default_rng(random_state)
        self.noise_std = noise_std
        self.imbalance_factor = float(imbalance_factor)

        # Fixed covariate distribution (e.g., mu1, Sigma1)
        self.mu = np.array([0.0, 0.0])
        self.cov = np.array([[1.0, 0.2],
                             [0.2, 1.0]])

        # Fixed concept regime (e.g., regime A from earlier)
        self.W, self.b = self._regime_A()

        # Same class bias vector as drifting stream
        self.class_prior_bias = np.array([2.0, 0.5, -0.5, -1.0])

        self.global_step = 0  # we can still keep a counter for t, but it won't affect distribution

    @staticmethod
    def _regime_A():
        W = np.array([
            [1.0,  0.6],   # class 0
            [-1.0, 0.3],   # class 1
            [0.4, -1.0],   # class 2
            [0.6,  1.0],   # class 3
        ])
        b = np.array([0.0, -0.5, 0.1, 0.2])
        return W, b

    def sample_one_linear(self):
        t_global = float(self.global_step)
        self.global_step += 1

        x = self.rng.multivariate_normal(self.mu, self.cov)

        logits = self.W @ x + self.b
        logits = logits + self.rng.normal(0.0, self.noise_std, size=4)
        logits = logits + self.imbalance_factor * self.class_prior_bias

        y = int(np.argmax(logits))

        return x.astype(np.float32), y, t_global  # t_global here is just an index

    def sample_batch(self, batch_size):
        X = np.zeros((batch_size, 2), dtype=np.float32)
        y = np.zeros(batch_size, dtype=np.int64)
        t = np.zeros(batch_size, dtype=np.float32)

        for i in range(batch_size):
            xi, yi, ti = self.sample_one_linear()
            X[i] = xi
            y[i] = yi
            t[i] = ti

        return X, y, t
    
    def sample_test_batch(self, batch_size, train_batchsize):
        """
        Draw a batch of samples: X shape (batch_size, 2), y shape (batch_size,), t shape (batch_size,)
        """
        X = np.zeros((batch_size, 2), dtype=np.float32)
        y = np.zeros(batch_size, dtype=np.int64)
        t = np.zeros(batch_size, dtype=np.float32)

        for i in range(batch_size):
            xi, yi, ti = self.sample_one_linear()
            X[i] = xi
            y[i] = yi
            t[i] = ti
        synchronize_steps = train_batchsize - batch_size
        self.global_step += synchronize_steps
        return X, y, t


def federated_mixed_drift_stream_with_queues(
    num_rounds,
    num_clients,
    batch_size,
    num_drifted_clients,
    drift_clients_mode="independent",  # "independent" or "shared"
    arrival_rate=1.0,
    test_batch_size=500,
    noise_std=0.05,
    imbalance_factor=0.0,
    samples_per_cycle=10000,
    random_state=None,
    queue_maxlen=None,
    var_threshold_factor=1.0,
):
    """
    Federated stream where:
      - num_drifted_clients clients see drifting data
      - the rest see stationary data
      - drifted clients can either share one drift pattern ("shared")
        or have independent drift streams ("independent").

    NEW:
      - Each client has a fixed-size queue (queue_maxlen, default = batch_size).
      - For each client we keep per-label stats for samples currently in the queue:
          * mean[label] (vector in R^2)
          * variance[label] (per-dimension)
          * count[label]
      - When arrival_rate > 1, each new sample is admitted/evicted according to:
          1) If label is new in queue => insert, evict oldest sample overall.
          2) Else if distance^2 from label-mean > var_threshold_factor * avg_variance[label]
             => insert, evict oldest sample of same label (fallback: oldest overall).
          3) Else discard the sample.
    """
    def _force_insert(cid, sample):
        """
        Sliding-window insert: if queue is full, evict oldest overall.
        Always inserts the new sample, updating stats accordingly.
        """
        q = queues[cid]
        x, y, t = sample
        if len(q) >= queue_maxlen:
            x_old, y_old, t_old = q.pop(0)
        q.append((x, y, t))

    def _insert_with_policy(cid, sample):
        """
        Insert or discard a sample based on the user's rules
        when arrival_rate > 1.
        """
        q = queues[cid]
        x, y, t = sample
        x = x.astype(np.float64)  # use float64 for stable stats

        # Case 1:  delete oldest overall if full
        if len(q) >= queue_maxlen:
                x_old, y_old, t_old = q.pop(0)
                q.append((x.astype(np.float32), y, t))
        return True
    
    assert 0 <= num_drifted_clients <= num_clients, "num_drifted_clients must be between 0 and num_clients"

    rng = np.random.default_rng(random_state)

    if queue_maxlen is None:
        queue_maxlen = batch_size
    queues = [[] for _ in range(int(num_clients))]

    arrivals_per_round = max(0, int(round(arrival_rate * batch_size)))
    use_policy = arrival_rate > 1.0

    # Decide which client IDs are drifted.
    drifted_client_ids = list(range(num_drifted_clients))
    stationary_client_ids = list(range(num_drifted_clients, num_clients))

    client_streams = [None] * num_clients
     # Per-client queues of (x, y, t)


    # --- Create streams for drifted clients ---
#-----------------------------
    if num_drifted_clients > 0:
        if drift_clients_mode == "shared":
              shared_phase_offset = int(0.125 * samples_per_cycle)
              shared_seed = int(rng.integers(1_000_000))
              for cid in drifted_client_ids:
                    client_streams[cid] = DriftStream4Class(
                        noise_std=noise_std,
                        imbalance_factor=imbalance_factor,
                        samples_per_cycle=samples_per_cycle,
                        random_state=shared_seed + cid,   # different randomness per client
                        initial_step=shared_phase_offset  # same drift schedule
                  )
        elif drift_clients_mode == "independent":

            phase_offsets = [
                int(((i + 0.5) / num_drifted_clients) * samples_per_cycle)
                for i in range(num_drifted_clients)
            ]
            counter=0
            for cid in drifted_client_ids:
                # phase_offset = int(rng.integers(0, samples_per_cycle))
                client_streams[cid] = DriftStream4Class(
                    noise_std=noise_std,
                    imbalance_factor=imbalance_factor,
                    samples_per_cycle=samples_per_cycle,
                    random_state=rng.integers(1_000_000) + cid,
                    initial_step=phase_offsets[counter]
                    )
                counter = counter + 1
        else:
            raise ValueError("drift_clients_mode must be 'independent' or 'shared'")
#-------------------------
    # --- Create streams for stationary clients ---
    for cid in stationary_client_ids:
        client_streams[cid] = StationaryStream4Class(
            noise_std=noise_std,
            imbalance_factor=imbalance_factor,
            random_state=rng.integers(1_000_000) + 10_000 + cid,
        )
#-------------------------test streams-------------------
        
    test_streams = [None] * num_clients
    # --- Create analogous test streams so test has same drift/stationary structure as clients ---
    if num_drifted_clients > 0:
        if drift_clients_mode == "shared":
              shared_phase_offset = int(0.125 * samples_per_cycle)# or choose one common offset
              shared_seed = rng.integers(1_000_000) + 999_000
              for cid in drifted_client_ids:
                    test_streams[cid] = DriftStream4Class(
                        noise_std=noise_std,
                        imbalance_factor=imbalance_factor,
                        samples_per_cycle=samples_per_cycle,
                        random_state=shared_seed + cid,   # different randomness per client
                        initial_step=shared_phase_offset  # same drift schedule
                  )
        elif drift_clients_mode == "independent":
            phase_offsets = [
                int(((i + 0.5) / num_drifted_clients) * samples_per_cycle)
                for i in range(num_drifted_clients)
            ]
           
            counter=0
            for cid in drifted_client_ids:
                # phase_offset = int(rng.integers(0, samples_per_cycle))
                test_streams[cid]= DriftStream4Class(
                    noise_std=noise_std,
                    imbalance_factor=imbalance_factor,
                    samples_per_cycle=samples_per_cycle,
                    random_state=rng.integers(1_000_000) + 1_000_000 + cid,
                    initial_step=phase_offsets[counter]
                    )
                counter = counter + 1

# Drifted test "clients"
    for cid in stationary_client_ids:
        test_streams[cid] = StationaryStream4Class(
            noise_std=noise_std,
            imbalance_factor=imbalance_factor,
            random_state=rng.integers(1_000_000) + 2_000_000 + cid,
            ) 
  
 #-------------------------

    for r in range(num_rounds):
        client_batches = []

        for cid in range(num_clients):
            q = queues[cid]
            stream = client_streams[cid]

            # 1) New arrivals for this client
            if arrivals_per_round > 0:
                X_new, y_new, t_new = stream.sample_batch(arrivals_per_round)
                for i in range(arrivals_per_round):
                    sample = (X_new[i], int(y_new[i]), float(t_new[i]))
                    if use_policy:
                        _insert_with_policy(cid, sample)
                    else:
                        _force_insert(cid, sample)

            if arrival_rate < 1.0:
    # Don't top-up, just use what's available
                batch_len = min(len(q), batch_size)
            else:
    # For normal/high arrival rates, enforce full batch
                batch_len = batch_size

            # 3) Pop batch for this client (and update stats accordingly)
            batch_samples = q[:batch_len]
            del q[:batch_len]

            X_c = np.stack([s[0] for s in batch_samples], axis=0).astype(np.float32)
            y_c = np.array([s[1] for s in batch_samples], dtype=np.int64)
            t_c = np.array([s[2] for s in batch_samples], dtype=np.float32)

            client_batches.append((X_c, y_c, t_c))

       
        # 4) Test batch: same structure as client[cid]
#    - each "test client" contributes some samples
#    - ratio drifted / stationary matches num_drifted_clients / num_clients

# Base number of test samples per client
# 4) Test batches: per-client + global

        base_n = test_batch_size // num_clients
        remainder = test_batch_size % num_clients

        # test_batches_per_client = [None] * num_clients
        X_test_list, y_test_list, t_test_list = [], [], []

        for cid in range(num_clients):
            stream = test_streams[cid]
            n_c = base_n + (1 if cid < remainder else 0)
            if n_c <= 0:
                # keep empty batch for consistency
                Xc = np.zeros((0, 2), dtype=np.float32)
                yc = np.zeros((0,), dtype=np.int64)
                tc = np.zeros((0,), dtype=np.float32)
            else:
                Xc, yc, tc = stream.sample_test_batch(n_c, batch_size)  # advances by batch_size

        #     test_batches_per_client[cid] = (Xc, yc, tc)
            if Xc.shape[0] > 0:
              X_test_list.append(Xc)
              y_test_list.append(yc)
              t_test_list.append(tc)

# Global test batch = concat all per-client pieces
        X_test = np.concatenate(X_test_list, axis=0) if X_test_list else np.zeros((0,2), np.float32)
        y_test = np.concatenate(y_test_list, axis=0) if y_test_list else np.zeros((0,), np.int64)
        t_test = np.concatenate(t_test_list, axis=0) if t_test_list else np.zeros((0,), np.float32)

# Optional shuffle (use rng for reproducibility if you want)
        if len(X_test) > 0:
         idx = np.random.permutation(len(X_test))
         X_test, y_test, t_test = X_test[idx], y_test[idx], t_test[idx]

# Yield BOTH
        yield r, client_batches, (X_test, y_test, t_test)

    

def synclass1_model():
    inp = layers.Input(shape=(gv.DATA_DIM,), name="main_input")

    x = layers.Dense(32, activation='relu')(inp)
    x = layers.Dense(32, activation='relu')(x)

    # Logits output
    out = layers.Dense(gv.NUM_CLASSES)(x)

    model = Model(inputs=inp, outputs=out)
    return model



def _flatten_weights(weights_list):
    """Flatten a list/tuple of numpy arrays into a single 1D vector."""
    return np.concatenate([w.ravel() for w in weights_list])




def aggregate_with_rbf_and_aging(
    round_idx,
    global_weights,
    num_clients,
    return_dict,
    agent_list,
    gamma=1.5,
    eps=1e-12,
    age_lambda=0.2          # 0.0 disables aging (all ages weight = 1)
    ):
    """
    FedAvg * RBF-similarity * Aging aggregation.

    Aging factor uses: age_score = exp(-age_lambda * age)
    where age = current_time - timeofupdate.
    """


    if num_clients == 0:    
        return [w.copy() for w in global_weights]

    client_updates = []
    client_samples = []
    client_ages=[]
    
    arrived_updates = [
               k for k, v in return_dict.items()
               if k.endswith("_round_arrived") and v == round_idx                          
              ]
    latest_updates = {}

    for key in arrived_updates:
      m = re.match(r"(\d+)_r(\d+)_round_arrived", key)
      if m is None:
        continue

      cid = int(m.group(1))
      created_round = int(m.group(2))

      if (cid not in latest_updates or
        created_round > latest_updates[cid][0]):
        latest_updates[cid] = (created_round, key)

    arrived_updates = [v[1] for v in latest_updates.values()]

    for arrival_key in arrived_updates:                     
           prefix = arrival_key.replace("_round_arrived", "")           
           update = return_dict[f"{prefix}_weights"]
           client_updates.append(update)
           
           num_samples = return_dict[f"{prefix}_num_samples"]
           client_samples.append(num_samples)
           
           parts = arrival_key.split("_")
           original_round = int(parts[1][1:])  # remove the 'r'
           client_age = return_dict[f"{prefix}_round_arrived"]-original_round
           client_ages.append(client_age)
           
           print(f'RBF aggregating in this round {arrival_key}, samples {num_samples}, age{client_age}')

    # ----- 2) Flatten updates for similarity computation -----
    if len(client_updates) == 0:
     return [w.copy() for w in global_weights]
 
    for k, update in enumerate(client_updates):
       flat = _flatten_weights(update)
    #   flat = flat / (np.linalg.norm(flat) + 1e-12)
       print(
          f"Client {k}: "
          f"shape={flat.shape}, "
          f"norm={np.linalg.norm(flat):.6f}, "
          f"mean={flat.mean():.6f}, "
          f"std={flat.std():.6f}, "
          f"min={flat.min():.6f}, "
          f"max={flat.max():.6f}"
      )
 
    flat_updates = np.stack([_flatten_weights(u) for u in client_updates], axis=0)  # (K, D)

    # ----- 3) RBF similarity matrix between client updates -----
    X = flat_updates
    sq_norms = np.sum(X * X, axis=1, keepdims=True)          # (K,1)
    sq_dists = sq_norms + sq_norms.T - 2.0 * (X @ X.T)       # (K,K)
    sq_dists = np.maximum(sq_dists, 0.0)

    K = len(client_updates)
    if(K==1):
        sim_scores = np.ones(1,dtype=float)
    else:
        off = sq_dists[~np.eye(K, dtype=bool)]
        med = np.median(off)
        if med <= eps:
          gamma_eff = gamma 
        else:
          gamma_eff = 0.1/ med

         # degenerate case K=1
    sim_matrix = np.exp(-gamma_eff * sq_dists)   

    K = len(client_updates)
    if(K==1):
       sim_scores = np.ones(1,dtype=float)
    else:
       sim_matrix = np.exp(-gamma_eff*sq_dists)           
       np.fill_diagonal(sim_matrix, 0.0)
       sim_scores = sim_matrix.sum(axis=1)                    
       sim_scores = np.maximum(sim_scores, eps)
  

    # ----- 4) FedAvg-style sample weights -----
    sample_w = np.asarray(client_samples, dtype=float)
    sample_w = np.maximum(sample_w, 0.0)
    if sample_w.sum() <= 0:
        sample_w = np.ones_like(sample_w)
    sample_w = np.maximum(sample_w, eps)
    

    age_scores = np.ones(len(client_ages), dtype=float)
    if age_lambda and age_lambda > 0.0:
        for k in range(len(client_ages)):
            age = client_ages[k]
    
            if age is None:
                age = 0.0
    
            age_scores[k] = np.exp(-age_lambda * max(float(age), 0.0))
    age_scores = np.maximum(age_scores, eps)

  #  sim_scores = 0.2+0.8*sim_scores
    # ----- 6) Combine all three multiplicatively + renormalize -----
    combined_w = sample_w * sim_scores * age_scores
    combined_w = np.maximum(combined_w, eps)
    combined_w = combined_w / (combined_w.sum() + eps)
    print("combined weights with samples, rbf and aging:", combined_w)

    # ----- 7) Apply weighted average of updates to global weights -----
    new_global_weights = []

    for layer_idx in range(len(global_weights)):
       agg_update_layer = np.zeros_like(global_weights[layer_idx])

       for k in range(len(client_updates)):
          agg_update_layer += combined_w[k] * client_updates[k][layer_idx]
  
       new_global_weights.append(global_weights[layer_idx] + agg_update_layer)

    return new_global_weights


def aggregate_with_sw_fedavg(
    global_weights,
    num_clients,
    client_dict,
    agent_list,
    client_num_samples,
    gamma=1.0,
    eps=1e-12,
    age_lambda=1.0          # 0.0 disables aging (all ages weight = 1)
    ):
    """
    FedAvg * RBF-similarity * Aging aggregation.

    Aging factor uses: age_score = exp(-age_lambda * age)
    where age = current_time - timeofupdate.
    """
    current_time=None
    if current_time is None:
        current_time = time.time()  # seconds since epoch by default

    if num_clients == 0:
        return [w.copy() for w in global_weights]

    # ----- 1) Collect client updates + timestamps -----
    client_updates = []
    client_times = []

    for k in range(num_clients):
        entry = client_dict[str(agent_list[k])]
        t_u = client_dict[str(agent_list[k]) + "_time"]
        client_updates.append(entry)
        client_times.append(t_u)

     # ----- 4) FedAvg-style sample weights -----
    sample_w = np.asarray(client_num_samples, dtype=float)
    sample_w = np.maximum(sample_w, 0.0)
    if sample_w.sum() <= 0:
        sample_w = np.ones_like(sample_w)
    sample_w = sample_w / (sample_w.sum() + eps)

    # ----- 5) Aging weights -----
    # If age_lambda == 0 => all ones (no aging).
    age_scores = np.ones(num_clients, dtype=float)

    if age_lambda and age_lambda > 0.0:
        for k in range(num_clients):
            t_u = client_times[k]
            if t_u is None:
                # If no timestamp, treat as "fresh" (age=0) OR you can treat as old.
                age = 0.0
            else:
                age = float(current_time) - float(t_u)
            # decay: newer => larger weight
            age_scores[k] = np.exp(-age_lambda * max(age, 0.0))

        age_scores = np.maximum(age_scores, eps)
        age_scores = age_scores / (age_scores.sum() + eps)
    else:
        # normalized ones (optional)
        age_scores = age_scores / (age_scores.sum() + eps)

    # ----- 6) Combine all three multiplicatively + renormalize -----
    combined_w = sample_w * age_scores
    combined_w = np.maximum(combined_w, eps)
    combined_w = combined_w / (combined_w.sum() + eps)
    print("combined weights with rbf and aging:", combined_w)

    # ----- 7) Apply weighted average of updates to global weights -----
    new_global_weights = []
    for layer_idx in range(len(global_weights)):
        agg_update_layer = sum(
            combined_w[k] * client_updates[k][layer_idx] for k in range(num_clients)
        )
        new_global_weights.append(global_weights[layer_idx] + agg_update_layer)

    return new_global_weights


class PageHinkley:
    """
    Online Page-Hinkley drift detector (univariate).

    Detects a sustained *increase* in the monitored signal.
    To detect a decrease, call update() with -x instead of x.
    """
    def __init__(self, agent, delta=0.1, lambd=2.0, min_instances=30, signal_type="loss"):
        """
        delta: small tolerance for slight changes (insensitivity zone)
        lambd: threshold for raising an alarm
        min_instances: wait for this many samples before triggering
        """
        self.delta = float(delta)
        self.lambd = float(lambd)
        self.min_instances = int(min_instances)
        self.signal_type=signal_type
        self.agent = agent

        self.reset()

    def reset(self):
        self.t = 0
        self.mean = 0.0
        self.cum_sum = 0.0
        self.min_cum_sum = 0.0
        self.ph_stat = 0.0
        self.drift = False

    def update(self, x):
        """
        Feed one new observation x.
        Returns True if drift detected at this step, else False.
        """
        self.t += 1

        # Incremental mean
        self.mean += (x - self.mean) / self.t

        # Cumulative sum of deviations (for increase detection)
        self.cum_sum += (x - self.mean - self.delta)

        # Track minimum of cumulative sum
        self.min_cum_sum = min(self.min_cum_sum, self.cum_sum)

        # Page-Hinkley statistic
        self.ph_stat = self.cum_sum - self.min_cum_sum
       
        # Drift decision
        if self.t > self.min_instances and self.ph_stat > self.lambd:
            self.drift = True
           #print("------PH stat for signal agent, t, val, signal:",self.agent, self.t, self.ph_stat, self.signal_type)
            # You can either reset here or leave it accumulating
         #   self.reset()
            return True

        return False


class ZPageHinkley:
    """
    Scale-adaptive (z-score) Page-Hinkley drift detector (univariate).

    Detects a sustained *increase* in the monitored signal.
    To detect a decrease, call update() with -x instead of x.
    """

    def __init__(
        self,
        agent,
        alpha=0.02,          # EWMA rate for mean/variance
        delta_z=0.0,         # tolerance in z-space
        lambd_z=10.0,        # threshold in sigma-units
        min_instances=30,
        signal_type="loss",
        eps=1e-8,
    ):
        self.alpha = float(alpha)
        self.delta_z = float(delta_z)
        self.lambd_z = float(lambd_z)
        self.min_instances = int(min_instances)
        self.signal_type = signal_type
        self.agent = agent
        self.eps = eps

        self.reset()

    def reset(self):
        self.t = 0

        # EWMA mean and variance
        self.mean = 0.0
        self.var = 0.0

        # PH statistics
        self.cum_sum = 0.0
        self.min_cum_sum = 0.0
        self.ph_stat = 0.0

        self.drift = False

    def update(self, x):
        """
        Feed one new observation x.
        Returns True if drift detected at this step, else False.
        """
        self.t += 1

        # ---- EWMA mean ----
        if self.t == 1:
            self.mean = x
            self.var = 0.0
            return False

        prev_mean = self.mean
        self.mean = (1 - self.alpha) * self.mean + self.alpha * x

        # ---- EWMA variance (stable form) ----
        self.var = (1 - self.alpha) * self.var + self.alpha * (x - prev_mean) ** 2
        std = math.sqrt(self.var) + self.eps

        # ---- Z-score ----
        z = (x - self.mean) / std

        # ---- Page-Hinkley on z-score ----
        self.cum_sum += (z - self.delta_z)
        self.min_cum_sum = min(self.min_cum_sum, self.cum_sum)
        self.ph_stat = self.cum_sum - self.min_cum_sum

        # ---- Drift decision ----
        if self.t > self.min_instances and self.ph_stat > self.lambd_z:
            self.drift = True
            return True

        return False





class LossStabilityTest:
    def __init__(self, window=10, min_increase=0.4, std_mult=3.0):
        self.window = int(window)
        self.min_increase = float(min_increase)
        self.std_mult = float(std_mult)
        self.buf = deque(maxlen=self.window)

    def update(self, loss_val):
        self.buf.append(float(loss_val))
        if len(self.buf) < self.window:
            return False, {}

        arr = np.array(self.buf, dtype=np.float32)
        half = self.window // 2
        early = arr[:half]
        late  = arr[half:]

        early_mean, late_mean = float(early.mean()), float(late.mean())
        early_std,  late_std  = float(early.std() + 1e-8), float(late.std() + 1e-8)

        mean_up = (late_mean - early_mean) / max(early_mean, 1e-8) > self.min_increase
        std_up  = late_std > self.std_mult * early_std

        unstable = mean_up and std_up
        stats = {
            "early_mean": early_mean, "late_mean": late_mean,
            "early_std": early_std,   "late_std": late_std
        }
        return unstable, stats
    


class RelativeLossStabilityTest:
    """
    Window-based stability test using *relative* mean loss increase.

    Flags instability when:
      1) Mean loss increases by more than `rel_increase` fraction, AND
      2) Loss variance increases significantly.
    """

    def __init__(
        self,
        window=10,
        rel_increase=0.4,     # 40% relative increase
        std_mult=3.0,
        eps=1e-8,
        mean_floor=1e-3      # prevents explosion when early mean is tiny
    ):
        self.window = int(window)
        self.rel_increase = float(rel_increase)
        self.std_mult = float(std_mult)
        self.eps = eps
        self.mean_floor = mean_floor
        self.buf = deque(maxlen=self.window)

    def update(self, loss_val):
        self.buf.append(float(loss_val))

        if len(self.buf) < self.window:
            return False, {}

        arr = np.asarray(self.buf, dtype=np.float32)
        half = self.window // 2

        early = arr[:half]
        late  = arr[half:]

        early_mean = float(early.mean())
        late_mean  = float(late.mean())

        early_std  = float(early.std() + self.eps)
        late_std   = float(late.std()  + self.eps)

        # ---- Relative mean increase (percentage-style) ----
        denom = max(abs(early_mean), self.mean_floor)
        rel_mean_increase = (late_mean - early_mean) / denom

        mean_up = rel_mean_increase > self.rel_increase
        std_up  = late_std > self.std_mult * early_std

        unstable = mean_up and std_up

        stats = {
            "early_mean": early_mean,
            "late_mean": late_mean,
            "relative_mean_increase": rel_mean_increase,
            "early_std": early_std,
            "late_std": late_std,
        }

        return unstable, stats

    



def build_2step_accumulators(logits, y_int, num_classes, per_example_loss, scope="accum2", eps=1e-9):
    """
    Accumulates stats every training step.
    You can read aggregated metrics whenever you want (e.g., every 2 steps),
    then reset accumulators.

    logits: [B, C]
    y_int:  [B] int32/int64
    per_example_loss: [B] (weighted or unweighted)
    """
    y_int = tf.cast(y_int, tf.int32)
    pred = tf.argmax(logits, axis=1, output_type=tf.int32)  # [B]

    with tf.compat.v1.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        # How many *steps* have been accumulated in the current window
        step_ct = tf.Variable(0, trainable=False, dtype=tf.int32, name="step_ct")

        # Accumulate total loss sum and example count (for overall mean loss)
        loss_sum = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="loss_sum")
        ex_ct    = tf.Variable(0.0, trainable=False, dtype=tf.float32, name="ex_ct")

        # Per-label loss sum and per-label example count
        loss_sum_by_label = tf.Variable(tf.zeros([num_classes], tf.float32),
                                        trainable=False, name="loss_sum_by_label")
        ex_ct_by_label    = tf.Variable(tf.zeros([num_classes], tf.float32),
                                        trainable=False, name="ex_ct_by_label")

        # TP/FP/FN per label (for aggregated F1)
        tp_acc = tf.Variable(tf.zeros([num_classes], tf.float32), trainable=False, name="tp")
        fp_acc = tf.Variable(tf.zeros([num_classes], tf.float32), trainable=False, name="fp")
        fn_acc = tf.Variable(tf.zeros([num_classes], tf.float32), trainable=False, name="fn")

    # --- step contributions ---
    # Overall loss contribution
    step_loss_sum = tf.reduce_sum(per_example_loss)                 # scalar
    step_ex_ct    = tf.cast(tf.shape(per_example_loss)[0], tf.float32)

    # Per-label loss contributions
    step_loss_sum_by_label = tf.math.unsorted_segment_sum(
        per_example_loss, y_int, num_segments=num_classes
    )  # [C]
    step_ex_ct_by_label = tf.math.unsorted_segment_sum(
        tf.ones_like(per_example_loss, dtype=tf.float32), y_int, num_segments=num_classes
    )  # [C]

    # Per-label TP/FP/FN contributions
    y_oh = tf.one_hot(y_int, depth=num_classes, dtype=tf.float32)   # [B,C]
    p_oh = tf.one_hot(pred,  depth=num_classes, dtype=tf.float32)   # [B,C]

    step_tp = tf.reduce_sum(y_oh * p_oh, axis=0)                    # [C]
    step_fp = tf.reduce_sum((1.0 - y_oh) * p_oh, axis=0)            # [C]
    step_fn = tf.reduce_sum(y_oh * (1.0 - p_oh), axis=0)            # [C]

    # --- update accumulators each training step ---
    update_accum_op = tf.group(
        tf.compat.v1.assign_add(step_ct, 1),
        tf.compat.v1.assign_add(loss_sum, step_loss_sum),
        tf.compat.v1.assign_add(ex_ct, step_ex_ct),
        tf.compat.v1.assign_add(loss_sum_by_label, step_loss_sum_by_label),
        tf.compat.v1.assign_add(ex_ct_by_label, step_ex_ct_by_label),
        tf.compat.v1.assign_add(tp_acc, step_tp),
        tf.compat.v1.assign_add(fp_acc, step_fp),
        tf.compat.v1.assign_add(fn_acc, step_fn),
        name="update_accum_op"
    )

    # --- aggregated metrics from accumulators ---
    mean_loss = tf.math.divide_no_nan(loss_sum, ex_ct)  # scalar

    mean_loss_by_label = tf.math.divide_no_nan(loss_sum_by_label, ex_ct_by_label)  # [C]

    precision = tp_acc / (tp_acc + fp_acc + eps)  # [C]
    recall    = tp_acc / (tp_acc + fn_acc + eps)  # [C]
    f1_by_label = (2.0 * precision * recall) / (precision + recall + eps)         # [C]
    f1_macro = tf.reduce_mean(f1_by_label)                                         # scalar

    label_counts = tf.cast(ex_ct_by_label, tf.int32)  # [C]

    read_agg = {
        "step_ct": step_ct,
        "loss": mean_loss,
        "loss_per_label": mean_loss_by_label,
        "f1_per_label": f1_by_label,
        "f1_macro": f1_macro,
        "label_counts": label_counts,
    }

    # --- reset accumulators (after you read every 2 steps) ---
    reset_accum_op = tf.group(
        tf.compat.v1.assign(step_ct, 0),
        tf.compat.v1.assign(loss_sum, 0.0),
        tf.compat.v1.assign(ex_ct, 0.0),
        tf.compat.v1.assign(loss_sum_by_label, tf.zeros([num_classes], tf.float32)),
        tf.compat.v1.assign(ex_ct_by_label, tf.zeros([num_classes], tf.float32)),
        tf.compat.v1.assign(tp_acc, tf.zeros([num_classes], tf.float32)),
        tf.compat.v1.assign(fp_acc, tf.zeros([num_classes], tf.float32)),
        tf.compat.v1.assign(fn_acc, tf.zeros([num_classes], tf.float32)),
        name="reset_accum_op"
    )

    return update_accum_op, read_agg, reset_accum_op

