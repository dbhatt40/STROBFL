# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 15:01:55 2025

@author: Divya
"""

import numpy as np
from tensorflow.keras import layers, Model
import global_vars as gv
import time

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
        mu_A = np.array([0.0, 0.0])
        mu_B = np.array([2.0, 0.0])
        mu_C = np.array([0.0, 2.0])
        mu_D = np.array([-2.0, -1.0])

        # Covariances
        cov_A = np.array([[1.0, 0.2],
                          [0.2, 1.0]])
        cov_B = np.array([[1.5, -0.3],
                          [-0.3, 1.0]])
        cov_C = np.array([[0.8, 0.0],
                          [0.0, 1.2]])
        cov_D = np.array([[1.0, 0.5],
                          [0.5, 1.5]])

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
        transition_width = 0.4  # fraction of quarter used for transition

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

    def sample_one(self):
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

    def sample_batch(self, batch_size):
        """
        Draw a batch of samples: X shape (batch_size, 2), y shape (batch_size,), t shape (batch_size,)
        """
        X = np.zeros((batch_size, 2), dtype=np.float32)
        y = np.zeros(batch_size, dtype=np.int64)
        t = np.zeros(batch_size, dtype=np.float32)

        for i in range(batch_size):
            xi, yi, ti = self.sample_one()
            X[i] = xi
            y[i] = yi
            t[i] = ti

        return X, y, t


class StationaryStream4Class:
    """
    Stationary 4-class generator:
      - Same interface as DriftStream4Class (sample_one, sample_batch)
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

    def sample_one(self):
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
            xi, yi, ti = self.sample_one()
            X[i] = xi
            y[i] = yi
            t[i] = ti

        return X, y, t


def federated_mixed_drift_stream_with_queues(
    num_rounds,
    num_clients,
    batch_size,
    num_drifted_clients,
    drift_clients_mode="independent",  # "independent" or "shared"
    arrival_rate=1.0,
    test_batch_size=256,
    noise_std=0.2,
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
    queues = [[] for _ in range(num_clients)]

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
            shared_stream = DriftStream4Class(
                noise_std=noise_std,
                imbalance_factor=imbalance_factor,
                samples_per_cycle=samples_per_cycle,
                random_state=rng.integers(1_000_000),
            )
            for cid in drifted_client_ids:
                client_streams[cid] = shared_stream  # all share same drift
        elif drift_clients_mode == "independent":
            for cid in drifted_client_ids:
                phase_offset = int(rng.integers(0, samples_per_cycle))
                client_streams[cid] = DriftStream4Class(
                    noise_std=noise_std,
                    imbalance_factor=imbalance_factor,
                    samples_per_cycle=samples_per_cycle,
                    random_state=rng.integers(1_000_000) + cid,
                    initial_step=phase_offset
                )
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
#-------------------------
        
    test_streams = [None] * num_clients
    # --- Create analogous test streams so test has same drift/stationary structure as clients ---

# Drifted test "clients"
    for cid in stationary_client_ids:
        test_streams[cid] = StationaryStream4Class(
            noise_std=noise_std,
            imbalance_factor=0,
            random_state=rng.integers(1_000_000) + 2_000_000 + cid,
            ) 
    if num_drifted_clients > 0:
        if drift_clients_mode == "shared":
        # one shared drifting test stream
            shared_test_stream = DriftStream4Class(
                noise_std=noise_std,
                imbalance_factor=imbalance_factor,
                samples_per_cycle=samples_per_cycle,
                random_state=rng.integers(1_000_000) + 999_000,
                )
            for cid in drifted_client_ids:
                test_streams[cid] = shared_test_stream
        elif drift_clients_mode == "independent":
            for cid in drifted_client_ids:
               phase_offset = int(rng.integers(0, samples_per_cycle))
               test_streams[cid] = DriftStream4Class(
                            noise_std=noise_std,
                            imbalance_factor=imbalance_factor,
                            samples_per_cycle=samples_per_cycle,
                            random_state=rng.integers(1_000_000) + 1_000_000 + cid,
                            initial_step=phase_offset,
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
        base_n = test_batch_size // num_clients
        remainder = test_batch_size % num_clients  # distribute leftover

        X_test_list = []
        y_test_list = []
        t_test_list = []

        for cid in range(num_clients):
                stream = test_streams[cid]
    # Give first 'remainder' clients one extra sample
                n_c = base_n + (1 if cid < remainder else 0)
                if n_c <= 0:
                   continue

                X_c_test, y_c_test, t_c_test = stream.sample_batch(n_c)
                X_test_list.append(X_c_test)
                y_test_list.append(y_c_test)
                t_test_list.append(t_c_test)

# Concatenate into one global test batch
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        t_test = np.concatenate(t_test_list, axis=0)

# (optional) sanity: shuffle test batch
        idx = np.random.permutation(len(X_test))
        X_test = X_test[idx]
        y_test = y_test[idx]
        t_test = t_test[idx]
                
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

    # ----- 2) Flatten updates for similarity computation -----
    flat_updates = np.stack([_flatten_weights(u) for u in client_updates], axis=0)  # (K, D)

    # ----- 3) RBF similarity matrix between client updates -----
    X = flat_updates
    sq_norms = np.sum(X * X, axis=1, keepdims=True)          # (K,1)
    sq_dists = sq_norms + sq_norms.T - 2.0 * (X @ X.T)       # (K,K)
    sq_dists = np.maximum(sq_dists, 0.0)

    off = sq_dists[~np.eye(num_clients, dtype=bool)]
    if off.size > 0:
        gamma_eff = 1.0 / (np.median(off) + eps)
    else:
        gamma_eff = gamma  # degenerate case K=1
    sim_matrix = np.exp(-gamma_eff * sq_dists)               # (K,K)

    np.fill_diagonal(sim_matrix, 0.0)
    sim_scores = sim_matrix.sum(axis=1)                      # (K,)
    sim_scores = np.maximum(sim_scores, eps)
    sim_scores = sim_scores / (sim_scores.sum() + eps)

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
    combined_w = sample_w * sim_scores * age_scores
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
