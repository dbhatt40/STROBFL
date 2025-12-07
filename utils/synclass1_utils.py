# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 15:01:55 2025

@author: Divya
"""

import numpy as np
from tensorflow.keras import layers, Model
import global_vars as gv


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
    ):
        self.rng = np.random.default_rng(random_state)
        self.noise_std = float(noise_std)
        self.imbalance_factor = float(imbalance_factor)
        self.samples_per_cycle = int(samples_per_cycle)

        # Keep a global counter for "time"
        self.global_step = 0

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

        We split one cycle into 4 quarters and smoothly interpolate
        near the boundaries for gradual drift.
        """
        # position within cycle in [0,1)
        cycle_pos = (t % self.samples_per_cycle) / float(self.samples_per_cycle)

        # 4 equal segments: [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1.0)
        quarter = 0.25
        idx = int(cycle_pos // quarter)  # 0,1,2,3
        idx_next = (idx + 1) % 4

        # Within each quarter, use the last alpha fraction as a transition region
        # for gradual mixing between regime idx and idx_next.
        local_pos = (cycle_pos - idx * quarter) / quarter  # in [0,1)
        transition_width = 0.4  # fraction of quarter used for transition

        if local_pos < (1.0 - transition_width):
            # mostly pure regime idx
            mix = 0.0
        else:
            # linearly mix from idx to idx_next
            # local_pos in [1 - transition_width, 1) -> mix in [0,1)
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

        # simple linear interpolation; for cov we do element-wise (not PSD-guaranteed but OK for sim)
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
        # t_global can exist just for completeness, but does not change anything
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
):
    """
    Federated stream where:
      - num_drifted_clients clients see drifting data
      - the rest see stationary data
      - drifted clients can either share one drift pattern ("shared")
        or have independent drift streams ("independent").

    Parameters
    ----------
    num_rounds : int
        Number of FL rounds (T).
    num_clients : int
        Total number of clients (N).
    batch_size : int
        Local training batch per client per round (B).
    num_drifted_clients : int
        Number of clients that have drift (0 <= num_drifted_clients <= num_clients).
    drift_clients_mode : {"independent", "shared"}
        "independent": each drifted client has its own DriftStream4Class.
        "shared": all drifted clients share one DriftStream4Class.
    arrival_rate : float
        Fraction of batch_size that arrives as new samples in the queue each round.
    test_batch_size : int
        Number of test samples per round.
    noise_std, imbalance_factor, samples_per_cycle, random_state :
        Passed into the stream constructors.
    """

    assert 0 <= num_drifted_clients <= num_clients, "num_drifted_clients must be between 0 and num_clients"

    rng = np.random.default_rng(random_state)

    # Decide which client IDs are drifted.
    # Here we simply take the first num_drifted_clients; you can randomize if you want.
    drifted_client_ids = list(range(num_drifted_clients))
    stationary_client_ids = list(range(num_drifted_clients, num_clients))

    client_streams = [None] * num_clients

    # --- Create streams for drifted clients ---
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
                client_streams[cid] = DriftStream4Class(
                    noise_std=noise_std,
                    imbalance_factor=imbalance_factor,
                    samples_per_cycle=samples_per_cycle,
                    random_state=rng.integers(1_000_000) + cid,
                )
        else:
            raise ValueError("drift_clients_mode must be 'independent' or 'shared'")

    # --- Create streams for stationary clients ---
    for cid in stationary_client_ids:
        client_streams[cid] = StationaryStream4Class(
            noise_std=noise_std,
            imbalance_factor=imbalance_factor,
            random_state=rng.integers(1_000_000) + 10_000 + cid,
        )

    # One global test stream: you can choose drifted or stationary.
    # Here I use a drifting stream for tests.
    test_stream = DriftStream4Class(
        noise_std=noise_std,
        imbalance_factor=imbalance_factor,
        samples_per_cycle=samples_per_cycle,
        random_state=rng.integers(1_000_000),
    )

    # Per-client queues of (x, y, t)
    queues = [[] for _ in range(num_clients)]
    arrivals_per_round = max(0, int(round(arrival_rate * batch_size)))

    for r in range(num_rounds):
        client_batches = []

        for cid in range(num_clients):
            q = queues[cid]
            stream = client_streams[cid]

            # 1) New arrivals for this client
            if arrivals_per_round > 0:
                X_new, y_new, t_new = stream.sample_batch(arrivals_per_round)
                q.extend(
                    (X_new[i], int(y_new[i]), float(t_new[i]))
                    for i in range(arrivals_per_round)
                )

            # 2) Ensure we have at least batch_size samples
            if len(q) < batch_size:
                missing = batch_size - len(q)
                X_extra, y_extra, t_extra = stream.sample_batch(missing)
                q.extend(
                    (X_extra[i], int(y_extra[i]), float(t_extra[i]))
                    for i in range(missing)
                )

            # 3) Pop batch for this client
            batch_samples = q[:batch_size]
            del q[:batch_size]

            X_c = np.stack([s[0] for s in batch_samples], axis=0).astype(np.float32)
            y_c = np.array([s[1] for s in batch_samples], dtype=np.int64)
            t_c = np.array([s[2] for s in batch_samples], dtype=np.float32)

            client_batches.append((X_c, y_c, t_c))

        # 4) Test batch (drifting)
        X_test, y_test, t_test = test_stream.sample_batch(test_batch_size)

        yield r, client_batches, (X_test, y_test, t_test)


def synclass1_model():
    inp = layers.Input(shape=(gv.DATA_DIM,), name="main_input")

    x = layers.Dense(32, activation='relu')(inp)
    x = layers.Dense(32, activation='relu')(x)

    # Logits output
    out = layers.Dense(gv.NUM_CLASSES)(x)

    model = Model(inputs=inp, outputs=out)
    return model