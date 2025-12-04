# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 15:01:55 2025

@author: Divya
"""

import numpy as np
import tensorflow as tf
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

        # Bias to enforce class imbalance
        self.class_prior_


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
    inp = layers.Input(shape=(gv.NUM_DIM,), name="main_input")

    x = layers.Dense(32, activation='relu')(inp)
    x = layers.Dense(32, activation='relu')(x)

    # Logits output
    out = layers.Dense(gv.NUM_CLASSES)(x)

    model = Model(inputs=inp, outputs=out)
    return model