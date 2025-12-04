# -*- coding: utf-8 -*-
"""
Created on Sun Nov 23 11:47:20 2025

@author: Divya
"""

import numpy as np

class DriftStream4Class:
    """
    Core infinite stream for 4-class classification with:
      - covariate drift (P(X|t) changes)
      - both gradual and abrupt concept drift (P(y|X,t) changes)
      - configurable class imbalance
    """

    def __init__(
        self,
        noise_std=0.2,
        imbalance_factor=0.0,
        samples_per_cycle=10000,
        random_state=None,
    ):
        """
        Parameters
        ----------
        noise_std : float
            Gaussian noise on logits.
        imbalance_factor : float in [0, 1]
            0.0 -> near-uniform class frequencies.
            1.0 -> strong skew toward a dominant class.
        samples_per_cycle : int
            After this many samples, the drift pattern repeats in phase.
        random_state : int or None
            RNG seed.
        """
        self.rng = np.random.default_rng(random_state)
        self.noise_std = noise_std
        self.imbalance_factor = float(imbalance_factor)
        self.samples_per_cycle = int(samples_per_cycle)
        self.global_step = 0  # counts how many samples we've generated total

        # Covariate regimes: means & covariances
        self.mus = [
            np.array([0.0, 0.0]),     # phase ~ [0.0, 0.25)
            np.array([2.5, 0.5]),     # [0.25, 0.5)
            np.array([1.0, 3.0]),     # [0.5, 0.75)
            np.array([-1.0, 1.5])     # [0.75, 1.0)
        ]
        self.covs = [
            np.array([[1.0, 0.2], [0.2, 1.0]]),
            np.array([[1.2, -0.1], [-0.1, 1.0]]),
            np.array([[0.7, 0.3], [0.3, 1.3]]),
            np.array([[1.5, 0.4], [0.4, 0.8]])
        ]
        self.cov_points = (0.0, 0.25, 0.5, 0.75, 1.0)

        # Concept regimes: W,b for 4 classes each.
        self.WA, self.bA = self._regime_A()
        self.WB, self.bB = self._regime_B()
        self.WC, self.bC = self._regime_C()

        # prior bias for imbalance: class 0 most favored, then others
        # scale will be multiplied by imbalance_factor
        self.class_prior_bias = np.array([2.0, 0.5, -0.5, -1.0])

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

    @staticmethod
    def _regime_B():
        W = np.array([
            [0.5, -1.0],
            [1.0,  0.5],
            [-0.8, 0.3],
            [0.3,  1.2],
        ])
        b = np.array([0.2, 0.1, -0.4, 0.0])
        return W, b

    @staticmethod
    def _regime_C():
        W = np.array([
            [1.2,  0.2],
            [-0.5, 1.5],
            [0.3, -1.2],
            [-1.0, -0.5]
        ])
        b = np.array([0.1, 0.3, -0.2, 0.4])
        return W, b

    def _pick_covariate_regime(self, phase):
        # cov_points = [0.0,0.25,0.5,0.75,1.0] -> 4 regimes
        for j in range(len(self.cov_points) - 1):
            if self.cov_points[j] <= phase < self.cov_points[j + 1]:
                return self.mus[j], self.covs[j]
        return self.mus[-1], self.covs[-1]

    def _concept_params(self, phase):
        """
        Both gradual and abrupt concept drift:
          - [0.0, 0.3): Regime A
          - [0.3, 0.4): gradual A -> B interpolation
          - [0.4, 0.7): Regime B
          - [0.7, 0.8): abrupt switch to C (no interpolation)
          - [0.8, 1.0): Regime C
        """
        if phase < 0.3:
            return self.WA, self.bA
        elif phase < 0.4:
            # gradual interpolation A -> B
            alpha = (phase - 0.3) / 0.1  # in [0,1)
            W = (1 - alpha) * self.WA + alpha * self.WB
            b = (1 - alpha) * self.bA + alpha * self.bB
            return W, b
        elif phase < 0.7:
            # Regime B (stable)
            return self.WB, self.bB
        elif phase < 0.8:
            # abrupt B -> C: already switched to C
            return self.WC, self.bC
        else:
            # still C
            return self.WC, self.bC

    def sample_one(self):
        """
        Sample a single point (x, y, t_global) from the drifting process.
        """
        t_global = self.global_step / float(self.samples_per_cycle)
        phase = t_global % 1.0

        # Covariate regime
        mu, cov = self._pick_covariate_regime(phase)
        x = self.rng.multivariate_normal(mu, cov)

        # Concept regime (with drift)
        W, b = self._concept_params(phase)

        logits = W @ x + b
        logits = logits + self.rng.normal(0.0, self.noise_std, size=4)

        # Add imbalance bias
        logits = logits + self.imbalance_factor * self.class_prior_bias

        y = int(np.argmax(logits))

        self.global_step += 1
        return x.astype(np.float32), y, float(t_global)

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

def federated_drift_stream_with_queues(
    num_rounds,
    num_clients,
    batch_size,
    arrival_rate=1.0,
    test_batch_size=256,
    noise_std=0.2,
    imbalance_factor=0.0,
    samples_per_cycle=10000,
    random_state=None,
):
    """
    Infinite drift stream adapted for FL with per-client queues and arrival rate.

    For each round r in [0, num_rounds):
      - For each client c in [0, num_clients):
          * Generate arrivals = int(arrival_rate * batch_size) new samples,
            append to that client's queue.
          * If queue has < batch_size, top up from stream to reach batch_size.
          * Pop batch_size samples from queue -> (X_c, y_c, t_c) training batch.
      - Also generate one test batch (X_test, y_test, t_test).

    Parameters
    ----------
    num_rounds : int
        T: number of federated rounds.
    num_clients : int
        N: number of clients.
    batch_size : int
        B: local training batch per client per round.
    arrival_rate : float
        Fraction of batch_size that 'arrives' NEW per round into the queue.
        Example: 1.0 -> B new samples per round per client.
                 0.5 -> 0.5 * B new samples, rest from backlog (if any).
                 2.0 -> 2B new samples, queue grows over time.
    test_batch_size : int
        Number of test samples generated per round (global test set).
    noise_std, imbalance_factor, samples_per_cycle, random_state :
        Passed to DriftStream4Class.

    Yields
    ------
    round_idx : int
    client_batches : list of length num_clients
        Each element is (X_client, y_client, t_client), where:
            X_client: (B, 2)
            y_client: (B,)
            t_client: (B,)
    test_batch : tuple
        (X_test, y_test, t_test)
    """
    stream = DriftStream4Class(
        noise_std=noise_std,
        imbalance_factor=imbalance_factor,
        samples_per_cycle=samples_per_cycle,
        random_state=random_state,
    )

    # Per-client queues: store (x, y, t) tuples waiting to be used
    queues = [[] for _ in range(num_clients)]

    arrivals_per_round = max(0, int(round(arrival_rate * batch_size)))

    for r in range(num_rounds):
        client_batches = []

        for cid in range(num_clients):
            q = queues[cid]

            # 1) New arrivals into this client's queue
            if arrivals_per_round > 0:
                X_new, y_new, t_new = stream.sample_batch(arrivals_per_round)
                q.extend(
                    (X_new[i], int(y_new[i]), float(t_new[i]))
                    for i in range(arrivals_per_round)
                )

            # 2) Ensure we have at least batch_size samples in queue.
            if len(q) < batch_size:
                missing = batch_size - len(q)
                X_extra, y_extra, t_extra = stream.sample_batch(missing)
                q.extend(
                    (X_extra[i], int(y_extra[i]), float(t_extra[i]))
                    for i in range(missing)
                )

            # 3) Pop batch_size samples from queue as this client's batch
            batch_samples = q[:batch_size]
            del q[:batch_size]

            X_c = np.stack([s[0] for s in batch_samples], axis=0).astype(np.float32)
            y_c = np.array([s[1] for s in batch_samples], dtype=np.int64)
            t_c = np.array([s[2] for s in batch_samples], dtype=np.float32)

            client_batches.append((X_c, y_c, t_c))

        # 4) Global test batch for this round
        X_test, y_test, t_test = stream.sample_batch(test_batch_size)

        yield r, client_batches, (X_test, y_test, t_test)
