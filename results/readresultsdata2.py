# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 18:36:38 2025

@author: Divya
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
#  CONFIG: 50 rounds, 10 clients, 8 selected each round
#  (Replace the simulated block with your real CSV)
# =====================================================

# ---- LOAD REAL CSV ----
# df = pd.read_csv("your_file.csv")   # <--- uncomment for real data

# ---- SIMULATED EXAMPLE (remove when using real CSV) ----
num_rounds = 50
num_clients = 10
clients_per_round = 8
rng = np.random.default_rng(3)

rows = []
drift_types = ["none", "u", "cd", "cs"]  # u = unstable drift

for r in range(num_rounds):
    selected = rng.choice(num_clients, size=clients_per_round, replace=False)
    for c in selected:
        drift = rng.choice(drift_types, p=[0.6, 0.2, 0.1, 0.1])
        acc = rng.uniform(0.6, 0.98)
        loss = rng.uniform(0.1, 1.0)
        rows.append([r, c, acc, loss, drift])

df = pd.DataFrame(rows, columns=["round","client","acc","loss","drift"])
# ---- END SIMULATION ----

# Normalize drift values
df["drift"] = df["drift"].fillna("none").str.lower()
df["drift"] = df["drift"].replace({"null": "none"})  # safety

# =====================================================
#  COLOR + LEGEND LABELS
# =====================================================
color_map = {
    "none": "tab:blue",    # no drift
    "u":    "tab:orange",  # unstable drift
    "cd":   "tab:red",     # concept drift
    "cs":   "tab:green",   # covariate shift
}

label_map = {
    "none": "No drift",
    "u":    "Unstable drift (u)",
    "cd":   "Concept drift (cd)",
    "cs":   "Covariate shift (cs)",
}

# ---- Desired legend order ----
drift_order = ["none", "u", "cd", "cs"]

# =====================================================
#  PLOT
# =====================================================
fig, ax = plt.subplots(figsize=(13, 6))

for drift_type in drift_order:
    group = df[df["drift"] == drift_type]
    if group.empty:
        continue
    ax.scatter(
        group["round"], group["client"],
        c=color_map[drift_type],
        s=50,
        alpha=0.8,
        label=label_map[drift_type]
    )

ax.set_xlabel("Round")
ax.set_ylabel("Client ID")
ax.set_title("Client Participation per Round (Drift Only)\n50 Rounds, 10 Clients, 8 Sampled per Round")
ax.set_yticks(range(num_clients))
ax.grid(axis="x", alpha=0.3)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()
