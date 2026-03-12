# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 19:38:29 2026

@author: Divya
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# ============================================================
# Settings
# ============================================================
CSV_PATH =  "results_data1.csv"

DRIFTED_IDS = [0, 1, 2, 3]
STATIONARY_IDS = [4, 5, 6, 7]
ALL_IDS = DRIFTED_IDS + STATIONARY_IDS

# ---- Drift episodes (inclusive round ranges) ----
DRIFT_EPISODES = [(8, 20), (30, 42)]   # <-- adjust to your setup

# ============================================================
# Load data
# ============================================================
df = pd.read_csv(CSV_PATH)

required_cols = {"t", "i", "drift"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# ============================================================
# Preprocess: binary detection per (round, client)
# ============================================================
df["detected"] = df["drift"].notna().astype(int)

# ============================================================
# Episode-level collapse:
# Detected(E, c) = 1 if client c fired at least once in episode E
# ============================================================
episode_records = []

for ep_id, (start, end) in enumerate(DRIFT_EPISODES):
    ep_df = df[(df["t"] >= start) & (df["t"] <= end)]

    # OR over all rounds in the episode
    detected_per_client = (
        ep_df.groupby("i")["detected"]
             .max()
             .reindex(ALL_IDS, fill_value=0)
    )

    for cid in ALL_IDS:
        episode_records.append({
            "episode": ep_id,
            "client": cid,
            "detected": int(detected_per_client.loc[cid]),
            "truth": int(cid in DRIFTED_IDS)   # only drifted clients are true positives
        })

episode_df = pd.DataFrame(episode_records)

# ============================================================
# Confusion matrix (episode-level)
# ============================================================
TP = int(((episode_df["detected"] == 1) & (episode_df["truth"] == 1)).sum())
FN = int(((episode_df["detected"] == 0) & (episode_df["truth"] == 1)).sum())
FP = int(((episode_df["detected"] == 1) & (episode_df["truth"] == 0)).sum())
TN = int(((episode_df["detected"] == 0) & (episode_df["truth"] == 0)).sum())

# ============================================================
# Metrics
# ============================================================
precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
fpr       = FP / (FP + TN) if (FP + TN) > 0 else 0.0
fnr       = FN / (FN + TP) if (FN + TP) > 0 else 0.0

# ============================================================
# Heatmap table
# ============================================================
grid = pd.DataFrame(
    [[TP, FN],
     [FP, TN]],
    index=[
        "Drifted Clients\n(once per episode)\n n=0,1,2,3",
        "Stationary Clients\n n=4,5,6,7"
    ],
    columns=[
        "Detected Drift",
        "No Drift Detected"
    ]
)

# Semantic color encoding
# 0 = TN, 1 = TP, 2 = FP, 3 = FN
color_code = np.array([
    [1, 3],
    [2, 0],
])

# cmap = ListedColormap([
#     "#d9d9d9",  # TN
#     "#66c2a5",  # TP
#     "#fc8d62",  # FP
#     "#fdae61",  # FN
# ])
grey_blue_cmap = LinearSegmentedColormap.from_list(
    "grey_blue",
    ["#f0f0f0", "#c6dbef", "#6baed6", "#2171b5"]
)

# ============================================================
# Plot
# ============================================================
sns.set(style="white")
fig, ax = plt.subplots(figsize=(8.4, 4.8))

sns.heatmap(
    color_code,
    annot=grid.values,
    fmt="d",
    cmap=grey_blue_cmap,
    cbar=False,
    linewidths=0.6,
    linecolor="white",
    xticklabels=grid.columns,
    yticklabels=grid.index,
    annot_kws={"fontsize": 14},   # <-- increase number size
    ax=ax
)

episodes_str = ", ".join([f"{a}–{b}" for a, b in DRIFT_EPISODES])
# Keep labels outside the heatmap (anchor to the left of the tick)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")

# But left-justify the multi-line text inside the label block
for lab in ax.get_yticklabels():
    lab.set_multialignment("left")

# Add some space between labels and the heatmap
ax.tick_params(axis="y", pad=12)

# Optional: give a bit more left margin (usually small now)
fig.subplots_adjust(left=0.25)
ax.set_title(
    "Episode-Level Drift Detection Confusion Matrix\n"
    f"(Each Client Counted Once per Drift Episode: {episodes_str})",
    fontsize=13,
    pad=12
)

ax.set_xlabel("")
ax.set_ylabel("")

metrics_text = (
    f"Precision = {precision:.3f}\n"
    f"Recall (TPR) = {recall:.3f}\n"
    f"FPR = {fpr:.3f}\n"
    f"FNR = {fnr:.3f}"
)

ax.text(
    1.05, 0.5,
    metrics_text,
    transform=ax.transAxes,
    fontsize=12,
    va="center",
    ha="left"
)

plt.tight_layout()
plt.show()
