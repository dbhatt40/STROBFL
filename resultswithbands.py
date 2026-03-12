# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 15:49:04 2026

@author: Divya
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob

# -------------------------------------------------
# Helper: add shaded drift bands
# -------------------------------------------------
def add_band(ax, start, end, color, label=None, alpha=0.15, zorder=0):
    ax.axvspan(
        start,
        end,
        facecolor=color,
        alpha=alpha,
        lw=0,
        zorder=zorder,
        label=label
    )

# -------------------------------------------------
# Load and combine result files
# -------------------------------------------------


files = []

files  += glob("./results/syn-s44s.txt")
#files  += glob("./results/adam-aq.txt")

files  += glob("./results/syn-a44s.txt")
# files  += glob("./results/syn-s08s.txt")
print("Found files:")
for f in files:
    print(f)

dfs = []
methods = ["STROBFL", "STRSAGA", "SVRG"]

i = 0
for f in files:
    df = pd.read_csv(f)

    # extract directory names (if you need them)
    level1 = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(f)))))
    level2 = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(f))))
    level3 = os.path.basename(os.path.dirname(os.path.dirname(f)))

    df["level1"] = level1
    df["level2"] = level2
    df["level3"] = level3
    df["level4"] = methods[i]   # method name
    i += 1

    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)
print(all_df.head())

# -------------------------------------------------
# Plot
# -------------------------------------------------
sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=(10, 6))

sns.lineplot(
    data=all_df,
    x="t",
    y="eval_success",
    hue="level4",      # method
    style="level4",
    markers=True,
    ax=ax,
    zorder=2
)

# -------------------------------------------------
# Drift areas (example ranges – adjust to your setup)
# -------------------------------------------------
high_drift   = (8, 20)
medium_drift = (30, 42)

add_band(ax, *high_drift,   color="red",    label="Drift Area 1 (High)",   alpha=0.12, zorder=0)
add_band(ax, *medium_drift, color="green", label="Drift Area 2 (Medium)", alpha=0.12, zorder=0)

# -------------------------------------------------
# Axes labels and limits
# -------------------------------------------------
ax.set_xlabel("Round", fontsize=16)
ax.set_ylabel("Evaluation Success", fontsize=16)

ax.set_xlim(all_df["t"].min(), all_df["t"].max())

# -------------------------------------------------
# Build ONE clean legend (methods + drift areas)
# -------------------------------------------------
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))  # deduplicate

ax.legend(
    by_label.values(),
    by_label.keys(),
    title="Method / Drift",
    loc="upper right",
    fontsize=12,
    title_fontsize=12,
    labelspacing=1.2
)

ax.set_title(
    "Global Validation Accuracy Across Rounds with Drift Regions",
    fontsize=16
)

plt.tight_layout()
plt.show()
