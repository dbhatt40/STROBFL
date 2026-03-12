# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 03:45:07 2026

@author: Divya
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------
# Input files (STROBFL vs ADAM)
# ----------------------------
files = {
    "Drift:0%":   "syn-s10s.txt",   # STROBFL
    "Drift:0%a":  "syn-a10s.txt",   # ADAM
    "Drift:50%":  "syn-s14s.txt",   # STROBFL
    "Drift:50%a": "syn-a14s.txt",   # ADAM
    "Drift:100%": "syn-s18s.txt",   # STROBFL
    "Drift:100%a":"syn-a18s.txt",   # ADAM
}

# files = {
#     "STROBFL":   "syn-s44.txt",   # STROBFL
#     "STRSAGA":  "syn-strsaga44.txt",   # ADAM
#     "SVRG":  "syn-svrg44.txt",   # STROBFL
# }

# ----------------------------
# Load + reshape
# ----------------------------
dfs = []
for label, f in files.items():
    # If your .txt is whitespace-separated, switch to:
    # df = pd.read_csv(f, delim_whitespace=True)
    df = pd.read_csv(f)

    df = df[["t", "eval_success"]].copy()
    df["Method"] = label
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

# Parse "Drift" (shown once) and "Algo" (two boxes per drift)
def parse_label(label: str):
    if label.endswith("a"):  # ADAM variant
        return label[:-1], "ADAM"       # remove trailing 'a'
    return label, "STROBFL"

all_df[["Drift", "Algo"]] = all_df["Method"].apply(lambda s: pd.Series(parse_label(s)))

# Ensure consistent ordering: Drift levels and Algo order
drift_order = ["Drift:0%", "Drift:50%", "Drift:100%"]
algo_order = ["STROBFL", "ADAM"]

# ----------------------------
# Plot: Drift once + two boxes (STROBFL/ADAM) per drift
# ----------------------------
sns.set(style="whitegrid")
plt.figure(figsize=(7, 5))

ax = sns.boxplot(
    data=all_df,
    x="Drift",
    y="eval_success",
    hue="Algo",
    order=drift_order,
    hue_order=algo_order,
    palette={"STROBFL": "tab:orange", "ADAM": "tab:blue"},
    showfliers=True,
)

# ----------------------------
# Overlay mean ± std per (Drift, Algo)
# ----------------------------
stats = (
    all_df.groupby(["Drift", "Algo"])["eval_success"]
    .agg(["mean", "std"])
    .reset_index()
)

# Seaborn boxplot places the two hue boxes around each category center.
# These offsets work well for 2 hues.
offset = {"STROBFL": -0.2, "ADAM": 0.2}

for _, row in stats.iterrows():
    if row["Drift"] not in drift_order:
        continue
    x_center = drift_order.index(row["Drift"])
    x = x_center + offset[row["Algo"]]

    ax.errorbar(
        x=x,
        y=row["mean"],
        yerr=row["std"],
        fmt="o",
        color="black",
        capsize=6,
        linewidth=2,
        zorder=10,
    )

# ----------------------------
# Labels / title / legend
# ----------------------------
plt.ylabel("Global Accuracy (%)")
plt.title("Comparison of STROBFL vs ADAM under Different Drift Levels")
plt.legend(title=None, loc="best")
plt.tight_layout()
plt.show()

