# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 11:22:36 2026

@author: Divya
"""
# -*- coding: utf-8 -*-
"""
Regular boxplot with three methods + legend
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------
# Input files (3 methods)
# ----------------------------
files = {
    "STROBFL":   "syn-s44.txt",
    "ADAM":  "syn-a44.txt",

}

# ----------------------------
# Load data
# ----------------------------
dfs = []
for label, f in files.items():
    # If your txt files are whitespace-separated, use:
    # df = pd.read_csv(f, delim_whitespace=True)
    df = pd.read_csv(f)

    df = df[["t", "eval_success"]].copy()
    df["Method"] = label
    print(df["eval_success"].mean())
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

# ----------------------------
# Plot
# ----------------------------
sns.set(style="whitegrid")
plt.figure(figsize=(7, 5))

ax = sns.boxplot(
    data=all_df,
    x="Method",
    y="eval_success",
    showfliers=True,
)

# ----------------------------
# Manual legend (three methods)
# ----------------------------
palette = sns.color_palette(n_colors=3)

legend_handles = [
    Patch(facecolor=palette[0], edgecolor="black", label="STROBFL"),
    Patch(facecolor=palette[1], edgecolor="black", label="STRSAGA"),
    Patch(facecolor=palette[2], edgecolor="black", label="SVRG"),
]

ax.legend(handles=legend_handles, title="Methods", loc="best")

# ----------------------------
# Labels & title
# ----------------------------
ax.set_title("Comparison of Coefficient of Determination", fontsize=18)
ax.set_ylabel("Coefficient of determination", fontsize=16)

plt.tight_layout()
plt.show()
