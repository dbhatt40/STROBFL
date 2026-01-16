# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 03:45:07 2026

@author: Divya
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

files = {
    "STROBFL-STROBFL": "results_A.txt",
    "STRSAGA-AVG": "results_D.txt",
    "SVRG-AVG":"results_E.txt",
}

dfs = []
for label, f in files.items():
    df = pd.read_csv(f)
    df = df[["t", "eval_success"]].copy()
    df["Method"] = label
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)


sns.set(style="whitegrid")

plt.figure(figsize=(7, 5))
ax = sns.boxplot(
    data=all_df,
    x="Method",
    y="eval_success",
    showfliers=True,
)

# ---- overlay mean and std ----
stats = all_df.groupby("Method")["eval_success"].agg(["mean", "std"]).reset_index()

for i, row in stats.iterrows():
    ax.errorbar(
        x=i,
        y=row["mean"],
        yerr=row["std"],
        fmt="o",
        color="black",
        capsize=6,
        linewidth=2,
        label="Mean ± Std" if i == 0 else ""
    )

plt.ylabel("Validation Accuracy (%)")
plt.title("Distribution of Validation Accuracy Across Aggregation Methods")
plt.legend()
plt.tight_layout()
plt.show()
