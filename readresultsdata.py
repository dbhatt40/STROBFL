# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 10:34:37 2025

@author: Divya
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from glob import glob

files = []


files  += glob("./results/SS-0.txt")
files  += glob("./results/FA-0.txt")
files  += glob("./results/SA-0.txt")
# files  += glob("./results/SA.txt")
# files  += glob("./results/SVS.txt")

print("Found files:")
for f in files:
    print(f)

dfs = []
methods = ['STRAP-FL','FedProx','SVRG']

values = []

i = 0
for f in files:
    df = pd.read_csv(f)
    # col = df.iloc[:,1]
    # print(f"{f} -> min: {col.min()}, max: {col.max()}")
    # values.append(col.mean()) 
    # print(values)
   # df["eval_success"] *= 100
    # extract directory names for labeling
    level1 = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(f)))))
    level2 = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(f))))
    level3 = os.path.basename(os.path.dirname(os.path.dirname(f)))
    level4 = os.path.basename(os.path.dirname(f))

    df["level1"] = level1   # e.g., experiment
    df["level2"] = level2   # e.g., dataset
    df["level3"] = level3  # e.g., method
    df["level4"] = methods[i]   # e.g., run
    i = i+1
    
    

    dfs.append(df)
    

means_per_df = [df["eval_success"].mean() for df in dfs]

print(means_per_df)

all_df = pd.concat(dfs, ignore_index=True)
print(all_df.head())

colors = {
    "STRAP-FL": "#006400",  # dark green
    "FedProx": "#8B0000",   # dark red
    "SVRG": "#000080"       # navy
}

sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))

ax = sns.lineplot(
    data=all_df,
    x="t",
    y="eval_success",   
    hue="level4",      # method
    style="level4",    # run
    markers=True,
    palette=colors,
    linewidth=2.8,
    markersize=3
)

ax.legend(
    title="Method",
    frameon=True,
    fontsize=16,
    labelspacing=1.2,   # vertical space between entries
    handlelength=3.0,   # length of line/marker handle
    handletextpad=1.2,  # space between handle and text
    borderpad=1.2       # padding inside legend box
)

# ax.axvspan(
#     2, 6,
#     alpha=0.6,
#     color="lightgray",
#     label="Incremental Drift Area"
# )

# ax.axvspan(
#     6, 9,
#     alpha=0.6,
#     color="lightpink",
#     label="Gradual Drift Area"
# )
# ax.axvspan(
#     9, 13,
#     alpha=0.6,
#     color="lightgray",
    
# )
# ax.axvspan(
#     13, 16,
#     alpha=0.6,
#     color="lightpink",

# )
# ax.axvspan(
#     16, 20,
#     alpha=0.4,
#     color="lightgray",
    
# )
# ax.axvspan(
#     20, 23,
#     alpha=0.6,
#     color="lightpink",

# )
# ax.axvspan(
#     23, 27,
#     alpha=0.4,
#     color="lightgray",
    
# )
# ax.axvspan(
#     27, 30,
#     alpha=0.6,
#     color="lightpink",

# )

# ax.axvspan(
#     30, 34,
#     alpha=0.4,
#     color="lightgray",
    
# )
# ax.axvspan(
#     34, 37,
#     alpha=0.6,
#     color="lightpink",

# )

# ax.axvspan(
#     37, 41,
#     alpha=0.4,
#     color="lightgray",
    
# )
# ax.axvspan(
#     41, 44,
#     alpha=0.6,
#     color="lightpink",

# )
# ax.axvspan(
#     44, 50,
#     alpha=0.4,
#     color="lightgray",
   
# )

# ax.axvspan(
#     30, 42,
#     alpha=0.15,
#     color="green",
#     label="Drift Area 2"
# )

# # Avoid duplicate legend entries
# handles, labels = ax.get_legend_handles_labels()
# unique = dict(zip(labels, handles))
# ax.legend(
#     unique.values(),
#     unique.keys(),
#     title="Method",
#     bbox_to_anchor=(1, 1),
#     loc="upper left"
# )

plt.title("Global validation accuracy with training rounds", fontsize=18)
plt.xlabel("Round (t)", fontsize=18)
plt.ylabel("Global accuracy (%)", fontsize=16)
plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
plt.tight_layout()
plt.show()

