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

files  += glob("./data/synthetic-class1/strobfl-strobfl/d4si4.txt")
files  += glob("./data/synthetic-class1/adam-avg/d4si4.txt")

# files  += glob("./data/synthetic-class1/strobfl-strobfl/D4II0.4.txt")
# files  += glob("./data/synthetic-class1/adam-avg/D4II0.4.txt")


#files  += glob("./data/synthetic-class1/strobfl-strobfl/D0-T50K10C0.8B50LR0.1/output_global_eval_loss.txt")
#files  += glob("./data/synthetic-class1/adam-avg/D0-T50K10C0.8B50LR0.1/output_global_eval_loss.txt")

#files  += glob("./data/synthetic-class1/strobfl-strobfl/D1-independentT50K10C0.8B50LR0.1/output_global_eval_loss.txt")
#files  += glob("./data/synthetic-class1/adam-avg/D1-independentT50K10C0.8B50LR0.1/output_global_eval_loss.txt")

# files  += glob("./data/synthetic-class1/strobfl-strobfl/D4-independentT50K10C0.8B50LR0.1/output_global_eval_loss.txt")
#files  += glob("./data/synthetic-class1/adam-avg/D4-independentT50K10C0.8B50LR0.1/output_global_eval_loss.txt")

# files  += glob("./data/synthetic-class1/strobfl-strobfl/D4-sharedT50K10C0.8B50LR0.1/output_global_eval_loss.txt")
# files  += glob("./data/synthetic-class1/adam-avg/D4-sharedT50K10C0.8B50LR0.1/output_global_eval_loss.txt")

# files  += glob("./data/synthetic-class1/strobfl-strobfl/D4IImbalance0.6/output_global_eval_loss.txt")
# files  += glob("./data/synthetic-class1/adam-avg/D4IImbalance0/output_global_eval_loss.txt")


#files  += glob("./data/synthetic-class1/strobfl-avg/d1output_global_eval_loss.txt")
# files  += glob("./data/synthetic-class1/strobfl-avg/d4ioutput_global_eval_loss.txt")
#files  += glob("./data/synthetic-class1/strobfl-avg/d4soutput_global_eval_loss.txt")
#files  += glob("./data/synthetic-class1/strobfl-avg/d0output_global_eval_loss.txt")
#files  += glob("./data/synthetic-class1/strobfl-strobfl/ArrivalRate2/d4ioutput_global_eval_loss.txt")



print("Found files:")
for f in files:
    print(f)

dfs = []
methods = ['STROBFL', 'ADAM']
#methods = ['Adam-No Drift','STROBFL-No Drift','Adam-4D/I','STROBFL-4D/I','Adam-4D/S','STROBFL-4D/S']

values = []

i = 0
for f in files:
    df = pd.read_csv(f)
    # col = df.iloc[:,1]
    # print(f"{f} -> min: {col.min()}, max: {col.max()}")
    # values.append(col.mean()) 
    # print(values)

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
    
    

#     dfs.append(df)

# all_df = pd.concat(dfs, ignore_index=True)
# print(all_df.head())



# sns.set(style="whitegrid")

# plt.figure(figsize=(10, 6))
# ax = sns.lineplot(
#     data=all_df,
#     x="t",
#     y="eval_success",
#     hue="level4",      # method
#     style="level4",    # run
#     markers=True
# )

# ax.legend(
#     title="Method",
#     labelspacing=1.2,   # vertical space between entries
#     handlelength=3.0,   # length of line/marker handle
#     handletextpad=1.2,  # space between handle and text
#     borderpad=1.2       # padding inside legend box
# )

# ax.axvspan(
#     8, 20,
#     alpha=0.15,
#     color="red",
#     label="Concept Drift"
# )

# ax.axvspan(
#     30, 42,
#     alpha=0.15,
#     color="green",
#     label="Covariate Shift"
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

# plt.xlabel("Round (t)")
# plt.ylabel("Validation Accuracy (%)")
# plt.title("Server Validation Accuracy across rounds- 50% shared drift & 0.4 imbalance")
# plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
# plt.tight_layout()
# plt.show()

