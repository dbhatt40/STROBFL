# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 05:31:27 2025

@author: Divya
"""

import pandas as pd
import matplotlib.pyplot as plt

# --- Read the three text files ---
file1 = "output_global_eval_loss_census_label.txt"
file2 = "output_global_eval_loss_sensor_label.txt"
# file3 = "output_global_eval_loss_strobfl2.txt"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
# df3 = pd.read_csv(file3)

# --- Plot eval_loss ---
plt.figure(figsize=(8, 5))
plt.plot(df1['t'], df1['eval_loss'], label='census', color='red', linewidth=2)
plt.plot(df2['t'], df2['eval_loss'], label='uci-sensor', color='blue', linewidth=2)
# plt.plot(df3['t'], df3['eval_loss'], label='strobfl', color='green', linewidth=2)

plt.xlabel('t (round)')
plt.ylabel('Evaluation Loss')
plt.title('STROBFL Evaluation Loss using Label Loss')
plt.legend()
plt.grid(True)
plt.show()

# --- Optionally, plot accuracy on a separate graph ---
plt.figure(figsize=(8, 5))
plt.plot(df1['t'], df1['eval_success'], label='census', color='red', linewidth=2)
plt.plot(df2['t'], df2['eval_success'], label='uci-sensor', color='blue', linewidth=2)
# plt.plot(df3['t'], df3['eval_success'], label='strobfl', color='green', linewidth=2)

plt.xlabel('t (round)')
plt.ylabel('Accuracy')
plt.title('STROBFL Accuracy using Label Loss')
plt.legend()
plt.grid(True)
plt.show()
