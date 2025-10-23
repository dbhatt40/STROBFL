# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 09:36:36 2025

@author: Divya
"""

import pandas as pd
import glob
import re
import os
import numpy as np

def convert_libsvm_to_plain(path_in, path_out, n_features=128):
    X = []
    y = []
    batch = []

    with open(path_in, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            # First token = label
            y_val = int(parts[0])

            # Last token = batch (if no colon)
            if ":" not in parts[-1]:
                batch_val = int(parts[-1])
                features = parts[1:-1]
            else:
                batch_val = np.nan
                features = parts[1:]

            # Initialize all feature values as zeros
            x = np.zeros(n_features, dtype=float)

            # Parse each feature index:value
            for kv in features:
                if ":" in kv:
                    k, v = kv.split(":")
                    idx = int(k) - 1   # 1-based → 0-based
                    if 0 <= idx < n_features:
                        x[idx] = float(v)

            X.append(x)
            y.append(y_val)
            batch.append(batch_val)

    # Combine into one DataFrame
    df = pd.DataFrame(X, columns=[f"f{i+1}" for i in range(n_features)])
    df.insert(0, "label", y)
    df["batch"] = batch

    # Save as CSV
    df.to_csv(path_out, index=False)
    print(f"✅ Saved converted file to: {path_out}")
    return df

# Folder containing the original batch files
input_path = "../data/gas_sensor/batchesDAT"
output_path = "../data/gas_sensor/batchesCSV"

# Create output directory if it doesn’t exist
os.makedirs(output_path, exist_ok=True)

# Get all batch files (batch1.csv ... batch10.csv)
batch_files = sorted(glob.glob(os.path.join(input_path, "batch*.dat")))


for file in batch_files:
    print(f"Processing {file} ...")
    base_name = os.path.splitext(file)[0]  # → "batch1"
    output_file = base_name + ".csv"             # → "batch1.csv"
  
    df_plain = convert_libsvm_to_plain(file, output_file, n_features=128)


print("✅ All batches cleaned and saved.")



# Example usage:

