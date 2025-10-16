# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 19:41:46 2025

@author: Divya
"""

#########################
# Purpose: Utility functions for the gas sensor data
########################

# split_gas_drift.py
import os
import math
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in the CSV.")
    if "batch" not in df.columns:
        raise ValueError("Expected a 'batch' column in the CSV.")
    # Features = everything except 'label' and 'batch'
    X = df.drop(columns=["label", "batch"], errors="ignore")
    y = df["label"]
    batches = df["batch"]
    return df, X, y, batches

def ensure_outdir(outdir):
    os.makedirs(outdir, exist_ok=True)

def save_split(X_train, X_test, y_train, y_test, prefix, outdir):
    X_train.to_csv(os.path.join(outdir, f"X_train_{prefix}.csv"), index=False)
    X_test.to_csv(os.path.join(outdir, f"X_test_{prefix}.csv"), index=False)
    y_train.to_csv(os.path.join(outdir, f"y_train_{prefix}.csv"), index=False)
    y_test.to_csv(os.path.join(outdir, f"y_test_{prefix}.csv"), index=False)

def plot_class_dist(y, title, outfile):
    counts = y.value_counts().sort_index()
    plt.figure()
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel("Class label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def iid_split(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

def chrono_split(df, train_frac=0.7):
    # Split by batch order (simulate drift across time)
    uniq_batches = sorted(df["batch"].unique())
    if len(uniq_batches) < 2:
        raise ValueError("Need at least 2 batches for a chronological split.")
    k = max(1, int(math.floor(len(uniq_batches) * train_frac)))
    train_batches = set(uniq_batches[:k])
    test_batches = set(uniq_batches[k:])

    train_df = df[df["batch"].isin(train_batches)].copy()
    test_df  = df[df["batch"].isin(test_batches)].copy()

    X_train = train_df.drop(columns=["label", "batch"], errors="ignore")
    y_train = train_df["label"]
    X_test  = test_df.drop(columns=["label", "batch"], errors="ignore")
    y_test  = test_df["label"]

    return (X_train, X_test, y_train, y_test, train_batches, test_batches)

def main():
    ap = argparse.ArgumentParser(description="Split Gas Sensor Drift CSV into IID and chronological sets.")
    ap.add_argument("--csv", default="gas_drift_all_batches.csv", help="Path to combined CSV")
    ap.add_argument("--outdir", default="splits", help="Directory to save outputs")
    ap.add_argument("--iid_test_size", type=float, default=0.2, help="IID split test size fraction")
    ap.add_argument("--chrono_train_frac", type=float, default=0.7, help="Fraction of batches for chronological TRAIN")
    ap.add_argument("--seed", type=int, default=42, help="Random state")
    args = ap.parse_args()

    ensure_outdir(args.outdir)
# Current working directory
    current_dir = Path.cwd()

# Go up two levels
    one_up = current_dir.parents[0]  # 0 = parent, 1 = grandparent

    print("Current directory:", current_dir)
    print("One level up:", one_up)
    data_dir = one_up / "data/gas_sensor" / args.csv
    print("Data dir:", data_dir)
	
    df, X, y, batches = load_data(data_dir)

#     # =============== IID (random) split ===============
#     Xtr_iid, Xte_iid, ytr_iid, yte_iid = train_test_split(
#         X, y, test_size=args.iid_test_size, random_state=args.seed, stratify=y
#     )
#     save_split(Xtr_iid, Xte_iid, ytr_iid, yte_iid, prefix="iid", outdir=args.outdir)

#     # Plots for IID
#     plot_class_dist(ytr_iid, "Class distribution (IID Train)", os.path.join(args.outdir, "class_dist_iid_train.png"))
#     plot_class_dist(yte_iid, "Class distribution (IID Test)",  os.path.join(args.outdir, "class_dist_iid_test.png"))

#     # =============== Chronological (by batch) split ===============
#     Xtr_ch, Xte_ch, ytr_ch, yte_ch, train_batches, test_batches = chrono_split(
#         df, train_frac=args.chrono_train_frac
#     )
#     save_split(Xtr_ch, Xte_ch, ytr_ch, yte_ch, prefix="chrono", outdir=args.outdir)

#     # Plots for Chrono
#     plot_class_dist(ytr_ch, f"Class distribution (Chrono Train: batches {sorted(train_batches)})",
#                     os.path.join(args.outdir, "class_dist_chrono_train.png"))
#     plot_class_dist(yte_ch, f"Class distribution (Chrono Test: batches {sorted(test_batches)})",
#                     os.path.join(args.outdir, "class_dist_chrono_test.png"))

#     # =============== Helpful summaries ===============
#     print("=== IID split ===")
#     print("Train shape:", Xtr_iid.shape, " Test shape:", Xte_iid.shape)
#     print("Train label balance (top 10):")
#     print(ytr_iid.value_counts(normalize=True).head(10))
#     print("Test label balance (top 10):")
#     print(yte_iid.value_counts(normalize=True).head(10))

#     print("\n=== Chronological split ===")
#     print("Train batches:", sorted(train_batches))
#     print("Test batches :", sorted(test_batches))
#     print("Train shape:", Xtr_ch.shape, " Test shape:", Xte_ch.shape)
#     print("Train label balance (top 10):")
#     print(ytr_ch.value_counts(normalize=True).head(10))
#     print("Test label balance (top 10):")
#     print(yte_ch.value_counts(normalize=True).head(10))

# if __name__ == "__main__":
#     main()
