# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 18:03:14 2025

@author: Divya

"""

import os
import glob
import pandas as pd
import numpy as np

# ---------- SETTINGS ----------
DATA_DIR   = '/../STROBFL/data/gas_sensor/'  # change if needed
GLOB_EXT   = ".dat"                         # switch to ".csv" if you converted
OUTPUT_CSV = "gas_drift_all_batches.csv"
OUTPUT_PARQUET = "gas_drift_all_batches.parquet"
# --------------------------------

def _read_one(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dat":
        # whitespace-delimited; many files have variable spacing
        df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    elif ext == ".csv":
        df = pd.read_csv(path, header=None)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    # Drop empty columns (can appear due to trailing spaces)
    df = df.dropna(axis=1, how="all")
    return df

def _guess_label_column(df):
    """
    Heuristic:
      - If the last column has a small number of unique integer-like values (<= 50),
        treat it as 'label'.
      - Otherwise, return None (no label detected).
    """
    last = df.columns[-1]
    col = df[last]
    # Integer-like check (allow floats that are whole numbers)
    as_int = pd.to_numeric(col, errors="coerce")
    if as_int.isna().mean() < 0.05:  # mostly numeric
        uniq = pd.unique(as_int.dropna())
        # Small number of classes (common for 6 gases)
        if len(uniq) <= 50:
            # also require most values be close to integers
            if np.all(np.isclose(uniq, np.round(uniq))):
                return last
    return None

def _add_batch_and_rename(df, batch_id, label_col):
    df = df.copy()
    df["batch"] = batch_id
    if label_col is not None:
        df = df.rename(columns={label_col: "label"})
    return df

# Collect files (sorted by the number in 'batch#.ext')
print(os.getcwd())
paths = sorted(	
    glob.glob(os.path.join(DATA_DIR, f"*{GLOB_EXT}")),
    key=lambda p: (
        # prioritize batch#.ext ordering
        int(''.join([c for c in os.path.basename(p) if c.isdigit()]) or 999)
    )
)

if not paths:
    raise FileNotFoundError(
        f"No {GLOB_EXT} files found in '{DATA_DIR}'. "
        f"Check DATA_DIR and file extensions."
    )

combined = []
for p in paths:
    name = os.path.basename(p).lower()
    # Try to pull batch id from filename like "batch7.dat"
    batch_id = None
    for tok in os.path.splitext(name)[0].replace('-', '_').split('_'):
        if tok.startswith("batch"):
            digits = ''.join([c for c in tok if c.isdigit()])
            if digits.isdigit():
                batch_id = int(digits)
                break
    if batch_id is None:
        # Fallback: find any digits in the filename
        digits = ''.join([c for c in name if c.isdigit()])
        batch_id = int(digits) if digits.isdigit() else -1

    df = _read_one(p)
    label_col = _guess_label_column(df)

    # If we detected a label in the last column,
    # split features vs label for cleaner names
    if label_col is not None:
        feature_cols = [c for c in df.columns if c != label_col]
        # Create friendly feature names f1..fN
        rename_map = {c: f"f{idx+1}" for idx, c in enumerate(feature_cols)}
        # Keep label name as-is (renamed to 'label' below)
        df = df.rename(columns=rename_map)

    df = _add_batch_and_rename(df, batch_id, label_col)
    combined.append(df)

all_df = pd.concat(combined, axis=0, ignore_index=True)

# Ensure column order: features..., label (if present), batch
cols = list(all_df.columns)
if "label" in cols:
    cols = [c for c in cols if c not in ("label", "batch")] + ["label", "batch"]
else:
    cols = [c for c in cols if c != "batch"] + ["batch"]
all_df = all_df[cols]

# Save
all_df.to_csv(OUTPUT_CSV, index=False)
try:
    all_df.to_parquet(OUTPUT_PARQUET, index=False)
except Exception as e:
    print(f"(Parquet save skipped: {e})")

print("Saved:")
print(f"  - {OUTPUT_CSV}  (rows={len(all_df):,}, cols={all_df.shape[1]})")
if os.path.exists(OUTPUT_PARQUET):
    print(f"  - {OUTPUT_PARQUET}")

# Quick sanity summaries
print("\nHead:")
print(all_df.head())

if "label" in all_df.columns:
    print("\nClass distribution by batch (first 10):")
    print(all_df.groupby(["batch", "label"]).size().reset_index(name="count").head(10))
else:
    print("\n(No label column detected — check the readme or set label manually.)")
