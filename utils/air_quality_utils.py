# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 16:51:13 2025

@author: Divya
"""

from __future__ import annotations
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pandas as pd

from tensorflow.keras import layers, Model, Input
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Conv1D,MaxPooling1D 
from keras.layers import BatchNormalization, GlobalAveragePooling1D 

from sklearn.preprocessing import StandardScaler

import global_vars as gv

import os
from typing import Dict, Tuple, Optional, Iterable
from sklearn.preprocessing import StandardScaler


def split_clients_xy(
    client_train_dfs: Dict[int, pd.DataFrame],
    *,
    label_col: str,
    drop_extra_cols: Optional[Iterable[str]] = ("datetime", "station_id"),
    as_numpy: bool = True,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Split each client's DataFrame into X_train (features) and Y_train (labels).

    Parameters
    ----------
    client_train_dfs : Dict[int, pd.DataFrame]
        Mapping station_id -> client's full training DataFrame.
    label_col : str
        Column name of the target label.
    drop_extra_cols : Iterable[str] or None
        Columns to drop from features (metadata columns like timestamp, station_id).
    as_numpy : bool
        If True, return numpy arrays instead of DataFrames/Series.

    Returns
    -------
    xy_splits : Dict[int, Tuple[X_train, Y_train]]
        Mapping from station_id to (X_train, Y_train)
    """
    xy_splits: Dict[int, Tuple] = {}

    for sid, df in client_train_dfs.items():
        if label_col not in df.columns:
            raise KeyError(f"Client {sid}: missing label column '{label_col}'.")

        drop_cols = [label_col]
        if drop_extra_cols:
            drop_cols += list(drop_extra_cols)
        drop_cols = [c for c in drop_cols if c in df.columns]

        X_train = df.drop(columns=drop_cols, errors="ignore")
        Y_train = df[label_col]

        if as_numpy:
            X_train = X_train.to_numpy(dtype=np.float32)
            Y_train = Y_train.to_numpy()

        xy_splits[sid] = (X_train, Y_train)

    return xy_splits


def data_air_quality(
    *,
    output_dir: Optional[str] = None,
    station_filter: Optional[Iterable[int]] = None   
) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """
    Read the combined Beijing Air Quality CSV and split into:
      1) A server test set consisting of the earliest `server_test_frac` of *all* rows
         by global timestamp (across all stations).
      2) A dict of client training sets keyed by station_id, from the *remaining* rows.

    Assumptions:
      - CSV is cleaned and contains `timestamp` and numeric `station_id` columns.
      - You want the earliest fraction globally (not per-station) for the server test set.

    Parameters
    ----------
    csv_path : str
        Path to the combined CSV.
    timestamp_col : str
        Name of the timestamp column.
    station_col : str
        Name of the station id column (numeric).
    parse_dates : bool
        If True, parse `timestamp_col` as datetime.
    server_test_frac : float
        Fraction (0–1) of the total data to allocate to the server test set,
        taken from the *beginning* (earliest timestamps).
    output_dir : Optional[str]
        If provided, writes:
          - server_test.csv
          - client_train_<station_id>.csv
        into this directory.
    station_filter : Optional[Iterable[int]]
        If provided, keep only these station ids (useful if you want a subset).
    drop_na : bool
        If True, drop rows with NA in the key columns.

    Returns
    -------
    server_test_df : pd.DataFrame
        Earliest fraction of the entire dataset by timestamp.
    client_train_dfs : Dict[int, pd.DataFrame]
        Mapping from station_id -> training DataFrame (remaining rows).
    """
    server_test_frac = 0.10
    timestamp_col = "datetime"
    station_col = "station_id"
    parse_dates = True
    drop_na = True
    if not 0.0 < server_test_frac < 1.0:
        raise ValueError("server_test_frac must be in (0, 1).")

    # Load
    parse_cols = [timestamp_col] if parse_dates else None
    csv_path = "/content/STROBFL/data/air_quality/AirQuality_Clean.csv"  
    df = pd.read_csv(csv_path, parse_dates=parse_cols)

    # Basic checks / cleanup
    if timestamp_col not in df.columns:
        raise KeyError(f"Missing timestamp column '{timestamp_col}' in CSV.")
    if station_col not in df.columns:
        raise KeyError(f"Missing station id column '{station_col}' in CSV.")

    # Ensure correct dtypes
    if parse_dates and not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")

    # Enforce numeric station ids
    df[station_col] = pd.to_numeric(df[station_col], errors="coerce").astype("Int64")

    # Optional NA drop on critical cols
    if drop_na:
        df = df.dropna(subset=[timestamp_col, station_col])

    # Optional subset of stations
    if station_filter is not None:
        station_filter = set(int(s) for s in station_filter)
        df = df[df[station_col].astype(int).isin(station_filter)]

    # Sort globally by timestamp (ascending: earliest first)
    df = df.sort_values(by=[timestamp_col, station_col]).reset_index(drop=True)

    # Compute split index (earliest 10% for server test set)
    n_total = len(df)
    n_test = max(1, int(round(server_test_frac * n_total)))
    server_test_df = df.iloc[:n_test].copy()
    remaining_df = df.iloc[n_test:].copy()

    # Build per-client training sets from the remaining rows
    client_train_dfs: Dict[int, pd.DataFrame] = {}
    if not remaining_df.empty:
        # Keep the intra-station chronological order
        remaining_df = remaining_df.sort_values(by=[station_col, timestamp_col]).reset_index(drop=True)
        for sid, g in remaining_df.groupby(station_col, sort=True):
            # sid is pandas Int64; convert to int for dict key
            client_train_dfs[int(sid)] = g.reset_index(drop=True)

# =============================================================================
#     # Optionally write to disk
#     if output_dir is not None:
#         os.makedirs(output_dir, exist_ok=True)
#         server_test_df.to_csv(os.path.join(output_dir, "server_test.csv"), index=False)
#         for sid, g in client_train_dfs.items():
#             g.to_csv(os.path.join(output_dir, f"client_train_{sid}.csv"), index=False)
# 	
# =============================================================================
    label_col = "PM2.5"   # or whatever your target column is
    drop_cols = [timestamp_col, station_col]  # metadata columns not used as features

    y_test = server_test_df[label_col].values               # shape (n_samples,)
    X_test = server_test_df.drop(columns=drop_cols + [label_col]).values  # shape (n_samples, n_features)
    client_xy = split_clients_xy(client_train_dfs,
        label_col="PM2.5",          # change to your actual target column
        drop_extra_cols=(timestamp_col, station_col),
        as_numpy=True
    )
    x_train,y_train = client_xy[0]
    X_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train.reshape(-1,1))
	
    X_test_scaler = X_scaler. transform(X_test)
    y_test_scaler = y_scaler.transform(y_test.reshape(-1,1))


    scaled_client_xy={}
    for sid, (X_train,Y_train) in client_xy.items():
        X_train_scaled = X_scaler.fit_transform(X_train)
        Y_train_scaled = y_scaler.fit_transform(Y_train.reshape(-1,1))		
        scaled_client_xy[sid] = (X_train_scaled,Y_train_scaled)
    return scaled_client_xy, X_test_scaler, y_test_scaler


def airquality_model():
	inp = Input(shape=(gv.DATA_DIM,), name='main_input')
	
	x = layers.Dense(128, activation="relu")(inp)
	x = layers.Dropout(0.2)(x)
	
	x = layers.Dense(64, activation="relu")(x)
	x = layers.Dropout(0.2)(x)
	
	out = layers.Dense(1, activation="linear")(x)  # regression output
	model = Model(inp, out)
	 	

	return model