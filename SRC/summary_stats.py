#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# -------------------------
# Config
# -------------------------
NODES_PATH = "/Users/danieldominguez/Documents/Code/DDNN_ILOH/data/daily_drivers/hydro_temp_data_trainval_2013plus.parquet"
SPLIT_DATE = pd.Timestamp("2025-01-01")

STREAMORDER_COL = "streamorde"
COMID_COL = "comid"
DATE_COL = "date"
Q_COL = "q_m3s_obs"
T_COL = "water_temp_C"

# Sentinel / placeholder values to treat as missing
BAD_VALUES = [-9998, -9998.0, -9999, -9999.0]

# -------------------------
# Load data
# -------------------------
df = pd.read_parquet(NODES_PATH)

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df[STREAMORDER_COL] = pd.to_numeric(df[STREAMORDER_COL], errors="coerce")
df[Q_COL] = pd.to_numeric(df[Q_COL], errors="coerce")
df[T_COL] = pd.to_numeric(df[T_COL], errors="coerce")

# Drop rows with missing stream order
df = df.dropna(subset=[STREAMORDER_COL]).copy()
df[STREAMORDER_COL] = df[STREAMORDER_COL].astype(int)

# Convert sentinel values to NaN
df.loc[df[Q_COL].isin(BAD_VALUES), Q_COL] = np.nan
df.loc[df[T_COL].isin(BAD_VALUES), T_COL] = np.nan

# True-observation flags
df["q_is_obs"] = df[Q_COL].notna()
df["t_is_obs"] = df[T_COL].notna()

# -------------------------
# Train/test split
# -------------------------
train_df = df[df[DATE_COL] < SPLIT_DATE].copy()
test_df  = df[df[DATE_COL] >= SPLIT_DATE].copy()

# -------------------------
# Observation counts by stream order
# -------------------------
train_counts = (
    train_df.groupby(STREAMORDER_COL)
    .agg(
        train_q_observations=("q_is_obs", "sum"),
        train_t_observations=("t_is_obs", "sum")
    )
    .reset_index()
)

test_counts = (
    test_df.groupby(STREAMORDER_COL)
    .agg(
        test_q_observations=("q_is_obs", "sum"),
        test_t_observations=("t_is_obs", "sum")
    )
    .reset_index()
)

total_counts = (
    df.groupby(STREAMORDER_COL)
    .agg(
        total_q_observations=("q_is_obs", "sum"),
        total_t_observations=("t_is_obs", "sum")
    )
    .reset_index()
)

obs_counts = (
    train_counts
    .merge(test_counts, on=STREAMORDER_COL, how="outer")
    .merge(total_counts, on=STREAMORDER_COL, how="outer")
    .fillna(0)
    .sort_values(STREAMORDER_COL)
)

obs_cols = [
    "train_q_observations", "train_t_observations",
    "test_q_observations", "test_t_observations",
    "total_q_observations", "total_t_observations"
]
obs_counts[obs_cols] = obs_counts[obs_cols].astype(int)

# -------------------------
# Sensor counts by stream order
# Count unique COMIDs only where there is a true observation
# -------------------------
q_sensor_counts = (
    df[df["q_is_obs"]]
    .groupby(STREAMORDER_COL)[COMID_COL]
    .nunique()
    .reset_index(name="n_q_sensors")
    .sort_values(STREAMORDER_COL)
)

t_sensor_counts = (
    df[df["t_is_obs"]]
    .groupby(STREAMORDER_COL)[COMID_COL]
    .nunique()
    .reset_index(name="n_t_sensors")
    .sort_values(STREAMORDER_COL)
)

sensor_counts = (
    q_sensor_counts
    .merge(t_sensor_counts, on=STREAMORDER_COL, how="outer")
    .fillna(0)
    .sort_values(STREAMORDER_COL)
)

sensor_counts[["n_q_sensors", "n_t_sensors"]] = sensor_counts[
    ["n_q_sensors", "n_t_sensors"]
].astype(int)

# -------------------------
# Stream order 7 summary
# -------------------------
so7 = df[df[STREAMORDER_COL] == 7].copy()
so7_obs = so7[so7["q_is_obs"] | so7["t_is_obs"]].copy()

if len(so7_obs) > 0:
    so7_min_date = so7_obs[DATE_COL].min()
    so7_max_date = so7_obs[DATE_COL].max()
    so7_n_q_obs = int(so7_obs["q_is_obs"].sum())
    so7_n_t_obs = int(so7_obs["t_is_obs"].sum())
    so7_n_q_sensors = so7.loc[so7["q_is_obs"], COMID_COL].nunique()
    so7_n_t_sensors = so7.loc[so7["t_is_obs"], COMID_COL].nunique()
else:
    so7_min_date = None
    so7_max_date = None
    so7_n_q_obs = 0
    so7_n_t_obs = 0
    so7_n_q_sensors = 0
    so7_n_t_sensors = 0

# -------------------------
# Debug check: show raw distinct COMIDs with observed Q/T
# -------------------------
print("\n=== Debug: total unique COMIDs with true observations ===")
print("Q sensors total:", df.loc[df["q_is_obs"], COMID_COL].nunique())
print("T sensors total:", df.loc[df["t_is_obs"], COMID_COL].nunique())

# -------------------------
# Print results
# -------------------------
print("\n=== Q and T observation counts by stream order ===")
print(obs_counts.to_string(index=False))

print("\n=== Sensor counts by stream order ===")
print(sensor_counts.to_string(index=False))

print("\n=== Stream order 7 summary ===")
print(f"n_q_observations: {so7_n_q_obs}")
print(f"n_t_observations: {so7_n_t_obs}")
print(f"n_q_sensors: {so7_n_q_sensors}")
print(f"n_t_sensors: {so7_n_t_sensors}")
print(f"min_date: {so7_min_date}")
print(f"max_date: {so7_max_date}")

# -------------------------
# Save outputs
# -------------------------
obs_counts.to_csv("qt_observation_counts_by_streamorder.csv", index=False)
sensor_counts.to_csv("sensor_counts_by_streamorder.csv", index=False)