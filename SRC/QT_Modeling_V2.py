#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader, Subset
from torch_geometric.nn import GATConv
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import joblib
import shap  # optional; kept per your script


# -------------------------
# Config
# -------------------------
NODES_PATH = "/Users/danieldominguez/Documents/Code/DDNN_ILOH/data/daily_drivers/hydro_temp_data_trainval_2013plus.parquet"
EDGES_PATH = "/Users/danieldominguez/Documents/Code/DDNN_ILOH/data/daily_drivers/edge_list.parquet"

SPLIT_DATE   = pd.Timestamp("2025-01-01")   # train < 2025-01-01; test >= 2025-01-01
LOOKBACK     = 7

H_LSTM       = 64
LSTM_LAYERS  = 1
LSTM_DROPOUT = 0.2

H_GNN        = 64
GNN_DROPOUT  = 0.2
GAT_HEADS    = 4

EPOCHS       = 200
LR           = 1e-4
BATCH_SIZE   = 1   # each batch = one time step window over all nodes
PATIENCE     = 20

# Only the true meteorological + basin drivers you want (no Flow columns)
BASE_FEATURES = [
    "pr", "srad", "vs", "vpd", "sph",
    "tmax_c", "tmin_c", "tmean_c",
    "rmin", "rmax", "Ra", "bsn_pr"
]

# Explicit statics + seasonality (must already exist in parquet)
STATIC_FEATURES = ["area_m2", "Slope", "LengthKM", "streamorde"]
SEASONAL_FEATURES = ["DOY_sin", "DOY_cos"]

Q_COL    = "q_m3s_obs"
T_COL    = "water_temp_C"
TARGETS  = [Q_COL, T_COL]

# weights for [Q, T] terms in the mixed loss
TASK_WEIGHTS = torch.tensor([0.7, 0.3], dtype=torch.float32)

# ---- NEW: Q weighting for high flows (> 100 m3/s) ----
Q_HIGH_THRESH_M3S = 100.0     # threshold in physical units
Q_HIGH_MAX_MULT   = 5.0       # max multiplicative weight applied to Q loss at highest flows
Q_HIGH_POWER      = 1.0       # 1.0 = linear ramp; >1 emphasizes extremes more

TARGET_COMID_FOR_PLOTS = 14787569

SAVE_DIR = "/Users/danieldominguez/Documents/Code/DDNN_ILOH/data/models"
FEATURES_PATH = os.path.join(SAVE_DIR, "features.pkl")  # persists feature order for downstream inference
EDGE_SUSPICIOUS_OUT = os.path.join(SAVE_DIR, "edge_suspicious_checks.csv")

# Edge settings
USE_COARSENED_GRAPH = True
MAX_HOPS_COARSEN    = 5000
COARSE_MODE         = "nearest"  # "nearest" downstream observed; (keeps graph sparse)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Helpers
# -------------------------
SENTINEL_MISS = -9998.0

def replace_sentinel_with_nan(df: pd.DataFrame, cols):
    """Replace -9998 sentinels with NaN in selected numeric columns."""
    df = df.copy()
    for c in cols:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            df.loc[df[c] == SENTINEL_MISS, c] = np.nan
    return df

def print_feature_sanity_ranges(nodes_df_raw: pd.DataFrame, FEATURES: list):
    """Quick sanity print for a few statics to catch order/magnitude issues early."""
    check_cols = [c for c in ["area_m2", "areasqkm", "Slope", "LengthKM", "streamorde"] if c in FEATURES]
    if not check_cols:
        return
    print("\n=== Raw feature sanity (min/max) for key statics ===")
    for c in check_cols:
        v = nodes_df_raw[c].to_numpy()
        v = v[np.isfinite(v)]
        if v.size == 0:
            print(f"{c:>10s}: all NA")
        else:
            print(f"{c:>10s}: min={np.min(v):.6g}  max={np.max(v):.6g}")

def scaler_feature_diagnostic(scaler: MinMaxScaler, FEATURES: list, max_lines=40):
    """Print scaler min/max pairs to verify they align with expected magnitudes."""
    if not hasattr(scaler, "data_min_") or not hasattr(scaler, "data_max_"):
        print("\nWARNING: scaler missing data_min_/data_max_; cannot run scaler diagnostics.")
        return

    mins = scaler.data_min_.astype(float)
    maxs = scaler.data_max_.astype(float)

    priority = [f for f in ["area_m2", "areasqkm", "Slope", "LengthKM", "streamorde",
                            "pr", "srad", "tmean_c", "vs", "vpd", "sph",
                            "tmax_c", "tmin_c", "rmin", "rmax", "Ra", "bsn_pr"]
                if f in FEATURES]
    fallback = [f for f in FEATURES[:min(len(FEATURES), 12)] if f not in priority]
    show = (priority + fallback)[:max_lines]

    print("\n=== Scaler_X training min/max (selected FEATURES) ===")
    for f in show:
        j = FEATURES.index(f)
        print(f"{f:>10s} | train_min={mins[j]:.6g}  train_max={maxs[j]:.6g}")

def build_edge_pairs_from_nodes(edges_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct upstream->downstream COMID edges by joining on node ids:
      (comid_up, tonode) matches (comid_dn, fromnode)
    Returns edge_pairs_all with columns [comid_up, comid_dn] (deduped).
    """
    edges_df = edges_df.dropna(subset=["fromnode", "tonode", "comid"]).copy()
    up = edges_df[["comid", "fromnode", "tonode"]].rename(columns={"comid": "comid_up"})
    dn = edges_df[["comid", "fromnode", "tonode"]].rename(columns={"comid": "comid_dn"})

    edge_pairs_all = (
        up.merge(dn, left_on="tonode", right_on="fromnode", how="inner")[["comid_up", "comid_dn"]]
          .drop_duplicates()
          .reset_index(drop=True)
    )
    return edge_pairs_all

def build_coarsened_edges(edge_pairs_all: pd.DataFrame, observed_comids: set,
                          max_hops=5000, mode="nearest") -> pd.DataFrame:
    """
    Collapse the full COMID network down to edges between observed COMIDs only.

    For each observed COMID, walk downstream through the full graph until you hit another
    observed COMID. Add an edge (src_observed -> downstream_observed).
      mode="nearest": stop at first encountered observed downstream node.
    """
    dn_map = {}
    for u, v in edge_pairs_all[["comid_up", "comid_dn"]].itertuples(index=False):
        try:
            u_i = int(u); v_i = int(v)
        except Exception:
            continue
        dn_map.setdefault(u_i, []).append(v_i)

    coarse_edges = set()

    for src in observed_comids:
        src = int(src)
        seen = {src}
        frontier = [src]
        hops = 0

        while frontier and hops < max_hops:
            curr = frontier.pop()

            for dn in dn_map.get(curr, []):
                if dn in seen:
                    continue
                seen.add(dn)

                if dn in observed_comids and dn != src:
                    coarse_edges.add((src, dn))
                    if mode == "nearest":
                        frontier = []
                        break
                else:
                    frontier.append(dn)

            hops += 1

    return pd.DataFrame(list(coarse_edges), columns=["comid_up", "comid_dn"])

def edge_qa_checks(edge_pairs: pd.DataFrame, nodes_df_raw: pd.DataFrame, all_comids: np.ndarray,
                   FEATURES: list, save_csv_path: str = None) -> None:
    """
    QA checks for edge health.
    """
    print("\n====================")
    print("EDGE QA CHECKS")
    print("====================")

    n_nodes = len(all_comids)
    n_edges = len(edge_pairs)

    print(f"Nodes in modeling set: {n_nodes}")
    print(f"Edges after filtering to modeling COMIDs: {n_edges}")
    print(f"Edge density: {n_edges / max(n_nodes,1):.4f} edges/node")

    if n_edges == 0:
        print("WARNING: No edges in graph after filtering/coarsening. Model will behave like per-node LSTM only.")
        return

    self_loops = int((edge_pairs["comid_up"].astype(int) == edge_pairs["comid_dn"].astype(int)).sum())
    dup_edges = int(edge_pairs.duplicated(subset=["comid_up", "comid_dn"]).sum())

    nodes_up = set(edge_pairs["comid_up"].astype(int).unique())
    nodes_dn = set(edge_pairs["comid_dn"].astype(int).unique())
    nodes_any = nodes_up | nodes_dn

    print(f"Nodes appearing as upstream:   {len(nodes_up)} ({100*len(nodes_up)/n_nodes:.2f}%)")
    print(f"Nodes appearing as downstream: {len(nodes_dn)} ({100*len(nodes_dn)/n_nodes:.2f}%)")
    print(f"Nodes appearing anywhere:      {len(nodes_any)} ({100*len(nodes_any)/n_nodes:.2f}%)")
    print(f"Self-loops: {self_loops}")
    print(f"Duplicate edges: {dup_edges}")

    up_counts = edge_pairs["comid_up"].astype(int).value_counts()
    dn_counts = edge_pairs["comid_dn"].astype(int).value_counts()

    out_deg = np.array([int(up_counts.get(int(c), 0)) for c in all_comids])
    in_deg  = np.array([int(dn_counts.get(int(c), 0)) for c in all_comids])

    print("\n--- Degree summary (all nodes) ---")
    print(f"Out-degree: mean={out_deg.mean():.3f}, max={out_deg.max()}, pct_zero={100*(out_deg==0).mean():.2f}%")
    print(f"In-degree:  mean={in_deg.mean():.3f}, max={in_deg.max()}, pct_zero={100*(in_deg==0).mean():.2f}%")

    print("\nEDGE QA DONE.")


# -------------------------
# Load node/edge data
# -------------------------
nodes_df_raw = pd.read_parquet(NODES_PATH)
print("NODES_PATH:", NODES_PATH)
print("nodes_df_raw rows:", len(nodes_df_raw))
print("unique COMIDs in nodes_df_raw:", nodes_df_raw["comid"].nunique())
print("min date:", nodes_df_raw["date"].min(), "max date:", nodes_df_raw["date"].max())

edges_df = pd.read_parquet(EDGES_PATH)

assert {"comid", "date", Q_COL}.issubset(nodes_df_raw.columns), f"{Q_COL} missing"
assert T_COL in nodes_df_raw.columns, f"{T_COL} missing"

nodes_df_raw["date"] = pd.to_datetime(nodes_df_raw["date"])
nodes_df_raw = nodes_df_raw.sort_values(["date", "comid"]).reset_index(drop=True)

# Preserve in-situ availability for Temp BEFORE any filling/scaling
nodes_df_raw["temp_avail_raw"] = ~nodes_df_raw[T_COL].isna()

# -------------------------
# Explicit FEATURES
# -------------------------
FEATURES = BASE_FEATURES + SEASONAL_FEATURES + STATIC_FEATURES

os.makedirs(SAVE_DIR, exist_ok=True)

required_cols = {"comid", "date", Q_COL, T_COL, "temp_avail_raw"} | set(FEATURES)
missing_cols = [c for c in sorted(required_cols) if c not in nodes_df_raw.columns]
if missing_cols:
    raise ValueError("Missing required columns in nodes_df_raw:\n" + "\n".join(missing_cols))

non_numeric = [f for f in FEATURES if not pd.api.types.is_numeric_dtype(nodes_df_raw[f])]
if non_numeric:
    raise TypeError("These FEATURES are not numeric dtypes:\n" + "\n".join(non_numeric))

joblib.dump(FEATURES, FEATURES_PATH)
print(f"Using explicit FEATURES (n={len(FEATURES)}), saved to: {FEATURES_PATH}")
print(FEATURES)

keep_cols = ["comid", "date"] + FEATURES + TARGETS + ["temp_avail_raw"]
nodes_df = nodes_df_raw[keep_cols].copy()

# -------------------------
# Replace -9998 with NA (recommended before scaling)
# -------------------------
nodes_df = replace_sentinel_with_nan(nodes_df, FEATURES + TARGETS)
print_feature_sanity_ranges(nodes_df, FEATURES)

# -------------------------
# Record where Q is exactly 0 (after sentinel->NaN, before any nan_to_num)
#   We'll later replace those *scaled Q values* with the mean scaled Q (train-only mean).
# -------------------------
q_is_zero_mask_rows = (nodes_df[Q_COL].notna()) & (nodes_df[Q_COL] == 0.0)
print(f"\nQ==0 rows (pre-scale): {int(q_is_zero_mask_rows.sum())} / {len(nodes_df)} "
      f"({100*q_is_zero_mask_rows.mean():.4f}%)")

# -------------------------
# COMID indexing
# -------------------------
all_comids = np.sort(nodes_df["comid"].unique())
comid_to_idx = {c: i for i, c in enumerate(all_comids)}
n_nodes = len(all_comids)

# -------------------------
# Build edge list (upstream -> downstream) and map to indices
# -------------------------
edge_pairs_all = build_edge_pairs_from_nodes(edges_df)

edge_pairs_direct = edge_pairs_all[
    edge_pairs_all["comid_up"].isin(all_comids) & edge_pairs_all["comid_dn"].isin(all_comids)
].drop_duplicates().reset_index(drop=True)

if USE_COARSENED_GRAPH:
    observed_comids = set(all_comids.astype(int))
    edge_pairs = build_coarsened_edges(
        edge_pairs_all=edge_pairs_all,
        observed_comids=observed_comids,
        max_hops=MAX_HOPS_COARSEN,
        mode=COARSE_MODE
    )
    edge_pairs["comid_up"] = edge_pairs["comid_up"].astype(int)
    edge_pairs["comid_dn"] = edge_pairs["comid_dn"].astype(int)
    edge_pairs = edge_pairs[
        edge_pairs["comid_up"].isin(observed_comids) & edge_pairs["comid_dn"].isin(observed_comids)
    ].drop_duplicates().reset_index(drop=True)

    print("\n=== Edge diagnostics (COARSENED) ===")
    print(f"edge_pairs_direct edges (direct-only): {len(edge_pairs_direct)}")
    print(f"edge_pairs coarsened edges: {len(edge_pairs)}")
else:
    edge_pairs = edge_pairs_direct
    print("\n=== Edge diagnostics (DIRECT ONLY) ===")
    print(f"edge_pairs filtered edges: {len(edge_pairs)}")

if len(edge_pairs) > 0:
    src = edge_pairs["comid_up"].map(comid_to_idx).to_numpy()
    dst = edge_pairs["comid_dn"].map(comid_to_idx).to_numpy()
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long).to(device)
else:
    edge_index = torch.empty((2, 0), dtype=torch.long).to(device)

print(f"Graph: {n_nodes} nodes, {edge_index.shape[1]} directed edges")
edge_qa_checks(edge_pairs, nodes_df_raw, all_comids, FEATURES, save_csv_path=EDGE_SUSPICIOUS_OUT)

# -------------------------
# Scale features/targets with train-only fit
# -------------------------
train_mask_rows = nodes_df["date"] < SPLIT_DATE
if train_mask_rows.sum() == 0:
    raise ValueError("No training rows found before SPLIT_DATE. Check SPLIT_DATE or data date range.")

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# NOTE: still using nan_to_num for scaler fit/transform; we'll overwrite scaled Q for Q==0 after transform.
fit_X = np.nan_to_num(nodes_df.loc[train_mask_rows, FEATURES].values, nan=0.0)
fit_y = np.nan_to_num(nodes_df.loc[train_mask_rows, TARGETS].values, nan=0.0)

scaler_X.fit(fit_X)
scaler_y.fit(fit_y)

if hasattr(scaler_X, "n_features_in_") and scaler_X.n_features_in_ != len(FEATURES):
    raise RuntimeError(f"scaler_X expects {scaler_X.n_features_in_} features but FEATURES has {len(FEATURES)}.")

scaler_feature_diagnostic(scaler_X, FEATURES, max_lines=40)

# Transform all rows using train-fitted scalers
X_all = np.nan_to_num(nodes_df[FEATURES].values, nan=0.0)
Y_all = np.nan_to_num(nodes_df[TARGETS].values, nan=0.0)
nodes_df[FEATURES] = scaler_X.transform(X_all)
nodes_df[TARGETS]  = scaler_y.transform(Y_all)

# -------------------------
# NEW: Replace scaled Q where raw Q was exactly 0 with the mean of scaled Q (train-only mean).
#      This prevents the model from seeing those zero-Q observations as "hard zeros" in scaled space.
# -------------------------
q_idx = TARGETS.index(Q_COL)

# scaled Q values in train, excluding raw Q==0 (and excluding NaN, though NaN already became 0.0 in Y_all)
train_nonzero_mask = train_mask_rows & (~q_is_zero_mask_rows) & nodes_df_raw[Q_COL].notna()
if train_nonzero_mask.sum() == 0:
    raise ValueError("No non-zero Q observations found in training period to compute mean scaled Q.")

mean_scaled_q_train = float(nodes_df.loc[train_nonzero_mask, Q_COL].mean())
nodes_df.loc[q_is_zero_mask_rows, Q_COL] = mean_scaled_q_train

print(f"\nFilled scaled Q for raw Q==0 rows with mean_scaled_q_train={mean_scaled_q_train:.6f}")
print(f"Number of Q==0 rows filled: {int(q_is_zero_mask_rows.sum())}")

# -------------------------
# Precompute the scaled threshold corresponding to Q_HIGH_THRESH_M3S for weighted Q loss
#   MinMaxScaler is per-column, so:
#     q_scaled = (q_phys - q_min) / (q_max - q_min)
# -------------------------
q_min = float(scaler_y.data_min_[q_idx])
q_max = float(scaler_y.data_max_[q_idx])
q_range = max(q_max - q_min, 1e-12)
q_thresh_scaled = float((Q_HIGH_THRESH_M3S - q_min) / q_range)
q_thresh_scaled = float(np.clip(q_thresh_scaled, 0.0, 1.0))

print(f"\nQ weighting setup:")
print(f"  Q_HIGH_THRESH_M3S = {Q_HIGH_THRESH_M3S}")
print(f"  scaler_y q_min/q_max = {q_min:.6g} / {q_max:.6g}")
print(f"  q_thresh_scaled = {q_thresh_scaled:.6f}")
print(f"  Q_HIGH_MAX_MULT = {Q_HIGH_MAX_MULT}, Q_HIGH_POWER = {Q_HIGH_POWER}")

# -------------------------
# Dense tensors
# X: [T, N, F], Y: [T, N, 2], TempMask: [T, N]
# -------------------------
dates = np.sort(nodes_df["date"].unique())
Tsteps = len(dates)
Fdim = len(FEATURES)

X_TNF     = np.zeros((Tsteps, n_nodes, Fdim), dtype=np.float32)
Y_TN2     = np.zeros((Tsteps, n_nodes, 2),   dtype=np.float32)
Tmask_TN  = np.zeros((Tsteps, n_nodes),      dtype=bool)

date_to_idx = {d: i for i, d in enumerate(dates)}

for d in dates:
    t = date_to_idx[d]
    sub = nodes_df[nodes_df["date"] == d]
    idxs = sub["comid"].map(comid_to_idx).to_numpy()
    X_TNF[t, idxs, :] = sub[FEATURES].to_numpy(dtype=np.float32)
    Y_TN2[t, idxs, 0] = sub[[Q_COL]].to_numpy(dtype=np.float32).ravel()
    Y_TN2[t, idxs, 1] = sub[[T_COL]].to_numpy(dtype=np.float32).ravel()
    Tmask_TN[t, idxs] = sub["temp_avail_raw"].to_numpy(dtype=bool)

# -------------------------
# Dataset + Dataloader (time-based split uses target date at t)
# -------------------------
class GraphSeqDataset(Dataset):
    def __init__(self, X_TNF, Y_TN2, Tmask_TN, lookback, dates):
        self.X = X_TNF
        self.Y = Y_TN2
        self.M = Tmask_TN
        self.lb = lookback
        self.dates = dates

    def __len__(self):
        return self.X.shape[0] - self.lb

    def __getitem__(self, idx):
        t = idx + self.lb              # target time index in [lb .. T-1]
        t0 = t - self.lb + 1
        t1 = t + 1

        x_win = self.X[t0:t1, :, :]  # [lb, N, F]
        x_win = np.transpose(x_win, (1, 0, 2)).astype(np.float32)  # [N, lb, F]

        y_t   = self.Y[t, :, :].astype(np.float32)     # [N, 2]
        m_t   = self.M[t, :].astype(bool)              # [N]

        return torch.from_numpy(x_win), torch.from_numpy(y_t), torch.from_numpy(m_t)

dataset = GraphSeqDataset(X_TNF, Y_TN2, Tmask_TN, LOOKBACK, dates)

t_dates = dates[LOOKBACK:]  # dates aligned to dataset indices
train_indices = [i for i, d in enumerate(t_dates) if d < SPLIT_DATE]
test_indices  = [i for i, d in enumerate(t_dates) if d >= SPLIT_DATE]

assert len(train_indices) > 0, "No training samples before split date."
assert len(test_indices)  > 0, "No testing samples on/after split date."

train_ds = Subset(dataset, train_indices)
test_ds  = Subset(dataset,  test_indices)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f"\nTraining on {len(train_ds)} time steps (< {SPLIT_DATE.date()}); "
      f"testing on {len(test_ds)} time steps (≥ {SPLIT_DATE.date()}).")

# -------------------------
# Model: LSTM + 2×GAT, two-output head (Q, Temp)
# -------------------------
class GraphLSTM_MTL(nn.Module):
    def __init__(self, in_dim, lstm_hidden, lstm_layers, lstm_dropout,
                 gnn_hidden, gnn_dropout, heads=4, out_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=(lstm_dropout if lstm_layers > 1 else 0.0),
            batch_first=True
        )
        self.drop_lstm = nn.Dropout(lstm_dropout)
        self.gat1 = GATConv(lstm_hidden, gnn_hidden // heads, heads=heads, dropout=gnn_dropout)
        self.gat2 = GATConv(gnn_hidden, gnn_hidden // heads, heads=heads, dropout=gnn_dropout)
        self.drop_gnn = nn.Dropout(gnn_dropout)
        self.head = nn.Linear(gnn_hidden, out_dim)

    def forward(self, x_win, edge_index):
        lstm_out, _ = self.lstm(x_win)                  # [N, lb, H_LSTM]
        node_embed = self.drop_lstm(lstm_out[:, -1, :]) # [N, H_LSTM]

        if edge_index is None or edge_index.numel() == 0:
            raise RuntimeError("edge_index is empty; set USE_COARSENED_GRAPH=True or ensure edges exist.")

        h = F.elu(self.gat1(node_embed, edge_index))    # [N, H_GNN]
        h = self.drop_gnn(h)
        h = F.elu(self.gat2(h, edge_index))             # [N, H_GNN]
        h = self.drop_gnn(h)
        yhat = self.head(h)                             # [N, 2]
        return yhat

model = GraphLSTM_MTL(
    in_dim=len(FEATURES),
    lstm_hidden=H_LSTM, lstm_layers=LSTM_LAYERS, lstm_dropout=LSTM_DROPOUT,
    gnn_hidden=H_GNN, gnn_dropout=GNN_DROPOUT, heads=GAT_HEADS, out_dim=2
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6)

# -------------------------
# Mixed loss: Weighted NSE-like term for Q, MSE for Temp (masked)
#   NEW: Upweight Q residuals when Q_true > 100 m3/s (implemented in scaled space)
# -------------------------
def _q_highflow_weights_from_scaled(q_true_scaled: torch.Tensor,
                                   q_thresh_scaled: float,
                                   max_mult: float,
                                   power: float,
                                   eps: float = 1e-8) -> torch.Tensor:
    """
    Returns per-node weights w_i >= 1.
    - For q <= thresh: w=1
    - For q >  thresh: w ramps up to max_mult as q approaches 1.
    """
    if max_mult <= 1.0:
        return torch.ones_like(q_true_scaled)

    thr = torch.tensor(q_thresh_scaled, device=q_true_scaled.device, dtype=q_true_scaled.dtype)
    one = torch.tensor(1.0, device=q_true_scaled.device, dtype=q_true_scaled.dtype)

    denom = torch.clamp(one - thr, min=eps)
    frac = torch.clamp((q_true_scaled - thr) / denom, min=0.0, max=1.0)
    if power != 1.0:
        frac = torch.pow(frac, power)

    w = 1.0 + (max_mult - 1.0) * frac
    return w

def nse_mse_mixed_loss(yhat, ytrue, w_task, mask_t=None,
                       q_thresh_scaled=0.0, q_max_mult=1.0, q_power=1.0, eps=1e-8):
    """
    yhat,ytrue: [N, 2] scaled; columns [Q, T]
    mask_t: [N] boolean; True where in-situ Temp exists
    w_task: tensor([w_q, w_t]) for combining terms
    """
    w_task = w_task.to(yhat.device)

    q_pred = yhat[:, 0]
    q_true = ytrue[:, 0]

    # ---- NEW: high-flow weights (based on true Q) ----
    w_q = _q_highflow_weights_from_scaled(
        q_true_scaled=q_true,
        q_thresh_scaled=q_thresh_scaled,
        max_mult=q_max_mult,
        power=q_power,
        eps=eps
    )

    # Weighted NSE-like term: SSE_w / SS_w around weighted mean
    # (minimize this term)
    q_true_mean_w = torch.sum(w_q * q_true) / (torch.sum(w_q) + eps)
    num_q = torch.sum(w_q * (q_true - q_pred) ** 2)
    den_q = torch.sum(w_q * (q_true - q_true_mean_w) ** 2) + eps
    nse_term_q = num_q / den_q

    if mask_t is None or (mask_t.sum() == 0):
        mse_t = torch.tensor(0.0, device=yhat.device)
        w_use = torch.tensor([1.0, 0.0], device=yhat.device)
    else:
        t_pred = yhat[mask_t, 1]
        t_true = ytrue[mask_t, 1]
        mse_t = torch.mean((t_true - t_pred) ** 2)
        w_use = w_task

    w_use = w_use / (w_use.sum() + eps)
    total = w_use[0] * nse_term_q + w_use[1] * mse_t
    return total, nse_term_q.detach(), mse_t.detach()

# -------------------------
# Training Loop
# -------------------------
best_val = float("inf")
best_state = None
epochs_no_improve = 0

print(f"\nTraining Graph-LSTM+GAT (multitask Q+T) on {len(train_ds)} time steps; "
      f"validating on {len(test_ds)} time steps.")
for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_losses = []; tr_q = []; tr_t = []
    for x_win, y_t, m_t in train_loader:
        x_win = x_win.squeeze(0).to(device).float()  # [N, lb, F]
        y_t   = y_t.squeeze(0).to(device).float()    # [N, 2]
        m_t   = m_t.squeeze(0).to(device)            # [N] bool

        optimizer.zero_grad()
        yhat = model(x_win, edge_index)
        loss, lq, lt = nse_mse_mixed_loss(
            yhat, y_t, TASK_WEIGHTS, mask_t=m_t,
            q_thresh_scaled=q_thresh_scaled,
            q_max_mult=Q_HIGH_MAX_MULT,
            q_power=Q_HIGH_POWER
        )
        loss.backward()
        optimizer.step()

        tr_losses.append(loss.item()); tr_q.append(lq.item()); tr_t.append(lt.item())

    model.eval()
    va_losses = []; va_q = []; va_t = []
    with torch.no_grad():
        for x_win, y_t, m_t in test_loader:
            x_win = x_win.squeeze(0).to(device).float()
            y_t   = y_t.squeeze(0).to(device).float()
            m_t   = m_t.squeeze(0).to(device)
            yhat  = model(x_win, edge_index)
            loss, lq, lt = nse_mse_mixed_loss(
                yhat, y_t, TASK_WEIGHTS, mask_t=m_t,
                q_thresh_scaled=q_thresh_scaled,
                q_max_mult=Q_HIGH_MAX_MULT,
                q_power=Q_HIGH_POWER
            )
            va_losses.append(loss.item()); va_q.append(lq.item()); va_t.append(lt.item())

    tr_mean = float(np.mean(tr_losses)); va_mean = float(np.mean(va_losses))
    print(f"Epoch {epoch:03d}/{EPOCHS} | Train mix: {tr_mean:.6f} "
          f"(Q NSEterm_w {np.mean(tr_q):.6f}, T MSE {np.mean(tr_t):.6f}) "
          f"| Val mix: {va_mean:.6f} (Q NSEterm_w {np.mean(va_q):.6f}, T MSE {np.mean(va_t):.6f})")

    scheduler.step(va_mean)

    if va_mean + 1e-6 < best_val:
        best_val = va_mean
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

if best_state is not None:
    model.load_state_dict(best_state)
    model.to(device)

# -------------------------
# Evaluation (original units) on test period
# -------------------------
def nse_numpy(sim, obs):
    num = np.sum((sim - obs) ** 2)
    den = np.sum((obs - np.mean(obs)) ** 2) + 1e-12
    return 1.0 - num / den

def rmse_numpy(sim, obs):
    return float(np.sqrt(np.mean((sim - obs) ** 2)))

def mae_numpy(sim, obs):
    return float(np.mean(np.abs(sim - obs)))

model.eval()
preds_scaled = []
obs_scaled   = []
temp_masks   = []

with torch.no_grad():
    for x_win, y_t, m_t in test_loader:
        x_win = x_win.squeeze(0).to(device).float()
        y_t   = y_t.squeeze(0).to(device).float()
        m_t   = m_t.squeeze(0).cpu().numpy().astype(bool)
        yhat  = model(x_win, edge_index).detach().cpu().numpy()
        preds_scaled.append(yhat)
        obs_scaled.append(y_t.detach().cpu().numpy())
        temp_masks.append(m_t)

preds_scaled = np.stack(preds_scaled, axis=0)
obs_scaled   = np.stack(obs_scaled,   axis=0)
temp_masks   = np.stack(temp_masks,   axis=0)

def inverse_two_columns(arr_2):
    flat = arr_2.reshape(-1, 2)
    inv  = scaler_y.inverse_transform(flat)
    return inv.reshape(arr_2.shape)

pred_phys = inverse_two_columns(preds_scaled)
obs_phys  = inverse_two_columns(obs_scaled)

pred_q = pred_phys[:, :, 0].ravel()
true_q = obs_phys[:, :, 0].ravel()
pred_t = pred_phys[:, :, 1].ravel()
true_t = obs_phys[:, :, 1].ravel()
mask_t = temp_masks.ravel()

q_nse  = nse_numpy(pred_q, true_q)
q_rmse = rmse_numpy(pred_q, true_q)
q_mae  = mae_numpy(pred_q, true_q)

if mask_t.any():
    t_nse  = nse_numpy(pred_t[mask_t], true_t[mask_t])
    t_rmse = rmse_numpy(pred_t[mask_t], true_t[mask_t])
    t_mae  = mae_numpy(pred_t[mask_t], true_t[mask_t])
else:
    t_nse = t_rmse = t_mae = np.nan

print("\n=== Test Metrics (original units) ===")
print(f"Q    -> NSE: {q_nse:.4f} | RMSE: {q_rmse:.4f} | MAE: {q_mae:.4f}")
print(f"Temp -> NSE: {t_nse:.4f} | RMSE: {t_rmse:.4f} | MAE: {t_mae:.4f}")

# -------------------------
# Visualizations (single COMID)
# -------------------------
if TARGET_COMID_FOR_PLOTS not in comid_to_idx:
    raise ValueError(f"COMID {TARGET_COMID_FOR_PLOTS} not found in dataset.")

catch_idx = comid_to_idx[TARGET_COMID_FOR_PLOTS]

pred_q_series = pred_phys[:, catch_idx, 0]
true_q_series = obs_phys[:,  catch_idx, 0]
pred_t_series = pred_phys[:, catch_idx, 1]
true_t_series = obs_phys[:,  catch_idx, 1]

plt.figure(figsize=(12, 4))
plt.plot(true_q_series, label="Observed Q", linewidth=2)
plt.plot(pred_q_series, label="Predicted Q", alpha=0.8, linewidth=1.8)
plt.legend()
plt.title(f"Hydrograph (Test Period) — COMID {TARGET_COMID_FOR_PLOTS}")
plt.xlabel("Time steps")
plt.ylabel("Discharge (m³/s)")
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(true_t_series, label="Observed Temp", linewidth=2)
plt.plot(pred_t_series, label="Predicted Temp", alpha=0.8, linewidth=1.8)
plt.legend()
plt.title(f"Water Temperature (Test Period) — COMID {TARGET_COMID_FOR_PLOTS}")
plt.xlabel("Time steps")
plt.ylabel("Temperature (°C)")
plt.show()

valid_q = np.isfinite(true_q) & np.isfinite(pred_q)
plt.figure(figsize=(5, 5))
plt.scatter(true_q[valid_q], pred_q[valid_q], alpha=0.3, s=10)
max_q = max(true_q[valid_q].max(), pred_q[valid_q].max())
min_q = min(true_q[valid_q].min(), pred_q[valid_q].min())
plt.plot([min_q, max_q], [min_q, max_q], linestyle="--")
plt.xlabel("Observed Q (m³/s)")
plt.ylabel("Predicted Q (m³/s)")
plt.title("Discharge: Observed vs Predicted (All Test Nodes/Times)")
plt.tight_layout()
plt.show()

valid_t = mask_t & np.isfinite(true_t) & np.isfinite(pred_t)
plt.figure(figsize=(5, 5))
plt.scatter(true_t[valid_t], pred_t[valid_t], alpha=0.3, s=10)
max_t = max(true_t[valid_t].max(), pred_t[valid_t].max())
min_t = min(true_t[valid_t].min(), pred_t[valid_t].min())
plt.plot([min_t, max_t], [min_t, max_t], linestyle="--")
plt.xlabel("Observed Temp (°C)")
plt.ylabel("Predicted Temp (°C)")
plt.title("Water Temperature: Observed vs Predicted (All Test Nodes/Times)")
plt.tight_layout()
plt.show()

# -------------------------
# Raw observed histograms in physical units
# -------------------------
def plot_obs_histograms_logy(nodes_df_in, q_col="q_m3s_obs", t_col="water_temp_C",
                             bins_q=250, bins_t=200, title_suffix="(all observations)"):
    q = nodes_df_in[q_col].to_numpy()
    q = q[np.isfinite(q)]
    plt.figure(figsize=(9, 4))
    plt.hist(q, bins=bins_q)
    plt.yscale("log")
    plt.ylim(bottom=1)
    plt.title(f"Observed Discharge Histogram — log y {title_suffix}")
    plt.xlabel("Observed Q (m³/s)")
    plt.ylabel("Count (log scale)")
    plt.tight_layout()
    plt.show()

    t = nodes_df_in[t_col].to_numpy()
    t = t[np.isfinite(t)]
    plt.figure(figsize=(9, 4))
    plt.hist(t, bins=bins_t)
    plt.yscale("log")
    plt.ylim(bottom=1)
    plt.title(f"Observed Water Temperature Histogram — log y {title_suffix}")
    plt.xlabel("Observed Temp (°C)")
    plt.ylabel("Count (log scale)")
    plt.tight_layout()
    plt.show()

nodes_raw = pd.read_parquet(NODES_PATH)
nodes_raw["date"] = pd.to_datetime(nodes_raw["date"])
nodes_raw = replace_sentinel_with_nan(nodes_raw, [Q_COL, T_COL])
plot_obs_histograms_logy(nodes_raw, q_col=Q_COL, t_col=T_COL, bins_q=250, bins_t=200)

# -------------------------
# Save model + scalers + FEATURES (order-critical)
# -------------------------
os.makedirs(SAVE_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(SAVE_DIR, "graph_lstm_gat_multitask_q_t_mixedloss.pth"))
joblib.dump(scaler_X, os.path.join(SAVE_DIR, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(SAVE_DIR, "scaler_y.pkl"))
joblib.dump(FEATURES, FEATURES_PATH)

print(f"\nSaved model/scalers/features to:\n  {SAVE_DIR}\n  features: {FEATURES_PATH}")
