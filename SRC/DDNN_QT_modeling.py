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
import shap


# -------------------------
# Config
# -------------------------
NODES_PATH = "/Users/danieldominguez/Documents/Code/PGDL_ILOH/PGDL_ILOH/data/daily_drivers/hydro_temp_data.parquet"
EDGES_PATH = "/Users/danieldominguez/Documents/Code/PGDL_ILOH/PGDL_ILOH/data/daily_drivers/edge_list.parquet"

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

BASE_FEATURES = [
    "pr","srad","vs","vpd","sph","tmax_c","tmin_c","tmean_c","rmin","rmax","Ra", "bsn_pr"
]

Q_COL    = "q_m3s_obs"
T_COL    = "water_temp_C"
TARGETS  = [Q_COL, T_COL]
# weights for [Q, T] terms in the mixed loss
TASK_WEIGHTS = torch.tensor([0.7, 0.3], dtype=torch.float32)

TARGET_COMID_FOR_PLOTS = 14787569

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load node/edge data
# -------------------------
nodes_df = pd.read_parquet(NODES_PATH)
edges_df = pd.read_parquet(EDGES_PATH)

assert {"comid", "date", Q_COL}.issubset(nodes_df.columns), "q_m3s_obs missing"
assert T_COL in nodes_df.columns, "water_temp_C missing"

nodes_df["date"] = pd.to_datetime(nodes_df["date"])
nodes_df = nodes_df.sort_values(["date", "comid"]).reset_index(drop=True)

# Preserve in-situ availability for Temp BEFORE any filling/scaling
nodes_df["temp_avail_raw"] = ~nodes_df[T_COL].isna()

# Seasonality terms (sine/cosine of day-of-year)
if "DOY_sin" not in nodes_df.columns or "DOY_cos" not in nodes_df.columns:
    doy = nodes_df["date"].dt.dayofyear.astype(float)
    nodes_df["DOY_sin"] = np.sin(2*np.pi*doy/366.0)
    nodes_df["DOY_cos"] = np.cos(2*np.pi*doy/366.0)

# Automatic static feature detection (constant per COMID)
static_candidates = []
skip_cols = set(["comid", "date"] + TARGETS + BASE_FEATURES + ["DOY_sin","DOY_cos","temp_avail_raw"])
for col in nodes_df.columns:
    if col in skip_cols:
        continue
    if pd.api.types.is_numeric_dtype(nodes_df[col]) and nodes_df.groupby("comid")[col].nunique().max() == 1:
        static_candidates.append(col)

force_include = ["areasqkm", "slope", "lengthkm", "streamorde", "lat", "lon"]
static_candidates = list(set(static_candidates + [c for c in force_include if c in nodes_df.columns]))
# drop coords if present
static_candidates = [f for f in static_candidates if f not in ["lat","lon"]]

# Final feature set (no lagged Q to avoid leakage)
FEATURES = [f for f in (BASE_FEATURES + ["DOY_sin","DOY_cos"] + static_candidates)
            if f in nodes_df.columns and pd.api.types.is_numeric_dtype(nodes_df[f])]

keep_cols = ["comid", "date"] + FEATURES + TARGETS + ["temp_avail_raw"]
nodes_df = nodes_df[keep_cols]

# -------------------------
# COMID indexing
# -------------------------
all_comids = np.sort(nodes_df["comid"].unique())
comid_to_idx = {c: i for i, c in enumerate(all_comids)}
n_nodes = len(all_comids)

# -------------------------
# Build edge list (upstream -> downstream) and map to indices
# -------------------------
# If your edges file has fromnode/tonode/COMID triplets, build adjacency via node joins
edges_df = edges_df.dropna(subset=["fromnode", "tonode", "comid"])
up = edges_df[["comid", "fromnode", "tonode"]].rename(columns={"comid": "comid_up"})
dn = edges_df[["comid", "fromnode", "tonode"]].rename(columns={"comid": "comid_dn"})
edge_pairs = up.merge(dn, left_on="tonode", right_on="fromnode", how="inner")[["comid_up", "comid_dn"]].drop_duplicates()

edge_pairs = edge_pairs[
    edge_pairs["comid_up"].isin(all_comids) & edge_pairs["comid_dn"].isin(all_comids)
].drop_duplicates()

src = edge_pairs["comid_up"].map(comid_to_idx).to_numpy()
dst = edge_pairs["comid_dn"].map(comid_to_idx).to_numpy()
edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long).to(device)

print(f"Graph: {n_nodes} nodes, {edge_index.shape[1]} directed edges")

# -------------------------
# Scale features/targets with train-only fit
# -------------------------
train_mask_rows = nodes_df["date"] < SPLIT_DATE

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

fit_X = np.nan_to_num(nodes_df.loc[train_mask_rows, FEATURES].values, nan=0.0)
fit_y = np.nan_to_num(nodes_df.loc[train_mask_rows, TARGETS].values, nan=0.0)

scaler_X.fit(fit_X)
scaler_y.fit(fit_y)

# Transform all rows using train-fitted scalers
X_all = np.nan_to_num(nodes_df[FEATURES].values, nan=0.0)
Y_all = np.nan_to_num(nodes_df[TARGETS].values, nan=0.0)  # keep zeros where missing
nodes_df[FEATURES] = scaler_X.transform(X_all)
nodes_df[TARGETS]  = scaler_y.transform(Y_all)

# Keep the raw availability mask for T (after indexing)
TEMP_AVAIL_ALL = nodes_df["temp_avail_raw"].to_numpy().astype(bool)

# -------------------------
# Dense tensors
# X: [T, N, F], Y: [T, N, 2], TempMask: [T, N]
# -------------------------
dates = np.sort(nodes_df["date"].unique())
Tsteps = len(dates); Fdim = len(FEATURES)

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
# Dataset + Dataloader (time-based split uses target date at t1)
# -------------------------
class GraphSeqDataset(Dataset):
    def __init__(self, X_TNF, Y_TN2, Tmask_TN, lookback, dates):
        self.X = X_TNF
        self.Y = Y_TN2
        self.M = Tmask_TN
        self.lb = lookback
        self.dates = dates

    def __len__(self):
        # targets will be at indices t = lb .. T-1  (inclusive)
        return self.X.shape[0] - self.lb

    def __getitem__(self, idx):
        # target time index
        t = idx + self.lb              # t in [lb .. T-1]

        # window: [t - lb + 1, ..., t]  (length = lb)
        t0 = t - self.lb + 1
        t1 = t + 1                      # slice is [t0:t1)

        x_win = self.X[t0:t1, :, :]     # [lb, N, F]
        x_win = np.transpose(x_win, (1, 0, 2)).astype(np.float32)  # [N, lb, F]

        y_t   = self.Y[t, :, :].astype(np.float32)     # [N, 2]
        m_t   = self.M[t, :].astype(bool)              # [N]

        return torch.from_numpy(x_win), torch.from_numpy(y_t), torch.from_numpy(m_t)


dataset = GraphSeqDataset(X_TNF, Y_TN2, Tmask_TN, LOOKBACK, dates)

t1_dates = dates[LOOKBACK:]
train_indices = [i for i, d in enumerate(t1_dates) if d < SPLIT_DATE]
test_indices  = [i for i, d in enumerate(t1_dates) if d >= SPLIT_DATE]

assert len(train_indices) > 0, "No training samples before split date."
assert len(test_indices)  > 0, "No testing samples on/after split date."

train_ds = Subset(dataset, train_indices)
test_ds  = Subset(dataset,  test_indices)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Training on {len(train_ds)} time steps (< {SPLIT_DATE.date()}); "
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
        # Two GAT layers
        self.gat1 = GATConv(lstm_hidden, gnn_hidden // heads, heads=heads, dropout=gnn_dropout)
        self.gat2 = GATConv(gnn_hidden, gnn_hidden // heads, heads=heads, dropout=gnn_dropout)
        self.drop_gnn = nn.Dropout(gnn_dropout)
        self.head = nn.Linear(gnn_hidden, out_dim)  # outputs [Q, T] (scaled)

    def forward(self, x_win, edge_index):
        # x_win: [N, lb, F]
        lstm_out, _ = self.lstm(x_win)                  # [N, lb, H_LSTM]
        node_embed = self.drop_lstm(lstm_out[:, -1, :]) # [N, H_LSTM]
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
# Mixed loss: NSE for Q, MSE for Temp (masked)
# -------------------------
def nse_mse_mixed_loss(yhat, ytrue, w, mask_t=None, eps=1e-8):
    """
    yhat,ytrue: [N, 2] scaled; columns [Q, T]
    mask_t: [N] boolean; True where in-situ Temp exists
    w: tensor([w_q, w_t]) for combining terms
    Returns: (total_loss, nse_term_q, mse_term_t)
    """
    w = w.to(yhat.device)

    # NSE term for Q (minimize 1 - NSE == num/den)
    q_pred = yhat[:, 0]
    q_true = ytrue[:, 0]
    num_q = torch.sum((q_true - q_pred) ** 2)
    den_q = torch.sum((q_true - torch.mean(q_true)) ** 2) + eps
    nse_term_q = num_q / den_q

    # MSE for T (masked)
    if mask_t is None or (mask_t.sum() == 0):
        mse_t = torch.tensor(0.0, device=yhat.device)
        w_use = torch.tensor([1.0, 0.0], device=yhat.device)
    else:
        t_pred = yhat[mask_t, 1]
        t_true = ytrue[mask_t, 1]
        mse_t = torch.mean((t_true - t_pred) ** 2)
        w_use = w

    # Normalize active weights
    w_use = w_use / (w_use.sum() + eps)
    total = w_use[0]*nse_term_q + w_use[1]*mse_t
    return total, nse_term_q.detach(), mse_t.detach()

# -------------------------
# Training Loop
# -------------------------
best_val = float("inf")
best_state = None
epochs_no_improve = 0

print(f"Training Graph-LSTM+GAT (multitask Q+T) on {len(train_ds)} time steps; "
      f"validating on {len(test_ds)} time steps.")
for epoch in range(1, EPOCHS + 1):
    model.train()
    tr_losses = []; tr_q = []; tr_t = []
    for x_win, y_t, m_t in train_loader:
        x_win = x_win.squeeze(0).to(device).float()  # [N, lb, F]
        y_t   = y_t.squeeze(0).to(device).float()    # [N, 2]
        m_t   = m_t.squeeze(0).to(device)            # [N] bool

        optimizer.zero_grad()
        yhat = model(x_win, edge_index)              # [N, 2]
        loss, lq, lt = nse_mse_mixed_loss(yhat, y_t, TASK_WEIGHTS, mask_t=m_t)
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
            loss, lq, lt = nse_mse_mixed_loss(yhat, y_t, TASK_WEIGHTS, mask_t=m_t)
            va_losses.append(loss.item()); va_q.append(lq.item()); va_t.append(lt.item())

    tr_mean = float(np.mean(tr_losses)); va_mean = float(np.mean(va_losses))
    print(f"Epoch {epoch:03d}/{EPOCHS} | Train mix: {tr_mean:.6f} "
          f"(Q NSEterm {np.mean(tr_q):.6f}, T MSE {np.mean(tr_t):.6f}) "
          f"| Val mix: {va_mean:.6f} (Q NSEterm {np.mean(va_q):.6f}, T MSE {np.mean(va_t):.6f})")

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
    num = np.sum((sim - obs)**2)
    den = np.sum((obs - np.mean(obs))**2) + 1e-12
    return 1.0 - num/den

def rmse_numpy(sim, obs):
    return float(np.sqrt(np.mean((sim - obs)**2)))

def mae_numpy(sim, obs):
    return float(np.mean(np.abs(sim - obs)))

model.eval()
preds_scaled = []
obs_scaled   = []
temp_masks   = []

with torch.no_grad():
    for x_win, y_t, m_t in test_loader:
        x_win = x_win.squeeze(0).to(device).float()
        y_t   = y_t.squeeze(0).to(device).float()               # [N, 2] scaled
        m_t   = m_t.squeeze(0).cpu().numpy().astype(bool)       # [N]
        yhat  = model(x_win, edge_index).detach().cpu().numpy() # [N,2]

        preds_scaled.append(yhat)
        obs_scaled.append(y_t.detach().cpu().numpy())
        temp_masks.append(m_t)

# stack to [T_test, N, 2]
preds_scaled = np.stack(preds_scaled, axis=0)
obs_scaled   = np.stack(obs_scaled,   axis=0)
temp_masks   = np.stack(temp_masks,   axis=0)  # [T_test, N]

# inverse transform back to physical units using scaler_y on columns [Q, T]
def inverse_two_columns(arr_2):
    flat = arr_2.reshape(-1, 2)
    inv  = scaler_y.inverse_transform(flat)
    return inv.reshape(arr_2.shape)

pred_phys = inverse_two_columns(preds_scaled)
obs_phys  = inverse_two_columns(obs_scaled)

# Flatten over time and nodes
pred_q = pred_phys[:, :, 0].ravel(); true_q = obs_phys[:, :, 0].ravel()
pred_t = pred_phys[:, :, 1].ravel(); true_t = obs_phys[:, :, 1].ravel()
mask_t = temp_masks.ravel()

# Overall metrics
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
# Visualizations (single COMID = 14787569)
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

# Discharge (Q) — all nodes, all test timesteps
valid_q = ~np.isnan(true_q) & ~np.isnan(pred_q)

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

# Temperature (T) — only where in-situ Temp is available
valid_t = mask_t & ~np.isnan(true_t) & ~np.isnan(pred_t)

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
# Save model + scalers
# -------------------------
save_dir = "/Users/danieldominguez/Documents/Code/PGDL_ILOH/PGDL_ILOH/models"
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_dir, "graph_lstm_gat_multitask_q_t_mixedloss.pth"))
joblib.dump(scaler_X, os.path.join(save_dir, "scaler_X.pkl"))
joblib.dump(scaler_y, os.path.join(save_dir, "scaler_y.pkl"))

