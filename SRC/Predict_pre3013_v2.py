#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Predict_pre2013_optionA_inference.py

Option A NON-GRAPH inference:
  - LSTM + Temporal Attention (NO graph)
  - scaler_X.pkl
  - scaler_y_by_task.pkl  (dict: {"q": MinMaxScaler, "t": MinMaxScaler})
  - features.pkl          (ordered list used during training)

Outputs:
- OUT_PATH: includes both
    * q_pred_m3s_raw
    * q_pred_m3s_constrained
    * temp_pred_C
- OUT_PATH_CONSTRAIN: legacy constrained-only file

Troubleshooting:
- raw driver sanity print
- feature range vs scaler training
- outlier row dump for extreme scaled X
- temp hotspot dump for extreme temp_scaled

NEW in this version (requested):
- Write debug outputs to CSV (not parquet)
- For temp hotspot rows, also export BOTH:
    (a) X_scaled values for all FEATURES at that (date, comid)
    (b) X_raw (unscaled) values for all FEATURES at that (date, comid)

CRITICAL FIXES (statics):
- Repair "streamorde", "LengthKM", "area_m2", "Slope" robustly:
    streamorde: treat <=0 OR == -9 as missing
    LengthKM:   treat <=0 OR very negative as missing
    area_m2:    treat <=0 OR very negative as missing
    Slope:      treat <0  OR very negative as missing
- Imputation priority:
    1) TRAIN_STATIC_MEDIANS_P (if available)
    2) analysis median of valid values
    3) scaler midpoint (train min/max midpoint)
    4) 0.0 fallback

NOTE:
- "keep in mind the high area_m2 for now" -> we do NOT cap large area_m2 values.
  We only repair missing/invalid (<=0 or sentinel-like) values.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================
NODES_PATH_ANALYSIS = "/Users/danieldominguez/Documents/Code/DDNN_ILOH/data/daily_drivers/hydro_temp_data_analysis_pre2013.parquet"

MODEL_DIR   = "/Users/danieldominguez/Documents/Code/DDNN_ILOH/data/models_fullbasin"
MODEL_PATH  = os.path.join(MODEL_DIR, "lstm_attn_multitask_q_t_mixedloss_fullbasin_optionA.pth")

SCALER_X_P  = os.path.join(MODEL_DIR, "scaler_X.pkl")
SCALER_Y_P  = os.path.join(MODEL_DIR, "scaler_y_by_task.pkl")
FEATURES_P  = os.path.join(MODEL_DIR, "features.pkl")

# Optional: if you can save this during training, it’s the best way to impute statics consistently
# e.g., {"Slope": 0.0123, "LengthKM": 3.4, "area_m2": 1.2e6, "streamorde": 2}
TRAIN_STATIC_MEDIANS_P = os.path.join(MODEL_DIR, "train_static_medians.pkl")

OUT_PATH           = NODES_PATH_ANALYSIS.replace(".parquet", "_predictions_optionA.parquet")
OUT_PATH_CONSTRAIN = NODES_PATH_ANALYSIS.replace(".parquet", "_predictions_optionA_constrained_only.parquet")

# Debug outputs (CSV)
DEBUG_FEATURE_OUTLIERS_PATH          = NODES_PATH_ANALYSIS.replace(".parquet", "_troubleshoot_feature_outliers.csv")
DEBUG_TEMP_HOTSPOTS_PATH             = NODES_PATH_ANALYSIS.replace(".parquet", "_troubleshoot_temp_hotspots.csv")
DEBUG_TEMP_HOTSPOTS_WITH_X_PATH      = NODES_PATH_ANALYSIS.replace(".parquet", "_troubleshoot_temp_hotspots_with_X.csv")
DEBUG_TEMP_BADROWS_PATH              = NODES_PATH_ANALYSIS.replace(".parquet", "_troubleshoot_bad_temp_rows.csv")
DEBUG_TEMP_BADROWS_WITH_X_PATH       = NODES_PATH_ANALYSIS.replace(".parquet", "_troubleshoot_bad_temp_rows_with_X.csv")

LOOKBACK = 7

H_LSTM       = 64
LSTM_LAYERS  = 1
LSTM_DROPOUT = 0.2

TARGET_COMID_FOR_PLOTS = 14787569
N_PLOT_DAYS = 500

# =========================
# TROUBLESHOOTING TOGGLES
# =========================
RUN_TROUBLESHOOTING = True
X_SCALED_EXTREME_THRESHOLD = 10.0
X_SCALED_MIN_THRESHOLD     = -5.0

TEMP_SCALED_HOT_THRESHOLD  = 1.25
N_HOTSPOT_ROWS_TO_SAVE     = 20000

# Optional mitigation (keep None unless you explicitly want to clip scaled temp)
TEMP_SCALED_CLIP = None  # e.g. (0.0, 1.0)

# =========================
# SENTINEL REPAIR (MATCH TRAINING)
# =========================
SENTINEL_MISS = -9998.0

# =========================
# TEMP QC (file audit)
# =========================
TEMP_COL_AUDIT = "water_temp_C"   # present in analysis parquet
TEMP_QC_MIN_C  = -5.0
TEMP_QC_MAX_C  = 60.0
APPLY_TEMP_QC_TO_FILE_DF = True  # audit-only; safe to leave True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# MODEL
# =============================================================================
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, h_seq: torch.Tensor) -> torch.Tensor:
        logits = self.score(h_seq).squeeze(-1)      # [N, L]
        alpha  = torch.softmax(logits, dim=1)       # [N, L]
        ctx    = torch.sum(h_seq * alpha.unsqueeze(-1), dim=1)  # [N, H]
        return ctx


class LSTM_Attn_MTL(nn.Module):
    def __init__(self, in_dim, lstm_hidden, lstm_layers, lstm_dropout, out_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=(lstm_dropout if lstm_layers > 1 else 0.0),
            batch_first=True
        )
        self.drop = nn.Dropout(lstm_dropout)
        self.attn = TemporalAttention(lstm_hidden)
        self.head = nn.Linear(lstm_hidden, out_dim)

    def forward(self, x_win: torch.Tensor) -> torch.Tensor:
        h_seq, _ = self.lstm(x_win)   # [N, L, H]
        h_seq = self.drop(h_seq)
        ctx = self.attn(h_seq)        # [N, H]
        return self.head(ctx)         # [N, 2] scaled


# =============================================================================
# HELPERS
# =============================================================================
def add_doy_terms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "DOY_sin" not in df.columns or "DOY_cos" not in df.columns:
        doy = df["date"].dt.dayofyear.astype(float)
        df["DOY_sin"] = np.sin(2 * np.pi * doy / 366.0)
        df["DOY_cos"] = np.cos(2 * np.pi * doy / 366.0)
    return df


def inverse_two_columns_by_task(arr_scaled_TN2: np.ndarray, scaler_y_q, scaler_y_t) -> np.ndarray:
    out = np.zeros_like(arr_scaled_TN2, dtype=np.float32)
    q_flat = arr_scaled_TN2[:, :, 0].reshape(-1, 1)
    t_flat = arr_scaled_TN2[:, :, 1].reshape(-1, 1)
    out[:, :, 0] = scaler_y_q.inverse_transform(q_flat).reshape(arr_scaled_TN2.shape[0], arr_scaled_TN2.shape[1]).astype(np.float32)
    out[:, :, 1] = scaler_y_t.inverse_transform(t_flat).reshape(arr_scaled_TN2.shape[0], arr_scaled_TN2.shape[1]).astype(np.float32)
    return out


def print_raw_driver_sanity(df_raw: pd.DataFrame):
    candidates = ["tmax_c", "tmin_c", "tmean_c", "pr", "vpd", "sph", "srad"]
    present = [c for c in candidates if c in df_raw.columns]
    if not present:
        print("\n[Sanity] None of the standard driver columns present for raw sanity check.")
        return

    print("\n[Sanity] Raw driver ranges (global):")
    for c in present:
        x = pd.to_numeric(df_raw[c], errors="coerce")
        if x.notna().any():
            print(f"  {c:>10s}  min/med/max = {float(x.min()):.4g} / {float(x.median()):.4g} / {float(x.max()):.4g}")
        else:
            print(f"  {c:>10s}  all NA after coercion")


def feature_range_diagnostics(X_all_raw: np.ndarray, FEATURES: list, scaler_X):
    if not (hasattr(scaler_X, "data_min_") and hasattr(scaler_X, "data_max_")):
        print("\n[FeatureRange] scaler_X missing data_min_/data_max_. Skipping range diagnostics.")
        return None

    mins = scaler_X.data_min_.astype(np.float64)
    maxs = scaler_X.data_max_.astype(np.float64)
    raw_min = np.nanmin(X_all_raw.astype(np.float64), axis=0)
    raw_max = np.nanmax(X_all_raw.astype(np.float64), axis=0)

    below = raw_min < mins
    above = raw_max > maxs

    rows = []
    for j, f in enumerate(FEATURES):
        rows.append({
            "feature": f,
            "train_min": mins[j],
            "train_max": maxs[j],
            "analysis_min": raw_min[j],
            "analysis_max": raw_max[j],
            "is_below_train_min": bool(below[j]),
            "is_above_train_max": bool(above[j]),
        })

    diag = pd.DataFrame(rows)
    n_flag = int((diag["is_below_train_min"] | diag["is_above_train_max"]).sum())
    print("\n[FeatureRange] Features outside training min/max:", n_flag, "of", len(FEATURES))
    if n_flag > 0:
        print(diag.loc[diag["is_below_train_min"] | diag["is_above_train_max"]].head(30).to_string(index=False))
    return diag


def save_feature_outlier_rows_csv(df_meta: pd.DataFrame, X_scaled: np.ndarray, FEATURES: list, out_path: str,
                                  min_thr: float = -5.0, abs_thr: float = 10.0, max_rows: int = 200000):
    bad_low = (X_scaled < min_thr)
    bad_abs = (np.abs(X_scaled) > abs_thr)
    bad_any = bad_low | bad_abs
    row_mask = bad_any.any(axis=1)

    n_bad = int(row_mask.sum())
    print(f"\n[XScaledOutliers] Rows with any X_scaled < {min_thr} or |X_scaled| > {abs_thr}: {n_bad:,}")
    if n_bad == 0:
        return

    feat_counts = bad_any.sum(axis=0)
    top_idx = np.argsort(-feat_counts)[:20]
    print("[XScaledOutliers] Top features by outlier count:")
    for j in top_idx:
        if feat_counts[j] > 0:
            print(f"  {FEATURES[j]:>20s} : {int(feat_counts[j])}")

    idx = np.where(row_mask)[0]
    if len(idx) > max_rows:
        idx = idx[:max_rows]

    export = df_meta.iloc[idx].copy()
    top10 = [FEATURES[j] for j in top_idx[:10]]
    feat_to_idx = {f: i for i, f in enumerate(FEATURES)}
    for f in top10:
        export[f"X_scaled__{f}"] = X_scaled[idx, feat_to_idx[f]]

    export.to_csv(out_path, index=False)
    print(f"[XScaledOutliers] Wrote debug outliers CSV: {out_path}  (rows={len(export):,})")


def replace_sentinel_with_nan_all_features(df: pd.DataFrame, FEATURES: list, sentinel: float = -9998.0) -> pd.DataFrame:
    """Match training: sentinel -> NaN for all FEATURES before scaling; later NaN->0 for X."""
    df = df.copy()
    n_total = 0
    for c in FEATURES:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        mask = (x == sentinel)
        ct = int(mask.sum())
        if ct > 0:
            n_total += ct
            df.loc[mask, c] = np.nan
    print(f"\n[SentinelAllFeatures] Replaced sentinel ({sentinel}) with NaN across FEATURES. total_replacements={n_total:,}")
    return df


def sign_summary(df, col):
    x = df[col].to_numpy()
    x = x[np.isfinite(x)]
    n = len(x)
    n_neg = int((x < 0).sum())
    n_zero = int((x == 0).sum())
    n_pos = int((x > 0).sum())
    return {
        "n": n,
        "n_neg": n_neg, "pct_neg": 100.0 * n_neg / max(n, 1),
        "n_zero": n_zero, "pct_zero": 100.0 * n_zero / max(n, 1),
        "n_pos": n_pos, "pct_pos": 100.0 * n_pos / max(n, 1),
        "min": float(np.nanmin(x)) if n else np.nan,
        "med": float(np.nanmedian(x)) if n else np.nan,
        "max": float(np.nanmax(x)) if n else np.nan,
    }


def audit_and_qc_analysis_temp(df: pd.DataFrame,
                              out_badrows_csv: str,
                              out_badrows_withX_csv: str,
                              FEATURES: list = None) -> pd.DataFrame:
    """
    Audit-only: inspect 'water_temp_C' in the analysis parquet to find corrupted rows.
    Writes a CSV of the bad rows; optionally also writes the bad rows with unscaled FEATURES.

    IMPORTANT: the inference model does NOT use analysis temp as an input feature.
    This is purely to trace upstream data issues.
    """
    if TEMP_COL_AUDIT not in df.columns:
        print(f"\n[TempAudit] Column '{TEMP_COL_AUDIT}' not present; skipping temp audit.")
        return df

    t_raw = pd.to_numeric(df[TEMP_COL_AUDIT], errors="coerce")
    t_valid = t_raw.dropna()

    print("\n[Analysis temp raw] n=", int(t_valid.shape[0]))
    if t_valid.shape[0] > 0:
        print("[Analysis temp raw] min/med/max =",
              float(t_valid.min()), float(t_valid.median()), float(t_valid.max()))
        print("[Analysis temp raw] pct > 45 (valid only) =",
              float((t_valid > 45.0).mean()))
        print("[Analysis temp raw] pct > 60 (valid only) =",
              float((t_valid > 60.0).mean()))
    else:
        print("[Analysis temp raw] all NaN after coercion")

    bad_mask = t_raw.notna() & ((t_raw < TEMP_QC_MIN_C) | (t_raw > TEMP_QC_MAX_C))
    n_bad = int(bad_mask.sum())
    print(f"\n[Temp QC] bad temp rows outside [{TEMP_QC_MIN_C}, {TEMP_QC_MAX_C}]C: {n_bad:,}")

    if n_bad > 0:
        cols = ["date", "comid", TEMP_COL_AUDIT]
        bad = df.loc[bad_mask, cols].copy()
        bad = bad.sort_values(TEMP_COL_AUDIT, ascending=False)
        print(bad.head(30).to_string(index=False))

        # write the minimal bad rows
        bad.to_csv(out_badrows_csv, index=False)
        print(f"[Temp QC] Wrote bad temp rows CSV: {out_badrows_csv}")

        # write bad rows with unscaled features (if requested and features available)
        if FEATURES is not None and all(f in df.columns for f in FEATURES):
            bad_with = df.loc[bad_mask, ["date", "comid", TEMP_COL_AUDIT] + FEATURES].copy()
            bad_with = bad_with.sort_values(TEMP_COL_AUDIT, ascending=False)
            bad_with.to_csv(out_badrows_withX_csv, index=False)
            print(f"[Temp QC] Wrote bad temp rows + FEATURES CSV: {out_badrows_withX_csv}")

    if APPLY_TEMP_QC_TO_FILE_DF and n_bad > 0:
        df = df.copy()
        df.loc[bad_mask, TEMP_COL_AUDIT] = np.nan
        print("[Temp QC] Set implausible analysis temps to NaN (audit-only).")

    return df


def grid_rectangular_check_or_fail(df: pd.DataFrame):
    n_dates = int(df["date"].nunique())
    n_comids = int(df["comid"].nunique())
    n_rows = int(len(df))
    expected = n_dates * n_comids
    print(f"\n[GridCheck] n_dates={n_dates:,} n_comids={n_comids:,} n_rows={n_rows:,} expected={expected:,}")
    if n_rows != expected:
        raise RuntimeError(
            "Analysis parquet is NOT a full rectangular (date, comid) grid after aggregation. "
            "Missing comid-days will remain zeros in X_TNF and can cause OOD predictions. "
            "Fix upstream or explicitly impute missing (date, comid) rows before inference."
        )


def _scaler_midpoint(scaler_X, FEATURES, col):
    """midpoint of training min/max for a given feature column (raw space)."""
    if not (hasattr(scaler_X, "data_min_") and hasattr(scaler_X, "data_max_")):
        return None
    feat_to_idx = {f: i for i, f in enumerate(FEATURES)}
    if col not in feat_to_idx:
        return None
    j = feat_to_idx[col]
    return float((float(scaler_X.data_min_[j]) + float(scaler_X.data_max_[j])) / 2.0)


def repair_statics_robust(df: pd.DataFrame, scaler_X, FEATURES: list,
                         train_static_medians_path: str = None) -> pd.DataFrame:
    """
    Robust repair for statics: streamorde, LengthKM, area_m2, Slope.

    Rules (tunable):
      - streamorde: missing if (<=0) OR (== -9) OR (<= -1000)
      - LengthKM:   missing if (<=0) OR (<= -1000)
      - area_m2:    missing if (<=0) OR (<= -1000)
      - Slope:      missing if (<0)  OR (<= -1000)

    Keeps high area_m2 values (no capping).
    """
    df = df.copy()

    # load training medians dict if present
    train_medians = None
    if train_static_medians_path and os.path.exists(train_static_medians_path):
        try:
            train_medians = joblib.load(train_static_medians_path)
            if not isinstance(train_medians, dict):
                train_medians = None
        except Exception:
            train_medians = None

    # compute analysis medians for valid values (used as fallback)
    def analysis_median_valid(col, valid_mask):
        x = pd.to_numeric(df[col], errors="coerce")
        x = x[valid_mask & x.notna()]
        if x.shape[0] == 0:
            return None
        return float(x.median())

    # helper for each static
    def repair_col(col, is_missing_mask, valid_mask_for_median):
        if col not in df.columns or col not in FEATURES:
            return

        x = pd.to_numeric(df[col], errors="coerce")
        n_bad = int(is_missing_mask.sum())
        if n_bad == 0:
            return

        print(f"\n[StaticFix] {col}: found {n_bad:,} invalid/missing rows. Setting to NaN then imputing.")
        df.loc[is_missing_mask, col] = np.nan

        # choose impute value
        impute_val = None

        # (1) training median
        if train_medians is not None and col in train_medians and np.isfinite(train_medians[col]):
            impute_val = float(train_medians[col])
            print(f"[StaticFix] {col}: imputing with TRAIN median = {impute_val:.6g} ({train_static_medians_path})")
        else:
            # (2) analysis median among valid values
            med = analysis_median_valid(col, valid_mask_for_median)
            if med is not None and np.isfinite(med):
                impute_val = float(med)
                print(f"[StaticFix] {col}: imputing with ANALYSIS median(valid) = {impute_val:.6g}")
            else:
                # (3) scaler midpoint
                mid = _scaler_midpoint(scaler_X, FEATURES, col)
                if mid is not None and np.isfinite(mid):
                    impute_val = float(mid)
                    print(f"[StaticFix] {col}: imputing with SCALER midpoint = {impute_val:.6g}")
                else:
                    # (4) 0 fallback
                    impute_val = 0.0
                    print(f"[StaticFix] {col}: imputing with 0.0 (fallback)")

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(impute_val)

    # streamorde
    if "streamorde" in df.columns and "streamorde" in FEATURES:
        x = pd.to_numeric(df["streamorde"], errors="coerce")
        miss = x.isna() | (x <= 0) | (x == -9) | (x <= -1000)
        valid = x.notna() & (x > 0) & (x != -9) & (x > -1000)
        repair_col("streamorde", miss, valid)

    # LengthKM
    if "LengthKM" in df.columns and "LengthKM" in FEATURES:
        x = pd.to_numeric(df["LengthKM"], errors="coerce")
        miss = x.isna() | (x <= 0) | (x <= -1000)
        valid = x.notna() & (x > 0) & (x > -1000)
        repair_col("LengthKM", miss, valid)

    # area_m2 (do NOT cap highs)
    if "area_m2" in df.columns and "area_m2" in FEATURES:
        x = pd.to_numeric(df["area_m2"], errors="coerce")
        miss = x.isna() | (x <= 0) | (x <= -1000)
        valid = x.notna() & (x > 0) & (x > -1000)
        repair_col("area_m2", miss, valid)

    # Slope
    if "Slope" in df.columns and "Slope" in FEATURES:
        x = pd.to_numeric(df["Slope"], errors="coerce")
        miss = x.isna() | (x < 0) | (x <= -1000)
        valid = x.notna() & (x >= 0) & (x > -1000)
        repair_col("Slope", miss, valid)

    return df


def write_temp_hotspots_csv(pred_dates, all_comids, temp_scaled_TN: np.ndarray,
                            out_path: str, hot_thr: float = 1.25, max_rows: int = 20000):
    mask = temp_scaled_TN > hot_thr
    n_hot = int(mask.sum())
    print(f"\n[TempHotspots] Count(temp_scaled > {hot_thr}) = {n_hot:,}")
    if n_hot == 0:
        # still write an empty file for traceability
        pd.DataFrame(columns=["date", "comid", "temp_scaled"]).to_csv(out_path, index=False)
        print(f"[TempHotspots] Wrote EMPTY hotspot CSV: {out_path}")
        return

    t_idx, n_idx = np.where(mask)
    hot_vals = temp_scaled_TN[t_idx, n_idx]
    order = np.argsort(-hot_vals)
    order = order[:min(len(order), max_rows)]

    out = pd.DataFrame({
        "date": np.array(pred_dates, dtype="datetime64[ns]")[t_idx[order]],
        "comid": all_comids[n_idx[order]],
        "temp_scaled": hot_vals[order],
    })
    out.to_csv(out_path, index=False)
    print(f"[TempHotspots] Wrote hotspot CSV: {out_path}  (rows={len(out):,})")


def write_temp_hotspots_with_X_csv(pred_dates, all_comids, preds_scaled_TN2: np.ndarray,
                                  X_TNF_scaled: np.ndarray,
                                  df_raw_sorted: pd.DataFrame,
                                  FEATURES: list,
                                  out_path: str,
                                  hot_thr: float = 1.25,
                                  max_rows: int = 20000):
    """
    Export hotspot rows with:
      - date, comid, temp_scaled
      - top10_ood_features_scaled (based on scaled X vs 0.5 midpoint)
      - ALL features in scaled form:  X_scaled__{f}
      - ALL features in raw form:     X_raw__{f}  (from df_raw_sorted)
    """
    t_sc = preds_scaled_TN2[:, :, 1]
    hot_mask = t_sc > hot_thr
    n_hot = int(hot_mask.sum())
    print(f"\n[TempHotspots+X] Count(temp_scaled > {hot_thr}) = {n_hot:,}")

    # indices are in prediction-time coordinates [0..T_pred-1]
    if n_hot > 0:
        hot_ti, hot_ni = np.where(hot_mask)
        hot_vals = t_sc[hot_ti, hot_ni]
        order = np.argsort(-hot_vals)[:min(len(hot_vals), max_rows)]
        sel_ti = hot_ti[order]
        sel_ni = hot_ni[order]
        sel_vals = hot_vals[order]
    else:
        # fallback: top max_rows anyway
        flat = np.argsort(t_sc.reshape(-1))[::-1]
        take = min(len(flat), max_rows)
        flat = flat[:take]
        sel_ti = (flat // t_sc.shape[1]).astype(int)
        sel_ni = (flat %  t_sc.shape[1]).astype(int)
        sel_vals = t_sc[sel_ti, sel_ni].astype(float)

    # We need raw X at the same (date, comid). Since grid is rectangular and df_raw_sorted is sorted by date,comid:
    # raw X can be retrieved by absolute time index: abs_t = LOOKBACK + ti
    # then node index ni, then feature j
    # We'll pre-build raw tensor too (cheap because it's just a reshape)
    Tsteps = len(np.sort(df_raw_sorted["date"].unique()))
    n_nodes = len(np.sort(df_raw_sorted["comid"].unique()))
    Fdim = len(FEATURES)

    X_raw_flat = df_raw_sorted.loc[:, FEATURES].to_numpy(dtype=np.float32)
    X_TNF_raw = X_raw_flat.reshape(Tsteps, n_nodes, Fdim)

    rows = []
    for ti, ni, v in zip(sel_ti, sel_ni, sel_vals):
        ti = int(ti); ni = int(ni)
        d = pd.to_datetime(pred_dates[ti])
        com = int(all_comids[ni])

        abs_t = LOOKBACK + ti  # align pred index to absolute day index
        x_scaled_row = X_TNF_scaled[abs_t, ni, :]
        x_raw_row    = X_TNF_raw[abs_t, ni, :]

        ood = np.abs(x_scaled_row - 0.5)
        top_feat_idx = np.argsort(-ood)[:10]
        top_feats = [(FEATURES[j], float(x_scaled_row[j])) for j in top_feat_idx]

        rec = {
            "date": d,
            "comid": com,
            "temp_scaled": float(v),
            "top10_ood_features_scaled": "; ".join([f"{name}={val:.4f}" for name, val in top_feats]),
        }

        for j, f in enumerate(FEATURES):
            rec[f"X_scaled__{f}"] = float(x_scaled_row[j])
        for j, f in enumerate(FEATURES):
            rec[f"X_raw__{f}"] = float(x_raw_row[j]) if np.isfinite(x_raw_row[j]) else np.nan

        rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"[TempHotspots+X] Wrote hotspot+X CSV: {out_path}  (rows={len(out):,})")


# =============================================================================
# MAIN
# =============================================================================
def main():
    required = [NODES_PATH_ANALYSIS, MODEL_PATH, SCALER_X_P, SCALER_Y_P, FEATURES_P]
    for p in required:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    df = pd.read_parquet(NODES_PATH_ANALYSIS)
    print("Loaded analysis df:", df.shape)

    if not {"comid", "date"}.issubset(df.columns):
        raise ValueError("Analysis parquet must contain columns: ['comid','date'].")

    df["date"] = pd.to_datetime(df["date"])
    df = add_doy_terms(df)

    # ---- Load FEATURES (order-critical) from training artifact
    FEATURES = joblib.load(FEATURES_P)
    FEATURES = list(dict.fromkeys(FEATURES))
    print(f"Loaded FEATURES (n={len(FEATURES)}): first 20 -> {FEATURES[:20]}")

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError("Missing required FEATURES in analysis parquet:\n" + "\n".join(missing[:200]))

    # Temp audit (file QA only) -> writes CSVs
    if RUN_TROUBLESHOOTING:
        df = audit_and_qc_analysis_temp(
            df,
            out_badrows_csv=DEBUG_TEMP_BADROWS_PATH,
            out_badrows_withX_csv=DEBUG_TEMP_BADROWS_WITH_X_PATH,
            FEATURES=FEATURES
        )

    has_dup_labels = bool(df.columns.duplicated().any())
    print("Duplicate column labels?", has_dup_labels)
    if has_dup_labels:
        dup_labels = df.columns[df.columns.duplicated()].tolist()
        print("WARNING: DataFrame has duplicate column labels. First 30:", dup_labels[:30])
        df = df.loc[:, ~df.columns.duplicated()].copy()
        print("Dropped duplicate-labeled columns. New shape:", df.shape)

    # ---- De-duplicate rows at (date, comid) level (mean numeric)
    dup_ct = int(df.duplicated(subset=["date", "comid"]).sum())
    if dup_ct > 0:
        print(f"WARNING: Found {dup_ct:,} duplicate (date, comid) rows. Aggregating numeric columns by mean.")
        num_cols = [c for c in df.columns if c not in ("date", "comid") and pd.api.types.is_numeric_dtype(df[c])]
        df = df.groupby(["date", "comid"], as_index=False)[num_cols].mean()
        df = add_doy_terms(df)
        df = df.sort_values(["date", "comid"]).reset_index(drop=True)

    # Keep only what we need (+ meta)
    keep_cols = ["comid", "date"] + FEATURES
    df = df.loc[:, keep_cols].copy()

    # ---- Load scalers
    scaler_X = joblib.load(SCALER_X_P)
    y_scalers = joblib.load(SCALER_Y_P)
    if not isinstance(y_scalers, dict) or ("q" not in y_scalers) or ("t" not in y_scalers):
        raise ValueError(f"{SCALER_Y_P} must be a dict with keys 'q' and 't'.")
    scaler_y_q = y_scalers["q"]
    scaler_y_t = y_scalers["t"]

    # ---- Strong checks: feature count and optional name match
    if hasattr(scaler_X, "n_features_in_"):
        n_expected = int(scaler_X.n_features_in_)
        if n_expected != len(FEATURES):
            raise ValueError(f"Feature count mismatch: scaler expects {n_expected}, but FEATURES has {len(FEATURES)}.")

    if hasattr(scaler_X, "feature_names_in_"):
        if list(scaler_X.feature_names_in_) != FEATURES:
            raise RuntimeError(
                "FEATURE order mismatch vs scaler_X.feature_names_in_. "
                "You are about to scale columns in the wrong order."
            )

    # Print y scaler ranges (helps catch accidental wrong file)
    try:
        print("\n[ScalerY] q min/max:", scaler_y_q.data_min_, scaler_y_q.data_max_)
        print("[ScalerY] t min/max:", scaler_y_t.data_min_, scaler_y_t.data_max_)
    except Exception:
        pass

    # ---- Rectangular grid check (hard fail if missing comid-days)
    grid_rectangular_check_or_fail(df)

    # --- TROUBLESHOOTING (raw sanity + feature range) BEFORE repairs
    if RUN_TROUBLESHOOTING:
        print_raw_driver_sanity(df)
        X_raw_for_diag = np.nan_to_num(df.loc[:, FEATURES].to_numpy(dtype=np.float32), nan=0.0).astype(np.float32)
        feature_range_diagnostics(X_raw_for_diag, FEATURES, scaler_X)

    # ---- Match training sentinel handling across ALL features
    df = replace_sentinel_with_nan_all_features(df, FEATURES, sentinel=SENTINEL_MISS)

    # ---- Robust static repairs (streamorde / LengthKM / area_m2 / Slope)
    df = repair_statics_robust(
        df,
        scaler_X=scaler_X,
        FEATURES=FEATURES,
        train_static_medians_path=TRAIN_STATIC_MEDIANS_P
    )

    # --- TROUBLESHOOTING: re-run range diagnostics AFTER repairs (this is what feeds scaler)
    if RUN_TROUBLESHOOTING:
        X_raw_after_repairs = np.nan_to_num(df.loc[:, FEATURES].to_numpy(dtype=np.float32), nan=0.0).astype(np.float32)
        feature_range_diagnostics(X_raw_after_repairs, FEATURES, scaler_X)

    # Save a raw-sorted copy for hotspot export (UNSCALED but post-repair, post-sentinel->NaN)
    # (This is what you want to inspect; it reflects fixed statics.)
    df_raw_for_export = df.sort_values(["date", "comid"]).reset_index(drop=True).copy()

    # -------------------------------------------------------------------------
    # Scale X (consistent with training: NaN -> 0.0, then MinMax transform)
    # -------------------------------------------------------------------------
    X_all = np.nan_to_num(df.loc[:, FEATURES].to_numpy(dtype=np.float32), nan=0.0).astype(np.float32)
    X_scaled = scaler_X.transform(X_all).astype(np.float32)
    df.loc[:, FEATURES] = X_scaled

    print("\nScaled X min/mean/max:", float(X_scaled.min()), float(X_scaled.mean()), float(X_scaled.max()))
    print("Pct <=0:", float((X_scaled <= 0).mean()), "Pct >=1:", float((X_scaled >= 1).mean()))
    print("Pct < -0.5:", float((X_scaled < -0.5).mean()), "Pct > 1.5:", float((X_scaled > 1.5).mean()))

    if RUN_TROUBLESHOOTING:
        df_meta_for_debug = df.loc[:, ["date", "comid"]].copy()
        save_feature_outlier_rows_csv(
            df_meta=df_meta_for_debug,
            X_scaled=X_scaled,
            FEATURES=FEATURES,
            out_path=DEBUG_FEATURE_OUTLIERS_PATH,
            min_thr=X_SCALED_MIN_THRESHOLD,
            abs_thr=X_SCALED_EXTREME_THRESHOLD,
            max_rows=200000
        )

    # -------------------------------------------------------------------------
    # Build dense X tensor: [T, N, F]
    # -------------------------------------------------------------------------
    all_comids = np.sort(df["comid"].unique())
    comid_to_idx = {int(c): i for i, c in enumerate(all_comids)}
    n_nodes = len(all_comids)

    dates = np.sort(df["date"].unique())
    Tsteps = len(dates)
    Fdim = len(FEATURES)
    print(f"\nAnalysis grid -> T={Tsteps:,} days | N={n_nodes:,} COMIDs | F={Fdim}")
    if Tsteps <= LOOKBACK:
        raise ValueError(f"Not enough timesteps ({Tsteps}) for LOOKBACK={LOOKBACK}.")

    # Because the grid is rectangular, we can reshape safely after sorting
    df_sorted = df.sort_values(["date", "comid"]).reset_index(drop=True)
    X_flat = df_sorted.loc[:, FEATURES].to_numpy(dtype=np.float32)
    X_TNF = X_flat.reshape(Tsteps, n_nodes, Fdim)  # SCALED

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    model = LSTM_Attn_MTL(
        in_dim=Fdim, lstm_hidden=H_LSTM, lstm_layers=LSTM_LAYERS, lstm_dropout=LSTM_DROPOUT, out_dim=2
    ).to(device)

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------
    preds_scaled_list = []
    pred_dates = []

    with torch.no_grad():
        for t in range(LOOKBACK, Tsteps):
            t0 = t - LOOKBACK + 1
            t1 = t + 1
            x_win = X_TNF[t0:t1, :, :]               # [lb, N, F]
            x_win = np.transpose(x_win, (1, 0, 2))   # [N, lb, F]
            x_win_t = torch.tensor(x_win, dtype=torch.float32, device=device)
            yhat = model(x_win_t)  # [N,2]
            preds_scaled_list.append(yhat.detach().cpu().numpy().astype(np.float32))
            pred_dates.append(dates[t])

    preds_scaled = np.stack(preds_scaled_list, axis=0)  # [T_pred, N, 2]

    # -------------------------------------------------------------------------
    # Output diagnostics: temp hotspots (CSV + CSV with X_scaled/X_raw)
    # -------------------------------------------------------------------------
    q_sc = preds_scaled[:, :, 0]
    t_sc = preds_scaled[:, :, 1]
    print("\nScaled output stats:")
    print("  Q_scaled    min/mean/max:", float(q_sc.min()), float(q_sc.mean()), float(q_sc.max()))
    print("  Temp_scaled min/mean/max:", float(t_sc.min()), float(t_sc.mean()), float(t_sc.max()))
    print(f"  Temp_scaled pct > {TEMP_SCALED_HOT_THRESHOLD}:", float((t_sc > TEMP_SCALED_HOT_THRESHOLD).mean()))

    if RUN_TROUBLESHOOTING:
        # 1) lean hotspot list
        write_temp_hotspots_csv(
            pred_dates=pred_dates,
            all_comids=all_comids,
            temp_scaled_TN=t_sc,
            out_path=DEBUG_TEMP_HOTSPOTS_PATH,
            hot_thr=TEMP_SCALED_HOT_THRESHOLD,
            max_rows=N_HOTSPOT_ROWS_TO_SAVE
        )

        # 2) hotspot list WITH X_scaled + X_raw
        write_temp_hotspots_with_X_csv(
            pred_dates=pred_dates,
            all_comids=all_comids,
            preds_scaled_TN2=preds_scaled,
            X_TNF_scaled=X_TNF,
            df_raw_sorted=df_raw_for_export.sort_values(["date", "comid"]).reset_index(drop=True),
            FEATURES=FEATURES,
            out_path=DEBUG_TEMP_HOTSPOTS_WITH_X_PATH,
            hot_thr=TEMP_SCALED_HOT_THRESHOLD,
            max_rows=N_HOTSPOT_ROWS_TO_SAVE
        )

    preds_scaled_for_inv = preds_scaled.copy()
    if TEMP_SCALED_CLIP is not None:
        lo, hi = TEMP_SCALED_CLIP
        preds_scaled_for_inv[:, :, 1] = np.clip(preds_scaled_for_inv[:, :, 1], lo, hi)
        print(f"\nApplied TEMP_SCALED_CLIP to scaled temp: [{lo}, {hi}] before inverse-transform.")

    pred_phys = inverse_two_columns_by_task(preds_scaled_for_inv, scaler_y_q, scaler_y_t)

    # -------------------------------------------------------------------------
    # Build output dataframe (no write yet)
    # -------------------------------------------------------------------------
    out_rows = []
    for ti, d in enumerate(pred_dates):
        q_raw = pred_phys[ti, :, 0]
        q_con = np.clip(q_raw, a_min=0.0, a_max=None)
        out_rows.append(pd.DataFrame({
            "date": d,
            "comid": all_comids,
            "q_pred_m3s_raw": q_raw,
            "q_pred_m3s_constrained": q_con,
            "temp_pred_C": pred_phys[ti, :, 1],
        }))

    pred_df = pd.concat(out_rows, axis=0, ignore_index=True)
    print("\nPreview predictions:")
    print(pred_df.head())

    # histograms
    plt.figure(figsize=(9, 4))
    q = pred_df["q_pred_m3s_raw"].to_numpy()
    q = q[np.isfinite(q)]
    plt.hist(q, bins=250)
    plt.yscale("log")
    plt.ylim(bottom=1)
    plt.title("Raw Histogram: Predicted Discharge (q_pred_m3s_raw) — log y-scale")
    plt.xlabel("Predicted Q (m³/s)")
    plt.ylabel("Count (log scale)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(9, 4))
    tt = pred_df["temp_pred_C"].to_numpy()
    tt = tt[np.isfinite(tt)]
    plt.hist(tt, bins=200)
    plt.yscale("log")
    plt.ylim(bottom=1)
    plt.title("Raw Histogram: Predicted Water Temperature (temp_pred_C) — log y-scale")
    plt.xlabel("Predicted Temp (°C)")
    plt.ylabel("Count (log scale)")
    plt.tight_layout()
    plt.show()

    print("\n=== Q sign summary (RAW) ===")
    print(sign_summary(pred_df, "q_pred_m3s_raw"))
    print("\n=== Q sign summary (CONSTRAINED) ===")
    print(sign_summary(pred_df, "q_pred_m3s_constrained"))

    if TARGET_COMID_FOR_PLOTS in set(all_comids.tolist()):
        s = pred_df[pred_df["comid"] == TARGET_COMID_FOR_PLOTS].sort_values("date")

        plt.figure(figsize=(12, 4))
        plt.plot(s["date"].iloc[:N_PLOT_DAYS], s["q_pred_m3s_raw"].iloc[:N_PLOT_DAYS], label="Pred Q (raw)", linewidth=1.8)
        plt.plot(s["date"].iloc[:N_PLOT_DAYS], s["q_pred_m3s_constrained"].iloc[:N_PLOT_DAYS], label="Pred Q (constrained)", linewidth=1.2, alpha=0.9)
        plt.title(f"Predicted Q — COMID {TARGET_COMID_FOR_PLOTS} (first {N_PLOT_DAYS} pred days)")
        plt.xlabel("Date")
        plt.ylabel("Q (m³/s)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 4))
        plt.plot(s["date"].iloc[:N_PLOT_DAYS], s["temp_pred_C"].iloc[:N_PLOT_DAYS], label="Pred Temp", linewidth=1.8)
        plt.title(f"Predicted Temp — COMID {TARGET_COMID_FOR_PLOTS} (first {N_PLOT_DAYS} pred days)")
        plt.xlabel("Date")
        plt.ylabel("Temp (°C)")
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # WRITE FILES (END)
    # -------------------------------------------------------------------------
    pred_df.to_parquet(OUT_PATH, index=False)
    print(f"\nWrote predictions (raw+constrained columns): {OUT_PATH}")

    pred_df_constrained_only = pred_df.loc[:, ["date", "comid", "q_pred_m3s_constrained", "temp_pred_C"]].copy()
    pred_df_constrained_only = pred_df_constrained_only.rename(columns={"q_pred_m3s_constrained": "q_pred_m3s"})
    pred_df_constrained_only.to_parquet(OUT_PATH_CONSTRAIN, index=False)
    print(f"Wrote predictions (constrained-only legacy): {OUT_PATH_CONSTRAIN}")

    print("\nWrote debug CSVs:")
    print("  Feature outliers:", DEBUG_FEATURE_OUTLIERS_PATH)
    print("  Temp hotspots (lean):", DEBUG_TEMP_HOTSPOTS_PATH)
    print("  Temp hotspots (+X):", DEBUG_TEMP_HOTSPOTS_WITH_X_PATH)
    print("  Bad analysis temps (audit):", DEBUG_TEMP_BADROWS_PATH)
    print("  Bad analysis temps (+X):", DEBUG_TEMP_BADROWS_WITH_X_PATH)


if __name__ == "__main__":
    main()
