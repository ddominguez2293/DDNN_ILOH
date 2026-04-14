#!/usr/bin/env python3
from pathlib import Path
import polars as pl

# ----------------------------
# Paths (EDIT THESE)
# ----------------------------
drivers_feather = Path("/Users/danieldominguez/Documents/Code/DDNN_ILOH/data/daily_drivers/drivers_daily_joined_PET.feather")
q_parquet       = Path("/Users/danieldominguez/Documents/Code/DDNN_ILOH/data/daily_drivers/q_obs.parquet")

# NHDPlus inputs (statics live here)
flowlines_gpkg  = Path("/Users/danieldominguez/Documents/Code/DDNN_ILOH/data/nhdplus/flowlines.gpkg")
catchments_gpkg = Path("/Users/danieldominguez/Documents/Code/DDNN_ILOH/data/nhdplus/catchments.gpkg")

out_nodes_parquet_dir = Path("/Users/danieldominguez/Documents/Code/DDNN_ILOH/data/daily_drivers/hydro_nodes_allcomids_parquet")
out_edges_parquet     = Path("/Users/danieldominguez/Documents/Code/DDNN_ILOH/data/daily_drivers/edge_list.parquet")

# ----------------------------
# 0) Build STATICS table (one row per COMID)
#   - area_m2     from catchments.areasqkm * 1e6
#   - Slope       from flowlines.slope
#   - LengthKM    from flowlines.lengthkm
#   - streamorde  from flowlines.streamorde
# ----------------------------
import geopandas as gpd

# Catchments statics
catch = gpd.read_file(catchments_gpkg, engine="pyogrio")[["comid", "areasqkm"]].copy()
catch["comid"] = catch["comid"].astype("int64")
catch_pl = (
    pl.from_pandas(catch)
    .with_columns([
        pl.col("comid").cast(pl.Int32),
        (pl.col("areasqkm").cast(pl.Float64, strict=False) * 1e6).alias("area_m2"),
    ])
    .select(["comid", "area_m2"])
)

# Flowlines statics + edge nodes
flow = gpd.read_file(flowlines_gpkg, engine="pyogrio")[["comid", "fromnode", "tonode", "slope", "lengthkm", "streamorde"]].copy()
flow["comid"] = flow["comid"].astype("int64")

flow_pl = (
    pl.from_pandas(flow)
    .with_columns([
        pl.col("comid").cast(pl.Int32),
        pl.col("fromnode").cast(pl.Int64, strict=False),
        pl.col("tonode").cast(pl.Int64, strict=False),
        pl.col("slope").cast(pl.Float64, strict=False).alias("Slope"),
        pl.col("lengthkm").cast(pl.Float64, strict=False).alias("LengthKM"),
        pl.col("streamorde").cast(pl.Int32, strict=False),
    ])
)

# Statics: one row per comid
statics = (
    flow_pl.select(["comid", "Slope", "LengthKM", "streamorde"])
    .unique(subset=["comid"], keep="first")
    .join(catch_pl, on="comid", how="left")
)

# ----------------------------
# 1) Build NODES: (drivers LEFT JOIN q_obs) LEFT JOIN statics
# ----------------------------
drivers = (
    pl.scan_ipc(str(drivers_feather))
    .with_columns([
        pl.col("comid").cast(pl.Int32),
        pl.col("date").cast(pl.Date),
    ])
)

qobs = (
    pl.scan_parquet(str(q_parquet))
    .select([
        pl.col("comid").cast(pl.Int32).alias("comid"),
        pl.col("Date").cast(pl.Date).alias("date"),
        pl.col("q_m3s").cast(pl.Float64).alias("q_m3s_obs"),
    ])
)

nodes = (
    drivers
    .join(qobs, on=["comid", "date"], how="left")
    .join(statics.lazy(), on="comid", how="left")  # <-- adds area_m2, Slope, LengthKM, streamorde
)

stats = nodes.select([
    pl.len().alias("n_rows"),
    pl.col("comid").n_unique().alias("n_comids"),
    pl.col("date").min().alias("min_date"),
    pl.col("date").max().alias("max_date"),
    pl.col("q_m3s_obs").is_not_null().mean().alias("pct_rows_with_q"),
    pl.col("comid").filter(pl.col("q_m3s_obs").is_not_null()).n_unique().alias("n_comids_with_any_q"),
    pl.col("area_m2").is_not_null().mean().alias("pct_rows_with_area_m2"),
    pl.col("Slope").is_not_null().mean().alias("pct_rows_with_Slope"),
    pl.col("LengthKM").is_not_null().mean().alias("pct_rows_with_LengthKM"),
    pl.col("streamorde").is_not_null().mean().alias("pct_rows_with_streamorde"),
]).collect()

print("\nNODES summary (with statics):\n", stats)

# Write nodes
out_nodes_parquet_dir.mkdir(parents=True, exist_ok=True)
nodes.sink_parquet(
    str(out_nodes_parquet_dir / "nodes.parquet"),
    compression="zstd"
)
print(f"\nWROTE nodes parquet to: {out_nodes_parquet_dir}")

# ----------------------------
# 2) Build EDGES: ALL flowlines edges (as before)
# ----------------------------
edge_df = (
    flow_pl
    .select([
        pl.col("comid").cast(pl.Int64),
        pl.col("fromnode").cast(pl.Int64),
        pl.col("tonode").cast(pl.Int64),
    ])
    .drop_nulls(["fromnode", "tonode"])
    .unique()
)

edge_df.write_parquet(str(out_edges_parquet), compression="zstd")
print(f"WROTE edges parquet to: {out_edges_parquet}")
print("Edge rows:", edge_df.height, " unique COMIDs:", edge_df.select(pl.col("comid").n_unique()).item())
