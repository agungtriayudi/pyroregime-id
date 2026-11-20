"""Aggregate FIRMS points to ~1 km grid cells per month.

    python grid_features.py --firms "data/raw/firms/VIIRS_SNPP_SP/**/*.csv" --out data/cube/cell_month.csv.gz

Features per (cell, month):
  n_det        number of detections (nominal/high confidence for VIIRS, >=30 for MODIS)
  active_days  number of days with >=1 detection
  persist_max  longest run of consecutive days with detections  -> proxy for smouldering peat fire
  night_frac   fraction of night-time detections                -> peat fires often persist at night
  frp_sum, frp_med, frp_max (MW)
Filter: nominal/high confidence; if a `type` column exists (SP archive), only type 0 (vegetation fire).
Grid: lat/lon 0.01 degree (~1.1 km at the equator). For strict area analyses, project to an equal-area
grid (e.g. EPSG:6933) and keep its cell_id.
"""
import argparse
import glob

import numpy as np
import pandas as pd

RES = 0.01


def load_firms(pattern):
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise SystemExit(f"no matching file: {pattern}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df = df.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
    conf = df["confidence"].astype(str).str.lower()
    if conf.str.fullmatch(r"\d+").all():                      # MODIS: 0-100
        keep = conf.astype(int) >= 30
    else:                                                      # VIIRS: l / n / h
        keep = conf.str[0].isin(["n", "h"])
    if "type" in df:                                           # SP: 0 = presumed vegetation fire
        keep &= df["type"].fillna(0).astype(int).eq(0)         # drop volcanoes, gas flares, static sources
    return df[keep].copy()


def cell_of(lat, lon, res=RES):
    iy = np.floor((lat + 90) / res).astype(np.int64)
    ix = np.floor((lon + 180) / res).astype(np.int64)
    return iy * 100000 + ix


def cell_center(cell_id, res=RES):
    iy, ix = cell_id // 100000, cell_id % 100000
    return iy * res - 90 + res / 2, ix * res - 180 + res / 2


def longest_run(days: np.ndarray) -> int:
    if len(days) == 0:
        return 0
    d = np.sort(np.unique(days))
    breaks = np.flatnonzero(np.diff(d) != 1)
    lens = np.diff(np.concatenate(([-1], breaks, [len(d) - 1])))
    return int(lens.max())


def build(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["acq_date"])
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["doy"] = (df["date"] - pd.Timestamp("2000-01-01")).dt.days
    df["cell"] = cell_of(df["latitude"].to_numpy(), df["longitude"].to_numpy())
    df["night"] = df["daynight"].astype(str).str.upper().eq("N")
    g = df.groupby(["cell", "month"])
    out = g.agg(n_det=("frp", "size"), frp_sum=("frp", "sum"), frp_med=("frp", "median"),
                frp_max=("frp", "max"), night_frac=("night", "mean"),
                active_days=("doy", "nunique")).reset_index()
    out["persist_max"] = g["doy"].apply(lambda s: longest_run(s.to_numpy())).to_numpy()
    lat, lon = cell_center(out["cell"].to_numpy())
    out["lat"], out["lon"] = lat, lon
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--firms", required=True, help="glob of FIRMS CSV files")
    ap.add_argument("--out", default="data/cube/cell_month.csv.gz")
    args = ap.parse_args()
    cube = build(load_firms(args.firms))
    cube.to_csv(args.out, index=False)
    print(f"{len(cube):,} cell-month rows, {cube['cell'].nunique():,} cells -> {args.out}")


if __name__ == "__main__":
    main()
