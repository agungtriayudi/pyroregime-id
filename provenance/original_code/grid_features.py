"""Agregasi titik FIRMS ke sel grid ~1 km per bulan.

    python grid_features.py --firms "data/raw/firms/VIIRS_SNPP_SP/**/*.csv" --out data/cube/cell_month.csv.gz

Fitur per (sel, bulan):
  n_det        jumlah deteksi (confidence nominal/high untuk VIIRS, >=30 untuk MODIS)
  active_days  jumlah hari dengan >=1 deteksi
  persist_max  hari berturut-turut terpanjang dengan deteksi  -> proksi api gambut membara
  night_frac   fraksi deteksi malam                          -> api gambut sering bertahan di malam hari
  frp_sum, frp_med, frp_max (MW)
Filter: confidence nominal/high; bila ada kolom `type` (arsip SP), hanya type 0 (api vegetasi).
Grid: lat/lon 0.01 derajat (~1,1 km di khatulistiwa). Untuk analisis luas yang ketat, proyeksikan ke
grid equal-area (mis. EPSG:6933) dan simpan cell_id-nya.
"""
import argparse
import glob

import numpy as np
import pandas as pd

RES = 0.01


def load_firms(pattern):
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise SystemExit(f"tidak ada file cocok: {pattern}")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df = df.drop_duplicates(subset=["latitude", "longitude", "acq_date", "acq_time"])
    conf = df["confidence"].astype(str).str.lower()
    if conf.str.fullmatch(r"\d+").all():                      # MODIS: 0-100
        keep = conf.astype(int) >= 30
    else:                                                      # VIIRS: l / n / h
        keep = conf.str[0].isin(["n", "h"])
    if "type" in df:                                           # SP: 0 = presumed vegetation fire
        keep &= df["type"].fillna(0).astype(int).eq(0)         # buang gunung api, gas flare, sumber statis
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
    ap.add_argument("--firms", required=True, help="glob CSV FIRMS")
    ap.add_argument("--out", default="data/cube/cell_month.csv.gz")
    args = ap.parse_args()
    cube = build(load_firms(args.firms))
    cube.to_csv(args.out, index=False)
    print(f"{len(cube):,} baris sel-bulan, {cube['cell'].nunique():,} sel -> {args.out}")


if __name__ == "__main__":
    main()
