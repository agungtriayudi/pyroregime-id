"""Fraksi gambut per sel 0,01 derajat dari peta gambut lokal, lalu gabung ke tabel static.

    python peat_fraction.py --config config.json
Masukan: cfg["peat_map"] = shapefile/GeoPackage/GeoJSON (poligon gambut) atau GeoTIFF (>0 = gambut).
Metode: rasterisasi pada 0,001 derajat per tile 1 derajat, lalu rerata blok 10 x 10 -> fraksi per sel.
Keluaran: cfg["static"] = cfg["static_gee"] + kolom peat_frac.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SUB = 10                     # 0,001 derajat per sub-piksel
FINE = 0.01 / SUB


def fraction_from_vector(path, bbox_tiles, cand):
    import geopandas as gpd
    from rasterio import features
    from rasterio.transform import from_origin
    gdf = gpd.read_file(path).to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    sidx = gdf.sindex
    out = []
    for (w, n) in bbox_tiles:                 # tile 1 x 1 derajat, pojok kiri-atas (w, n)
        cand_t = _cand_in_tile(cand, w, n)
        if not len(cand_t):
            continue
        hits = list(sidx.intersection((w, n - 1, w + 1, n)))
        size = int(round(1 / FINE))
        if hits:
            arr = features.rasterize(((g, 1) for g in gdf.geometry.iloc[hits]), out_shape=(size, size),
                                     transform=from_origin(w, n, FINE, FINE), fill=0, dtype="uint8", all_touched=False)
        else:
            arr = np.zeros((size, size), np.uint8)
        out.append(_block_mean(arr, w, n, cand_t))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["cell", "peat_frac"])


def fraction_from_raster(path, bbox_tiles, cand):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.transform import from_origin
    out = []
    size = int(round(1 / FINE))
    with rasterio.open(path) as src:
        for (w, n) in bbox_tiles:
            cand_t = _cand_in_tile(cand, w, n)
            if not len(cand_t):
                continue
            with WarpedVRT(src, crs="EPSG:4326", transform=from_origin(w, n, FINE, FINE), width=size, height=size,
                           resampling=Resampling.nearest) as vrt:
                arr = (vrt.read(1, masked=True).filled(0) > 0).astype(np.uint8)
            out.append(_block_mean(arr, w, n, cand_t))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["cell", "peat_frac"])


def _cand_in_tile(cand, w, n):
    iy, ix = cand // 100000, cand % 100000
    lat, lon = iy * 0.01 - 90 + 0.005, ix * 0.01 - 180 + 0.005
    return cand[(lon >= w) & (lon < w + 1) & (lat <= n) & (lat > n - 1)]


def _block_mean(arr, w, n, cand_t):
    frac = arr.reshape(arr.shape[0] // SUB, SUB, arr.shape[1] // SUB, SUB).mean(axis=(1, 3))   # 100 x 100 sel
    rows = np.arange(frac.shape[0]); cols = np.arange(frac.shape[1])
    lat_c = n - (rows + 0.5) * 0.01; lon_c = w + (cols + 0.5) * 0.01
    iy = np.floor((lat_c + 90) / 0.01 + 1e-9).astype(np.int64); ix = np.floor((lon_c + 180) / 0.01 + 1e-9).astype(np.int64)
    ids = iy[:, None] * 100000 + ix[None, :]
    df = pd.DataFrame({"cell": ids.ravel(), "peat_frac": frac.ravel()})
    return df[df["cell"].isin(cand_t)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--table-in", default=None, help="FOLLOWUP: tabel sel lain (kolom cell); bawaan cfg static_gee")
    ap.add_argument("--table-out", default=None, help="FOLLOWUP: keluaran; bawaan cfg static")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    followup = a.table_in is not None
    cand = (pd.read_csv(a.table_in, usecols=["cell"])["cell"].to_numpy() if followup
            else pd.read_csv(cfg["candidate_cells"])["cell"].to_numpy())
    tiles = sorted({(float(np.floor(lon)), float(np.ceil(lat)))
                    for lat, lon in zip((cand // 100000) * 0.01 - 90 + 0.005, (cand % 100000) * 0.01 - 180 + 0.005)})
    path = Path(cfg["peat_map"])
    if not path.exists():
        raise SystemExit(f"Peta gambut tidak ditemukan: {path}. Lihat CLAUDE.md bagian 'Peta gambut'.")
    pf = (fraction_from_raster if path.suffix.lower() in (".tif", ".tiff") else fraction_from_vector)(path, tiles, cand)
    st = pd.read_csv(a.table_in or cfg["static_gee"])
    st = st.drop(columns=[c for c in ["peat_frac"] if c in st]).merge(pf, on="cell", how="left")
    st["peat_frac"] = st["peat_frac"].fillna(0.0)
    # sel-tahun tanpa embedding sama sekali (laut lepas, di luar cakupan AlphaEarth/WorldCover) dibuang:
    # forecast.py mem-fit PCA pada semua baris static dan tidak menerima NaN; sel ini juga di luar topeng LSIB
    emb = [c for c in st if c.startswith("emb_")]
    if emb and not followup:                                     # tabel lanjutan: semua sel dipertahankan
        none = st[emb].isna().all(axis=1)
        if none.any():
            print(f"buang {int(none.sum()):,} baris static tanpa embedding ({st.loc[none, 'cell'].nunique():,} sel; "
                  f"in_indonesia maks {st.loc[none, 'in_indonesia'].max():.2f})")
            st = st[~none]
    dest = a.table_out or cfg["static"]
    st.to_csv(dest, index=False, float_format="%.5g")
    print(f"peat_frac untuk {len(pf):,} sel (rerata {pf['peat_frac'].mean():.3f}) -> {dest}")


if __name__ == "__main__":
    main()
