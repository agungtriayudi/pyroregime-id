"""Peat fraction per 0.01-degree cell from a local peat map, then merged into the static table.

    python peat_fraction.py --config config.json
Input: cfg["peat_map"] = shapefile/GeoPackage/GeoJSON (peat polygons) or GeoTIFF (>0 = peat).
Method: rasterisation at 0.001 degree per 1-degree tile, then 10 x 10 block mean -> fraction per cell.
Output: cfg["static"] = cfg["static_gee"] + column peat_frac.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SUB = 10                     # 0.001 degree per sub-pixel
FINE = 0.01 / SUB


def fraction_from_vector(path, bbox_tiles, cand):
    import geopandas as gpd
    from rasterio import features
    from rasterio.transform import from_origin
    gdf = gpd.read_file(path).to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    sidx = gdf.sindex
    out = []
    for (w, n) in bbox_tiles:                 # 1 x 1 degree tile, upper-left corner (w, n)
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
    frac = arr.reshape(arr.shape[0] // SUB, SUB, arr.shape[1] // SUB, SUB).mean(axis=(1, 3))   # 100 x 100 cells
    rows = np.arange(frac.shape[0]); cols = np.arange(frac.shape[1])
    lat_c = n - (rows + 0.5) * 0.01; lon_c = w + (cols + 0.5) * 0.01
    iy = np.floor((lat_c + 90) / 0.01 + 1e-9).astype(np.int64); ix = np.floor((lon_c + 180) / 0.01 + 1e-9).astype(np.int64)
    ids = iy[:, None] * 100000 + ix[None, :]
    df = pd.DataFrame({"cell": ids.ravel(), "peat_frac": frac.ravel()})
    return df[df["cell"].isin(cand_t)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--table-in", default=None, help="FOLLOWUP: other cell table (column cell); default cfg static_gee")
    ap.add_argument("--table-out", default=None, help="FOLLOWUP: output; default cfg static")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    followup = a.table_in is not None
    cand = (pd.read_csv(a.table_in, usecols=["cell"])["cell"].to_numpy() if followup
            else pd.read_csv(cfg["candidate_cells"])["cell"].to_numpy())
    tiles = sorted({(float(np.floor(lon)), float(np.ceil(lat)))
                    for lat, lon in zip((cand // 100000) * 0.01 - 90 + 0.005, (cand % 100000) * 0.01 - 180 + 0.005)})
    path = Path(cfg["peat_map"])
    if not path.exists():
        raise SystemExit(f"Peat map not found: {path}. See docs/REPRODUCE.md, section 'Peat map'.")
    pf = (fraction_from_raster if path.suffix.lower() in (".tif", ".tiff") else fraction_from_vector)(path, tiles, cand)
    st = pd.read_csv(a.table_in or cfg["static_gee"])
    st = st.drop(columns=[c for c in ["peat_frac"] if c in st]).merge(pf, on="cell", how="left")
    st["peat_frac"] = st["peat_frac"].fillna(0.0)
    # cell-years without any embedding (open sea, outside AlphaEarth/WorldCover coverage) are dropped:
    # forecast.py fits the PCA on all static rows and does not accept NaN; these cells are also outside the LSIB mask
    emb = [c for c in st if c.startswith("emb_")]
    if emb and not followup:                                     # follow-up table: all cells are kept
        none = st[emb].isna().all(axis=1)
        if none.any():
            print(f"dropping {int(none.sum()):,} static rows without embeddings ({st.loc[none, 'cell'].nunique():,} cells; "
                  f"in_indonesia max {st.loc[none, 'in_indonesia'].max():.2f})")
            st = st[~none]
    dest = a.table_out or cfg["static"]
    st.to_csv(dest, index=False, float_format="%.5g")
    print(f"peat_frac for {len(pf):,} cells (mean {pf['peat_frac'].mean():.3f}) -> {dest}")


if __name__ == "__main__":
    main()
