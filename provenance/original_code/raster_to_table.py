"""Ubah GeoTIFF hasil gee_export.py menjadi tabel per sel (grid identik dengan grid_features.py).

    pip install rasterio
    python raster_to_table.py --tifs "drive/pyroregime/a_*_2025*.tif" --cells data/cube/fingerprint_2012_2025.csv \
        --out data/cube/static_2025.csv          # -> dipakai sebagai --static di regime_features.py
    python raster_to_table.py --tifs "drive/pyroregime/m_*.tif" --out data/cube/gee_monthly.csv.gz

Nama file m_<region>_<YYYYMM>.tif -> kolom month; a_<region>_<YYYY>.tif -> kolom year.
BELUM dijalankan di lingkungan pengembang (butuh GeoTIFF nyata).
"""
import argparse
import glob
import re

import numpy as np
import pandas as pd
import rasterio

from grid_features import cell_of


def read_tif(path, keep=None):
    with rasterio.open(path) as src:
        arr = src.read().astype("float32")
        names = [d or f"b{i}" for i, d in enumerate(src.descriptions)]
        t = src.transform
    rows, cols = np.indices(arr.shape[1:])
    lon = t.c + (cols + 0.5) * t.a
    lat = t.f + (rows + 0.5) * t.e
    cell = cell_of(lat.ravel(), lon.ravel())
    df = pd.DataFrame(arr.reshape(arr.shape[0], -1).T, columns=names)
    df.insert(0, "cell", cell)
    df = df.dropna(how="all", subset=names)
    return df[df["cell"].isin(keep)] if keep is not None else df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tifs", required=True)
    ap.add_argument("--cells", default=None, help="CSV dengan kolom cell untuk membatasi ukuran tabel")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    keep = set(pd.read_csv(a.cells)["cell"]) if a.cells else None
    parts = []
    for f in sorted(glob.glob(a.tifs)):
        df = read_tif(f, keep)
        m = re.search(r"_(\d{6})\.tif$", f)
        y = re.search(r"_(\d{4})\.tif$", f)
        if m:
            df.insert(1, "month", f"{m.group(1)[:4]}-{m.group(1)[4:]}")
        elif y:
            df.insert(1, "year", int(y.group(1)))
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    key = [c for c in ("cell", "month", "year") if c in out]
    out = out.groupby(key, as_index=False).mean()      # sel di tepi dua tile/region
    out.to_csv(a.out, index=False)
    print(f"{len(out):,} baris -> {a.out}")


if __name__ == "__main__":
    main()
