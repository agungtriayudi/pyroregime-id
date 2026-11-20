"""Ringkas tabel sel-bulan menjadi "sidik jari rezim api" per sel.

    python regime_features.py --cube data/cube/cell_month.csv.gz --oni data/raw/oni.ascii.txt \
        --years 2012-2025 --min-det 20 --out data/cube/fingerprint_2012_2025.csv
    # jendela El Nino untuk uji pergeseran rezim (E3):
    python regime_features.py ... --years 2018-2020 --out data/cube/fp_2019.csv

Opsional --static CSV (kolom cell + mis. peat_frac, emb_A00..emb_A63 hasil gee_export.py) digabung per sel.
ONI: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt (kolom SEAS YR TOTAL ANOM).
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def load_oni(path):
    """Kembalikan ONI musim ASO (Agu-Sep-Okt) per tahun: puncak musim api Indonesia."""
    df = pd.read_csv(path, sep=r"\s+")
    df.columns = [c.upper() for c in df.columns]
    aso = df[df["SEAS"] == "ASO"].set_index("YR")["ANOM"]
    return aso.astype(float)


def fingerprint(cube: pd.DataFrame, oni: pd.Series | None, y0: int, y1: int, min_det: int,
                months=(1, 12)) -> pd.DataFrame:
    c = cube.copy()
    per = pd.PeriodIndex(c["month"], freq="M")
    c["year"], c["mon"] = per.year, per.month
    c = c[(c["year"] >= y0) & (c["year"] <= y1) & c["mon"].between(*months)]
    tot = c.groupby("cell")["n_det"].sum()
    cells = tot[tot >= min_det].index
    c = c[c["cell"].isin(cells)]
    nyear = y1 - y0 + 1

    # musim: distribusi deteksi per bulan kalender (12 fitur, jumlah = 1)
    season = c.pivot_table(index="cell", columns="mon", values="n_det", aggfunc="sum", fill_value=0)
    season = season.reindex(columns=range(1, 13), fill_value=0)
    season = season.div(season.sum(axis=1), axis=0)
    season.columns = [f"season_{m:02d}" for m in season.columns]

    # sudut musim puncak (sirkular) agar Des dan Jan berdekatan
    ang = 2 * np.pi * (np.arange(12)) / 12
    s = season.to_numpy()
    peak_sin, peak_cos = s @ np.sin(ang), s @ np.cos(ang)
    concentration = np.hypot(peak_sin, peak_cos)            # 1 = satu musim tajam, 0 = sepanjang tahun

    g = c.groupby("cell")
    w = c["n_det"]
    feats = pd.DataFrame({
        "log_det_per_year": np.log1p(g["n_det"].sum() / nyear),
        "recurrence": g["year"].nunique() / nyear,
        "persist_mean": (c["persist_max"] * w).groupby(c["cell"]).sum() / g["n_det"].sum(),
        "persist_max": g["persist_max"].max(),
        "night_frac": (c["night_frac"] * w).groupby(c["cell"]).sum() / g["n_det"].sum(),
        "log_frp_med": np.log1p(g["frp_med"].median()),
        "season_sin": pd.Series(peak_sin, index=season.index),
        "season_cos": pd.Series(peak_cos, index=season.index),
        "season_conc": pd.Series(concentration, index=season.index),
        "lat": g["lat"].first(), "lon": g["lon"].first(),
    })

    if oni is not None and nyear >= 4:
        annual = c[c["mon"].between(7, 11)].pivot_table(index="cell", columns="year", values="n_det",
                                                         aggfunc="sum", fill_value=0)
        annual = annual.reindex(columns=range(y0, y1 + 1), fill_value=0)
        o = oni.reindex(annual.columns).to_numpy()
        ok = ~np.isnan(o)
        if ok.sum() >= 4:
            rho = [spearmanr(row[ok], o[ok]).statistic if row[ok].std() > 0 else 0.0
                   for row in np.log1p(annual.to_numpy())]
            feats["enso_rho"] = pd.Series(np.nan_to_num(rho), index=annual.index)
        else:
            print("PERINGATAN: <4 tahun dengan ONI; enso_rho dilewati (jendela terlalu pendek)")
    return feats.join(season).reset_index()


def attach_static(fp, st, year=None):
    """Gabung atribut statis per sel. Tanpa `year`: rerata semua tahun (mis. embedding 2017-2025)."""
    if "year" in st:
        if year is not None and (st["year"] == year).any():
            st = st[st["year"] == year]
        elif year is not None:                                         # mis. 2014: belum ada embedding AlphaEarth
            print(f"PERINGATAN: static tahun {year} tidak ada; atribut lahan dirata-rata, embedding dibiarkan kosong")
            st = st.assign(**{c: np.nan for c in st if c.startswith("emb_")})
        st = st.drop(columns="year").groupby("cell", as_index=False).mean()
    fp = fp.merge(st, on="cell", how="left")
    if "in_indonesia" in fp:                                   # topeng batas negara (LSIB)
        fp = fp[fp["in_indonesia"].fillna(0) > 0.5].drop(columns="in_indonesia")
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True)
    ap.add_argument("--oni", default=None)
    ap.add_argument("--years", default="2012-2025")
    ap.add_argument("--min-det", type=int, default=20)
    ap.add_argument("--months", default="1-12", help="bulan kalender yang dipakai, mis. 1-10 untuk sidik jari tahun kejadian")
    ap.add_argument("--static", default=None, help="CSV per sel (cell, peat_frac, wc_*, in_indonesia, emb_* ...)")
    ap.add_argument("--static-year", type=int, default=None, help="pakai baris static tahun ini saja (kolom year)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    y0, y1 = map(int, args.years.split("-"))
    cube = pd.read_csv(args.cube)
    oni = load_oni(args.oni) if args.oni else None
    m0, m1 = map(int, args.months.split("-"))
    fp = fingerprint(cube, oni, y0, y1, args.min_det, (m0, m1))
    if args.static:
        fp = attach_static(fp, pd.read_csv(args.static), args.static_year)
    fp.to_csv(args.out, index=False)
    print(f"{len(fp):,} sel dengan >= {args.min_det} deteksi ({y0}-{y1}) -> {args.out}")


if __name__ == "__main__":
    main()
