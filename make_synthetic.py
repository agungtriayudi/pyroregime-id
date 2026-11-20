"""Complete SYNTHETIC inputs (same format as the real data) for testing run_all.py without internet.

    python make_synthetic.py --out data/synthetic
Produces: firms/firms_synthetic.csv, oni.ascii.txt, dmi.csv, gee_monthly.csv.gz, static.csv, fwi.csv, truth.csv
All parameters are invented for software testing; do NOT report them as results.

Planted regimes:
  0 smouldering peat : Aug-Oct (+ small Feb-Mar), very drought-sensitive, persistent, many night detections, low FRP
  1 land clearing    : Jun-Sep, moderately drought-sensitive, short, daytime, high FRP
  2 Papuan savanna   : Sep-Dec, weakly drought-sensitive, medium persistence
2026 shift: 25 % of the regime-2 cells behave like regime 1, and 50 new Papuan cells start burning (regime 1).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MONTHS = pd.period_range("2012-01", "2026-10", freq="M")
CLIM = {  # climatological rainfall (mm/month) per region
    "sum": [260, 190, 220, 250, 200, 130, 110, 120, 150, 230, 280, 280],
    "kal": [300, 280, 300, 280, 220, 150, 110, 90, 110, 190, 280, 320],
    "pap": [300, 280, 300, 250, 160, 90, 50, 30, 30, 60, 140, 250],
}
ONI_PEAK = {2015: 2.4, 2019: 0.6, 2023: 1.9, 2026: 2.8}
REG = {  # region, bbox, season (weight per month), base rate, drought beta, duration, detections/day, night, FRP, peat
    0: ("sumkal", None, {2: .3, 3: .3, 8: 1, 9: 1, 10: .8}, 0.10, 2.2, 8.0, 3.0, 0.45, 8.0, 0.8),
    1: ("sum", (98.0, 0.5, 103.0, 4.0), {6: .6, 7: 1, 8: 1, 9: .7}, 0.30, 0.8, 1.3, 1.5, 0.08, 25.0, 0.1),
    2: ("pap", (137.8, -8.8, 140.8, -6.0), {9: .6, 10: 1, 11: 1, 12: .6}, 0.30, 0.4, 2.5, 2.0, 0.20, 15.0, 0.05),
}
BOX = {"sum": (101.0, -3.5, 106.0, 1.5), "kal": (111.0, -3.5, 115.5, 0.5)}


def oni_series(rng):
    t = np.arange(len(MONTHS))
    oni = 0.4 * np.sin(2 * np.pi * t / 44) + rng.normal(0, 0.15, len(t))
    for y, a in ONI_PEAK.items():
        centre = (y - 2012) * 12 + 10                                     # peak ~Nov
        oni += a * np.exp(-0.5 * ((t - centre) / 4.0) ** 2)
    return pd.Series(oni, index=MONTHS)


def write_oni(oni, path):
    seas = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
    with open(path, "w") as f:
        f.write("SEAS  YR   TOTAL   ANOM\n")
        v = oni.to_numpy()
        for i, p in enumerate(MONTHS[1:-1], start=1):                    # season centred on month p
            a = v[i - 1:i + 2].mean()
            f.write(f"  {seas[(p.month - 1) % 12]}  {p.year}   {26.5 + a:.2f}   {a:.2f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--cells", type=int, default=140)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out); (out / "firms").mkdir(parents=True, exist_ok=True)
    oni = oni_series(rng)
    write_oni(oni, out / "oni.ascii.txt")
    pd.DataFrame({"month": MONTHS.astype(str), "dmi": 0.5 * oni.to_numpy() + rng.normal(0, 0.2, len(oni))}).to_csv(out / "dmi.csv", index=False)

    # ---- cells
    cells = []
    for r, (reg, bb, *_rest) in REG.items():
        for i in range(args.cells):
            region = reg if reg != "sumkal" else ("sum" if i % 2 == 0 else "kal")
            b = bb or BOX[region]
            cells.append({"lat": rng.uniform(b[1], b[3]), "lon": rng.uniform(b[0], b[2]), "region": region,
                          "regime": r, "regime_2026": r, "new2026": False})
    pap2 = [i for i, c in enumerate(cells) if c["regime"] == 2]
    for i in rng.choice(pap2, len(pap2) // 4, replace=False):
        cells[i]["regime_2026"] = 1
    for _ in range(50):
        b = REG[2][1]
        cells.append({"lat": rng.uniform(b[1], b[3]), "lon": rng.uniform(b[0], b[2]), "region": "pap",
                      "regime": -1, "regime_2026": 1, "new2026": True})
    C = pd.DataFrame(cells)
    C["lat"], C["lon"] = C["lat"].round(2) + 0.005, C["lon"].round(2) + 0.005
    from grid_features import cell_of
    C["cell"] = cell_of(C["lat"].to_numpy(), C["lon"].to_numpy())
    C = C.drop_duplicates("cell").reset_index(drop=True)
    C["peat_frac"] = [np.clip(REG[r][9] + rng.normal(0, .1), 0, 1) if r >= 0 else 0.05 for r in C["regime"]]

    # ---- monthly hydroclimate per region, then per cell
    mon = np.array([p.month for p in MONTHS]); yr = np.array([p.year for p in MONTHS])
    reg_p = {}
    for region, clim in CLIM.items():
        k = {"sum": .25, "kal": .35, "pap": .30}[region]
        dry_season = np.isin(mon, [6, 7, 8, 9, 10, 11])
        eff = np.where(dry_season, k, 0.1)
        reg_p[region] = np.array(clim)[mon - 1] * np.exp(-eff * oni.to_numpy() + rng.normal(0, 0.2, len(mon)))
    fire_rows, gee_rows, fwi_rows = [], [], []
    n_m = len(MONTHS)
    for _, c in C.iterrows():
        p = reg_p[c["region"]] * np.exp(rng.normal(0, 0.1, n_m))
        clim = np.array(CLIM[c["region"]])[mon - 1]
        p3 = pd.Series(p).rolling(3, min_periods=1).sum().to_numpy()
        c3 = pd.Series(clim).rolling(3, min_periods=1).sum().to_numpy()
        D = -np.log(p3 / c3)                                                # >0 = drier than normal
        dry_days = np.clip(np.round(30 / (1 + np.exp((p - 120) / 40)) + rng.normal(0, 1.5, n_m)), 0, 30)
        sw = np.clip((0.45 if c["peat_frac"] > .5 else 0.25) + 0.12 * np.tanh(-D) + rng.normal(0, .02, n_m), .05, .7)
        t2m = 299.5 + 1.2 * D + rng.normal(0, .3, n_m); d2m = t2m - (2.5 + 2.5 * np.clip(D, 0, None))
        fwi = np.clip(8 + 12 * D + rng.normal(0, 4, n_m), 0, None)
        burned = np.zeros(n_m)
        for ti in range(n_m):
            y, m = yr[ti], mon[ti]
            r = c["regime_2026"] if y == 2026 else c["regime"]
            if r < 0:
                continue
            _, _, season, base, beta, dur, dpd, night, frp, _ = REG[r]
            lam = base * season.get(m, 0.02) * np.exp(beta * D[ti])
            for _ in range(rng.poisson(lam)):
                start = pd.Timestamp(y, m, int(rng.integers(1, 28)))
                L = max(1, int(rng.exponential(dur)))
                burned[ti] = min(1, burned[ti] + 0.05 * L)
                for d in range(L):
                    for _ in range(max(1, rng.poisson(dpd))):
                        fire_rows.append((c["lat"] - 0.005 + rng.uniform(0, .0099), c["lon"] - 0.005 + rng.uniform(0, .0099),
                                          (start + pd.Timedelta(days=d)).date().isoformat(), int(rng.integers(0, 2400)),
                                          rng.choice(["n", "h"]), float(rng.lognormal(np.log(frp), .6)),
                                          "N" if rng.random() < night else "D", 0))
        for ti in range(n_m):
            gee_rows.append((c["cell"], str(MONTHS[ti]), p[ti], dry_days[ti], sw[ti], t2m[ti], d2m[ti], burned[ti], 0.0))
            fwi_rows.append((c["cell"], str(MONTHS[ti]), fwi[ti]))
    # static source (gas flare) that the type filter must remove
    for _ in range(3000):
        fire_rows.append((1.60 + rng.uniform(0, .005), 101.40 + rng.uniform(0, .005),
                          str(pd.Timestamp("2012-01-01") + pd.Timedelta(days=int(rng.integers(0, 5000))))[:10],
                          int(rng.integers(0, 2400)), "h", 50.0, "N", 2))
    cols = ["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp", "daynight", "type"]
    pd.DataFrame(fire_rows, columns=cols).to_csv(out / "firms" / "firms_synthetic.csv", index=False)
    pd.DataFrame(gee_rows, columns=["cell", "month", "precip_mm", "dry_days", "swvl1", "t2m", "d2m",
                                    "burned_frac", "modis_fire_days"]).to_csv(out / "gee_monthly.csv.gz", index=False)
    pd.DataFrame(fwi_rows, columns=["cell", "month", "fwi"]).to_csv(out / "fwi.csv", index=False)

    # ---- static per year: land cover, country mask, 64-D unit-length embeddings per year 2017-2025
    V = rng.normal(size=(3, 64)); u_peat, u_burn = rng.normal(size=64), rng.normal(size=64)
    fire_df = pd.DataFrame(fire_rows, columns=cols)
    st_rows = []
    for _, c in C.iterrows():
        r = max(c["regime"], 1) if c["regime"] < 0 else c["regime"]
        wc = {0: (.5, .2, .1, .2), 1: (.2, .2, .1, .5), 2: (.1, .3, .5, .1)}[r]
        for y in range(2017, 2026):
            e = V[r] + 0.6 * c["peat_frac"] * u_peat + rng.normal(0, .35, 64)
            e = e / np.linalg.norm(e)
            st_rows.append({"cell": c["cell"], "year": y, "peat_frac": c["peat_frac"], "wc_tree": wc[0],
                            "wc_shrub": wc[1], "wc_grass": wc[2], "wc_crop": wc[3], "in_indonesia": 1.0,
                            **{f"emb_A{i:02d}": e[i] for i in range(64)}})
    pd.DataFrame(st_rows).to_csv(out / "static.csv", index=False)
    C[["cell", "lat", "lon", "region", "regime", "regime_2026", "new2026", "peat_frac"]].rename(
        columns={"regime": "true_regime", "regime_2026": "true_regime_2026"}).to_csv(out / "truth.csv", index=False)
    print(f"{len(fire_rows):,} synthetic detections, {len(C)} cells, {n_m} months -> {out}")


if __name__ == "__main__":
    main()
