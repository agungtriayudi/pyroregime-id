"""Redraw Figures 3, 5 and 6 + a table of newly burning cells by province, ONLY from outputs of the registered run.

    python followup/fig_redraw.py --config config.json [--adm1 data/raw/boundaries/idn_adm1.geojson]

Runs no analysis; changes nothing under results/ except writing to results/followup/.
Outputs:
  results/followup/figures/fig3_regime_map.png     map of the 11 regimes, a distinct colour per regime, province boundaries (if given)
  results/followup/figures/fig5_seasonality.png    small multiples, one panel per regime
  results/followup/figures/fig6_transitions.png    transition matrices (%) + map of change 2023 -> 2026
  results/followup/tables/appeared_2026_by_province.md / .csv
  results/followup/fig_redraw.json                 consistency-check numbers against results.json
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 12 easily distinguished qualitative colours (tab10 + black + olive grey); enough for k <= 12
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
           "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#000000", "#6b8e23"]
MONTHS = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]


def load_adm1(path):
    if not path or not Path(path).exists():
        return None
    import geopandas as gpd
    g = gpd.read_file(path).to_crs("EPSG:4326")
    name = next((c for c in ("shapeName", "NAME_1", "name", "PROVINSI", "Provinsi") if c in g), None)
    if name is None:
        raise SystemExit(f"province-name column not recognised in {path}: {list(g.columns)}")
    return g.rename(columns={name: "province"})[["province", "geometry"]]


def draw_adm(ax, adm):
    if adm is not None:
        adm.boundary.plot(ax=ax, color="0.55", linewidth=0.3, zorder=0)


def fig3(lab, adm, path):
    fig, ax = plt.subplots(figsize=(12, 4.6))
    draw_adm(ax, adm)
    for r in sorted(lab["regime"].unique()):
        s = lab[lab["regime"] == r]
        ax.scatter(s["lon"], s["lat"], s=1.2, marker="s", color=PALETTE[int(r) % len(PALETTE)], linewidths=0,
                   label=f"Regime {r} (n = {len(s):,})", rasterized=True)
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°)"); ax.set_aspect("equal")
    ax.set_xlim(94.5, 141.6); ax.set_ylim(-9.6, 6.4)
    leg = ax.legend(loc="lower left", ncol=4, fontsize=7.5, markerscale=7, frameon=True)
    leg.get_frame().set_alpha(0.9)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig5(fp, lab, path):
    season = [f"season_{m:02d}" for m in range(1, 13)]
    d = fp[["cell"] + season].merge(lab[["cell", "regime"]], on="cell")
    allm = d[season].mean().to_numpy()
    regs = sorted(d["regime"].unique())
    nc = 4; nr = int(np.ceil(len(regs) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(11, 2.3 * nr), sharex=True, sharey=True, squeeze=False)
    for ax, r in zip(axes.flat, regs):
        m = d[d["regime"] == r][season].mean().to_numpy()
        ax.plot(range(1, 13), allm, color="0.7", lw=1.2, label="All cells")
        ax.plot(range(1, 13), m, color=PALETTE[int(r) % len(PALETTE)], lw=2.2, marker="o", ms=3)
        ax.set_title(f"Regime {r} (n = {int((d['regime'] == r).sum()):,})", fontsize=9)
        ax.set_xticks(range(1, 13)); ax.set_xticklabels(MONTHS, fontsize=7)
    for ax in list(axes.flat)[len(regs):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Share of detections", fontsize=8)
    axes.flat[0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def fig6(shift_dir, ev, adm, path):
    pairs = [("2015", "2019"), ("2019", "2023"), ("2023", "2026")]
    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    for i, (a, b) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, i])
        T = pd.read_csv(shift_dir / f"transition_{a}_{b}.csv", index_col=0) * 100
        im = ax.imshow(T.to_numpy(), cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(T.shape[1])); ax.set_xticklabels(T.columns, fontsize=8)
        ax.set_yticks(range(T.shape[0])); ax.set_yticklabels(T.index, fontsize=8)
        for (y, x), v in np.ndenumerate(T.to_numpy()):
            if v >= 10:
                ax.text(x, y, f"{v:.0f}", ha="center", va="center", fontsize=6.5, color="white" if v > 55 else "black")
        ax.set_title(f"{a} → {b} (% of row)", fontsize=10); ax.set_xlabel(f"Regime in {b}", fontsize=8)
        if i == 0:
            ax.set_ylabel(f"Regime in earlier year", fontsize=8)
    fig.colorbar(im, ax=fig.axes[:3], fraction=0.015, pad=0.01, label="%")
    # map 2023 -> 2026
    a23, a26 = ev[2023], ev[2026]
    both = a23.merge(a26, on="cell", suffixes=("_a", "_b"))
    cat = {"same regime": both[both["regime_a"] == both["regime_b"]]["cell"],
           "changed regime": both[both["regime_a"] != both["regime_b"]]["cell"],
           "burned only in 2026 (appeared)": a26[~a26["cell"].isin(a23["cell"])]["cell"],
           "burned only in 2023 (disappeared)": a23[~a23["cell"].isin(a26["cell"])]["cell"]}
    colors = {"same regime": "0.6", "changed regime": "#d62728",
              "burned only in 2026 (appeared)": "#ff7f0e", "burned only in 2023 (disappeared)": "#1f77b4"}
    ax = fig.add_subplot(gs[1, :])
    draw_adm(ax, adm)
    for k, cells in cat.items():
        c = cells.to_numpy()
        lat = (c // 100000) * 0.01 - 90 + 0.005; lon = (c % 100000) * 0.01 - 180 + 0.005
        ax.scatter(lon, lat, s=0.8, marker="s", color=colors[k], linewidths=0, label=f"{k} (n = {len(c):,})", rasterized=True)
    ax.set_aspect("equal"); ax.set_xlim(94.5, 141.6); ax.set_ylim(-9.6, 6.4)
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°)")
    ax.legend(loc="lower left", fontsize=8, markerscale=8, ncol=2)
    ax.set_title("2023 → 2026", fontsize=10)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    return {k: int(len(v)) for k, v in cat.items()}


def province_table(ev, regions, adm, out_dir):
    a23, a26 = ev[2023], ev[2026]
    app = a26[~a26["cell"].isin(a23["cell"])].copy()
    app["lat"] = (app["cell"] // 100000) * 0.01 - 90 + 0.005
    app["lon"] = (app["cell"] % 100000) * 0.01 - 180 + 0.005
    app["region"] = "other"
    for name, (w, s, e, n) in regions.items():
        app.loc[app["lon"].between(w, e) & app["lat"].between(s, n), "region"] = name
    if adm is not None:
        import geopandas as gpd
        pts = gpd.GeoDataFrame(app, geometry=gpd.points_from_xy(app["lon"], app["lat"]), crs="EPSG:4326")
        app = gpd.sjoin(pts, adm, how="left", predicate="within").drop(columns=["geometry", "index_right"])
        app["province"] = app["province"].fillna("(outside ADM1 polygons)")
    else:
        app["province"] = "(no ADM1 file)"
    t = (app.groupby(["region", "province"]).agg(cells=("cell", "size"), assigned_regime_mode=("regime", lambda s: int(s.mode().iloc[0])))
         .reset_index().sort_values("cells", ascending=False))
    t["share_%"] = (100 * t["cells"] / t["cells"].sum()).round(1)
    t.to_csv(out_dir / "appeared_2026_by_province.csv", index=False)
    lines = ["| Region | Province | Cells | Share (%) | Most common assigned regime |", "| --- | --- | --- | --- | --- |"]
    lines += [f"| {r.region} | {r.province} | {r.cells:,} | {r['share_%']} | {r.assigned_regime_mode} |"
              for _, r in t.iterrows()]
    (out_dir / "appeared_2026_by_province.md").write_text("\n".join(lines) + "\n")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adm1", default=None, help="GeoJSON/GPKG of province boundaries (ADM1); optional")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    res = Path(cfg["out"]); out = res / "followup"
    (out / "figures").mkdir(parents=True, exist_ok=True); (out / "tables").mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(res / "regimes" / "labels_selected.csv")
    fp = pd.read_csv(cfg["fingerprint"])
    adm = load_adm1(a.adm1)
    ev = {}
    for Y in (2015, 2019, 2023, 2026):
        p = res / "shift" / f"assigned_{Y}.csv"
        ev[Y] = pd.read_csv(p)
    fig3(lab, adm, out / "figures" / "fig3_regime_map.png")
    fig5(fp, lab, out / "figures" / "fig5_seasonality.png")
    counts = fig6(res / "shift", ev, adm, out / "figures" / "fig6_transitions.png")
    t = province_table(ev, cfg["regions"], adm, out / "tables")
    # consistency check against the registered results (Table 5)
    reg = json.load(open(res / "results.json"))
    check = {"counts_2023_2026": counts, "n_appeared_by_province_total": int(t["cells"].sum()),
             "adm1_file": a.adm1, "registered_shift_2023_2026": reg.get("shift", {}).get("2023_2026")}
    json.dump(check, open(out / "fig_redraw.json", "w"), indent=1, default=str)
    print(json.dumps(check, indent=1, default=str))


if __name__ == "__main__":
    main()
