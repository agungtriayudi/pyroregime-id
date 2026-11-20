"""Jalankan seluruh analisis naskah dari satu konfigurasi.

    python run_all.py --config config.json
Langkah: grid -> sidik jari -> clustering (+ ablasi embedding) -> sidik jari tahun kejadian -> pergeseran rezim
(+ baseline tahun netral) -> prakiraan -> results.json/tabel/gambar -> isi naskah (bila `manuscript` diberikan).
"""
import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score

PY = sys.executable


def run(*args):
    print("\n$", " ".join(map(str, args)), flush=True)
    subprocess.run([PY, *map(str, args)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--allow-deviation", action="store_true",
                    help="jalan walau berbeda dari registrasi; tercatat sebagai deviasi di results.json")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    out = Path(cfg["out"]); out.mkdir(parents=True, exist_ok=True)

    # ---- cek registrasi (desain dibekukan sebelum evaluasi 2026)
    from register import fingerprint
    _, hp, hc = fingerprint(cfg)
    reg_path = Path(cfg.get("registration", "registration/registration.json"))
    if reg_path.exists():
        reg = json.load(open(reg_path))
        same = reg["params_sha256"] == hp and reg["code_sha256"] == hc
        status = "registered" if same else "deviation"
        if not same and not a.allow_deviation:
            raise SystemExit("Parameter/kode analisis berbeda dari registrasi. Kembalikan perubahan, atau jalankan dengan "
                             "--allow-deviation dan catat alasannya di DEVIATIONS.md.")
    else:
        status = "unregistered"
        print("PERINGATAN: belum ada registrasi (python register.py). Hasil ditandai 'unregistered'.")
    work = out / "work"; work.mkdir(exist_ok=True)
    y0, y1 = cfg["ref_years"]

    if cfg.get("firms_glob") and not Path(cfg["cell_month"]).exists():
        run("grid_features.py", "--firms", cfg["firms_glob"], "--out", cfg["cell_month"])
    run("regime_features.py", "--cube", cfg["cell_month"], "--oni", cfg["oni"], "--years", f"{y0}-{y1}",
        "--min-det", cfg.get("min_det", 20), "--static", cfg["static"], "--out", cfg["fingerprint"])
    k = cfg.get("k_range", "3-12")
    run("cluster_regimes.py", "--fp", cfg["fingerprint"], "--method", *cfg.get("methods", ["kmeans", "gmm", "ae_kmeans"]),
        "--k", k, "--boot", cfg.get("cluster_boot", 50), "--out", out / "regimes")
    run("cluster_regimes.py", "--fp", cfg["fingerprint"], "--method", "kmeans", "gmm", "--k", k,
        "--boot", cfg.get("cluster_boot", 50), "--emb-weight", 0, "--out", out / "regimes_noemb")
    full = pd.read_csv(out / "regimes" / "labels_selected.csv")
    sel = json.load(open(out / "regimes" / "selection.json"))
    noemb = pd.read_csv(out / "regimes_noemb" / f"labels_{sel['method'] if sel['method'] in ('kmeans', 'gmm') else 'kmeans'}.csv")
    j = full.merge(noemb, on="cell", suffixes=("", "_ne"))
    json.dump({"ari_without_emb": float(adjusted_rand_score(j["regime"], j["regime_ne"]))}, open(out / "embedding.json", "w"))

    # ketahanan sensor: partisi dari kubus NOAA-20 (2018-2025) pada metode dan k terpilih
    if cfg.get("cell_month_noaa20"):
        fp2 = work / "fp_noaa20.csv"
        run("regime_features.py", "--cube", cfg["cell_month_noaa20"], "--oni", cfg["oni"], "--years", f"2018-{y1}",
            "--min-det", cfg.get("min_det", 20), "--static", cfg["static"], "--out", fp2)
        run("cluster_regimes.py", "--fp", fp2, "--method", sel["method"], "--k", sel["k"], "--boot", 0, "--out", out / "regimes_noaa20")
        n20 = pd.read_csv(out / "regimes_noaa20" / f"labels_{sel['method']}.csv")
        j2 = full.merge(n20, on="cell", suffixes=("", "_n20"))
        sens = {"sensor_ari": float(adjusted_rand_score(j2["regime"], j2["regime_n20"])), "sensor_n_cells": int(len(j2))}
    else:
        sens = {"sensor_ari": None, "sensor_n_cells": 0}
    json.dump(sens, open(out / "sensor.json", "w"))

    def event_fps(years):
        paths = []
        for Y in years:
            p = work / f"fp_ev{Y}.csv"
            run("regime_features.py", "--cube", cfg["cell_month"], "--years", f"{Y}-{Y}", "--months", cfg.get("event_months", "1-10"),
                "--min-det", cfg.get("event_min_det", 3), "--static", cfg["static"], "--static-year", Y - 1, "--out", p)
            paths.append(f"{Y}={p}")
        return paths
    boot = cfg.get("bootstrap", 1000)
    run("regime_shift.py", "--ref-fp", cfg["fingerprint"], "--labels", out / "regimes" / "labels_selected.csv",
        "--windows", *event_fps(cfg["shift_years"]), "--boot", boot, "--out", out / "shift")
    run("regime_shift.py", "--ref-fp", cfg["fingerprint"], "--labels", out / "regimes" / "labels_selected.csv",
        "--windows", *event_fps(cfg["neutral_years"]), "--boot", boot, "--out", out / "shift_neutral")
    fcfg = dict(cfg, labels=str(out / "regimes" / "labels_selected.csv"))
    json.dump(fcfg, open(work / "forecast_config.json", "w"), indent=1)
    run("forecast.py", "--config", work / "forecast_config.json")
    run("report.py", "--config", work / "forecast_config.json")
    # ---- metadata jalannya analisis + pemeriksaan token naskah + bundel untuk diunggah ke claude.ai
    res = json.load(open(out / "results.json"))
    res["meta"] = {"registration_status": status, "params_sha256": hp, "code_sha256": hc,
                   "synthetic": bool(cfg.get("synthetic", False))}
    if Path(cfg.get("provenance", "data/provenance.json")).exists():
        res["meta"]["provenance"] = json.load(open(cfg.get("provenance", "data/provenance.json")))
    json.dump(res, open(out / "results.json", "w"), indent=1, default=float)
    if Path("manuscript_tokens.txt").exists():
        run("fill_manuscript.py", "--check", "manuscript_tokens.txt", "--results", out / "results.json",
            "--tables", out / "tables")
    if cfg.get("manuscript") and Path(cfg["manuscript"]).exists():
        run("fill_manuscript.py", "--manuscript", cfg["manuscript"], "--results", out / "results.json",
            "--tables", out / "tables", "--out", out / "manuscript_filled.md")
    bundle = out / "results_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for p in [out / "results.json", out / "fill_values.json", *sorted((out / "tables").glob("*.md")),
                  *sorted((out / "figures").glob("*.png")), Path("registration/REGISTRATION.md"),
                  Path("DEVIATIONS.md"), Path("RUN_REPORT.md")]:
            if p.exists():
                z.write(p, p.relative_to(out) if out in p.parents else p)
    print(f"\nSELESAI ({status}). Unggah {bundle} ke chat Claude untuk mengisi naskah.")


if __name__ == "__main__":
    main()
