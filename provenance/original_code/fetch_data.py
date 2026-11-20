"""Ambil dan susun SEMUA masukan run_all.py dari sumber publik. Setiap tahap bisa diulang (melanjutkan).

    python fetch_data.py --config config.json                 # semua tahap berurutan
    python fetch_data.py --config config.json --stage verify  # cek kelengkapan saja

Tahap: firms -> grid -> candidates -> indices -> gee_monthly -> gee_static -> peat -> fwi -> verify
Kredensial dibaca dari lingkungan: FIRMS_MAP_KEY; GEE via `earthengine authenticate`; CDS via ~/.cdsapirc.
Semua sumber, rentang tanggal, dan perubahan ID dicatat di cfg["provenance"].
"""
import argparse
import io
import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from firms_download import daterange, fetch

PY = sys.executable
STAGES = ["firms", "grid", "candidates", "indices", "gee_monthly", "gee_static", "peat", "fwi", "verify"]
AVAIL = "https://firms.modaps.eosdis.nasa.gov/api/data_availability/csv/{key}/all"


def sh(*args):
    print("\n$", " ".join(map(str, args)), flush=True)
    subprocess.run([PY, *map(str, args)], check=True)


def load_prov(cfg):
    p = Path(cfg["provenance"])
    return json.load(open(p)) if p.exists() else {}


def save_prov(cfg, prov):
    Path(cfg["provenance"]).parent.mkdir(parents=True, exist_ok=True)
    json.dump(prov, open(cfg["provenance"], "w"), indent=1, default=str)


# ---------------------------------------------------------------- FIRMS
def firms_availability(key):
    r = requests.get(AVAIL.format(key=key), timeout=60)
    if r.status_code != 200:                                        # jangan cetak URL (berisi kunci)
        raise SystemExit(f"FIRMS data_availability HTTP {r.status_code}: {r.text[:200].replace(key, '<KEY>')}")
    df = pd.read_csv(io.StringIO(r.text))
    return {row["data_id"]: (row["min_date"], row["max_date"]) for _, row in df.iterrows()}


def download(key, source, start, end, cfg):
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    # FIRMS menghitung ~1 transaksi per hari data; jaga laju di bawah kuota 5.000 transaksi / 10 menit
    budget = cfg.get("firms_tx_per_10min", 4000)
    lock, nxt = threading.Lock(), [time.monotonic()]

    def one(job):
        f, bbox, d, days = job
        with lock:
            now = time.monotonic()
            wait = nxt[0] - now
            nxt[0] = max(nxt[0], now) + days * 600 / budget
        if wait > 0:
            time.sleep(wait)
        text = fetch(key, source, bbox, days, d)
        tmp = f.with_suffix(".part")                                # tulis atomik: tidak ada CSV terpotong
        tmp.write_text(text)
        tmp.replace(f)

    for region, bbox in cfg["regions"].items():
        out = Path(cfg["firms_dir"]) / source / region
        out.mkdir(parents=True, exist_ok=True)
        jobs = [(out / f"{d.isoformat()}_{days}d.csv", bbox, d, days)
                for d, days in daterange(start, end, cfg.get("firms_day_range", 5))]
        jobs = [j for j in jobs if not j[0].exists()]
        with ThreadPoolExecutor(cfg.get("firms_workers", 4)) as ex:
            list(ex.map(one, jobs))
        print(f"{source} {region}: {len(jobs)} berkas baru ({start} s.d. {end})", flush=True)


def stage_firms(cfg, prov):
    key = os.environ.get("FIRMS_MAP_KEY")
    if not key:
        raise SystemExit("FIRMS_MAP_KEY belum diset (lihat README).")
    av = firms_availability(key)
    prov["firms_availability"] = av
    snpp_sp_end = date.fromisoformat(str(av["VIIRS_SNPP_SP"][1])[:10])
    n20_sp_end = date.fromisoformat(str(av["VIIRS_NOAA20_SP"][1])[:10])
    today = date.today() - timedelta(days=1)
    download(key, "VIIRS_SNPP_SP", date.fromisoformat(cfg["firms_start"]), snpp_sp_end, cfg)
    if snpp_sp_end < today:                                          # NRT hanya setelah akhir arsip SP
        download(key, "VIIRS_SNPP_NRT", snpp_sp_end + timedelta(days=1), today, cfg)
    download(key, "VIIRS_NOAA20_SP", date.fromisoformat("2018-01-01"), min(n20_sp_end, date(2025, 12, 31)), cfg)
    # uji sensitivitas SP vs NRT pada bulan tumpang tindih (tidak masuk kubus)
    ov_start = snpp_sp_end.replace(day=1)
    nrt_start = date.fromisoformat(str(av["VIIRS_SNPP_NRT"][0])[:10])
    if nrt_start > ov_start:                                          # NRT tidak mencakup sebulan penuh arsip SP
        prov["sp_nrt_overlap"] = (f"NRT tumpang tindih tidak tersedia: arsip NRT mulai {nrt_start}, "
                                  f"arsip SP berakhir {snpp_sp_end}")
    elif (today - ov_start).days < 60:
        prov["sp_nrt_overlap"] = "NRT tumpang tindih tidak tersedia (NRT hanya ~2 bulan terakhir)"
    else:
        ov = dict(cfg, firms_dir=str(Path(cfg["firms_dir"]) / "_overlap"))
        try:
            download(key, "VIIRS_SNPP_NRT", ov_start, snpp_sp_end, ov)
        except Exception as ex:
            prov["sp_nrt_overlap"] = f"gagal: {ex}"
    prov["firms"] = {"snpp_sp": [cfg["firms_start"], str(snpp_sp_end)],
                     "snpp_nrt": [str(snpp_sp_end + timedelta(days=1)), str(today)] if snpp_sp_end < today else None,
                     "noaa20_sp": ["2018-01-01", str(min(n20_sp_end, date(2025, 12, 31)))]}


def stage_grid(cfg, prov):
    d = Path(cfg["firms_dir"])
    for k in ("cell_month", "cell_month_noaa20"):                  # grid_features.py tidak membuat folder keluaran
        Path(cfg[k]).parent.mkdir(parents=True, exist_ok=True)
    sh("grid_features.py", "--firms", str(d / "VIIRS_SNPP_[SN]*" / "**" / "*.csv"), "--out", cfg["cell_month"])
    sh("grid_features.py", "--firms", str(d / "VIIRS_NOAA20_SP" / "**" / "*.csv"), "--out", cfg["cell_month_noaa20"])
    ovl = list((d / "_overlap").rglob("*.csv"))
    if ovl:                                                         # SP vs NRT, bulan yang sama
        sp_end = prov["firms"]["snpp_sp"][1][:7]
        tmp = Path(cfg["cell_month"]).with_name("overlap_nrt.csv.gz")
        sh("grid_features.py", "--firms", str(d / "_overlap" / "**" / "*.csv"), "--out", tmp)
        nrt = pd.read_csv(tmp); sp = pd.read_csv(cfg["cell_month"])
        a, b = sp[sp["month"] == sp_end]["n_det"].sum(), nrt[nrt["month"] == sp_end]["n_det"].sum()
        prov["sp_nrt_overlap"] = {"month": sp_end, "sp_detections": int(a), "nrt_detections": int(b),
                                  "ratio_nrt_sp": float(b / a) if a else None}


def stage_candidates(cfg, prov):
    cm = pd.read_csv(cfg["cell_month"])
    per = pd.PeriodIndex(cm["month"], freq="M"); cm["year"], cm["mon"] = per.year, per.month
    y0, y1 = cfg["ref_years"]
    ref = cm[cm["year"].between(y0, y1)].groupby("cell")["n_det"].sum()
    forecast_cells = ref[ref >= cfg["min_det"]].index
    m0, m1 = map(int, cfg["event_months"].split("-"))
    ev = cm[cm["year"].isin(cfg["shift_years"] + cfg["neutral_years"]) & cm["mon"].between(m0, m1)]
    ev = ev.groupby(["cell", "year"])["n_det"].sum()
    event_cells = ev[ev >= cfg["event_min_det"]].index.get_level_values("cell").unique()
    n20 = pd.read_csv(cfg["cell_month_noaa20"])
    n20 = n20[pd.PeriodIndex(n20["month"], freq="M").year.isin(range(2018, y1 + 1))].groupby("cell")["n_det"].sum()
    n20_cells = n20[n20 >= cfg["min_det"]].index
    cand = pd.Index(forecast_cells).union(event_cells).union(n20_cells)
    pd.DataFrame({"cell": cand}).to_csv(cfg["candidate_cells"], index=False)
    pd.DataFrame({"cell": forecast_cells}).to_csv(cfg["forecast_cells"], index=False)
    prov["cells"] = {"candidate": int(len(cand)), "forecast_domain_before_lsib": int(len(forecast_cells))}
    print(f"kandidat {len(cand):,} sel; domain prakiraan (sebelum topeng LSIB) {len(forecast_cells):,} sel")


def stage_verify(cfg, prov):
    problems = []
    for k in ("cell_month", "cell_month_noaa20", "gee_monthly", "static", "fwi", "oni", "dmi"):
        if not Path(cfg[k]).exists():
            problems.append(f"{k}: berkas {cfg[k]} tidak ada")
    if not problems:
        end = cfg["panel_end"]
        cm = pd.read_csv(cfg["cell_month"], usecols=["month"])
        gm = pd.read_csv(cfg["gee_monthly"], usecols=["month"])
        fw = pd.read_csv(cfg["fwi"], usecols=["month"])
        dmi = pd.read_csv(cfg["dmi"])
        st = pd.read_csv(cfg["static"], nrows=5)
        # target butuh FIRMS sampai panel_end; prediktor cukup sampai bulan sebelumnya (bulan terbit prakiraan)
        pred_end = str(pd.Period(end, freq="M") - 1)
        checks = {"FIRMS": (cm["month"].max(), end), "GEE bulanan": (gm["month"].max(), pred_end),
                  "FWI": (fw["month"].max(), pred_end), "DMI": (dmi["month"].max(), pred_end)}
        oni_last = Path(cfg["oni"]).read_text().strip().splitlines()[-1].split()
        prov["oni_last_season"] = " ".join(oni_last[:2])
        for name, (last, need) in checks.items():
            if last < need:
                problems.append(f"{name} berakhir {last}, dibutuhkan sampai {need} (turunkan panel_end atau cari sumber lebih baru)")
        for col in ("peat_frac", "in_indonesia", "emb_A00", "wc_tree"):
            if col not in st:
                problems.append(f"static tidak punya kolom {col}")
    prov["verify"] = {"ok": not problems, "problems": problems}
    print("VERIFIKASI:", "semua masukan lengkap" if not problems else "\n  - " + "\n  - ".join(problems))
    if problems:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    stages = STAGES if a.stage == "all" else [a.stage]
    for st in stages:
        print(f"\n===== tahap {st} =====", flush=True)
        prov = load_prov(cfg)
        if st == "firms":
            stage_firms(cfg, prov)
        elif st == "grid":
            stage_grid(cfg, prov)
        elif st == "candidates":
            stage_candidates(cfg, prov)
        elif st == "indices":
            save_prov(cfg, prov); sh("climate_fetch.py", "--config", a.config, "--what", "indices"); continue
        elif st == "gee_monthly":
            sh("gee_fetch.py", "--config", a.config, "--kind", "monthly"); continue
        elif st == "gee_static":
            sh("gee_fetch.py", "--config", a.config, "--kind", "static"); continue
        elif st == "peat":
            sh("peat_fraction.py", "--config", a.config); continue
        elif st == "fwi":
            save_prov(cfg, prov); sh("climate_fetch.py", "--config", a.config, "--what", "fwi"); continue
        elif st == "verify":
            try:
                stage_verify(cfg, prov)
            finally:
                save_prov(cfg, prov)
            continue
        save_prov(cfg, prov)


if __name__ == "__main__":
    main()
