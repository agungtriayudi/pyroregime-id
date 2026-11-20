"""Download climate indices (ONI, DMI) and monthly FWI, and sample FWI to the cells.

    python climate_fetch.py --config config.json --what indices
    python climate_fetch.py --config config.json --what fwi

ONI : NOAA CPC text (SEAS YR TOTAL ANOM) -> cfg["oni"], stored as is.
DMI : NOAA PSL format, year x 12 months -> cfg["dmi"] (columns month, dmi).
FWI : Copernicus CDS, dataset and request template in cfg["fwi_cds"]; needs ~/.cdsapirc (see README).
      Monthly mean of the daily data, bilinear interpolation to the centres of the forecast-domain cells -> cfg["fwi"].
Data-source parameters (URLs, dataset IDs) are not analysis parameters: they may be corrected if they change, and
every change is recorded in data/provenance.json.
"""
import argparse
import copy
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests


def get_text(urls):
    last = None
    for u in urls:
        try:
            r = requests.get(u, timeout=60)
            if r.status_code == 200 and len(r.text) > 100:
                return u, r.text
            last = f"{u}: HTTP {r.status_code}"
        except Exception as ex:
            last = f"{u}: {ex}"
    raise RuntimeError(f"all URLs failed; last {last}")


def parse_psl_monthly(text, name):
    """PSL format: line 1 = first_year last_year; then lines 'year v1..v12'; the missing-value code follows them."""
    lines = [l.split() for l in text.strip().splitlines()]
    y0, y1 = int(lines[0][0]), int(lines[0][1])
    missing = None
    rows = []
    for parts in lines[1:]:
        if len(parts) == 13 and parts[0].isdigit() and y0 <= int(parts[0]) <= y1:
            rows.append(parts)
        elif len(parts) == 1 and missing is None and rows:
            missing = float(parts[0])
    out = []
    for p in rows:
        for m in range(12):
            v = float(p[m + 1])
            if missing is not None and np.isclose(v, missing):
                continue
            out.append({"month": f"{int(p[0])}-{m + 1:02d}", name: v})
    return pd.DataFrame(out)


def parse_cpc_dmi(text):
    """NOAA CPC IODMI format: lines 'Year Month WTIO SETIO DMI' (ERSSTv6, 1991-2020 climatology)."""
    out = []
    for l in text.splitlines():
        p = l.split()
        if len(p) == 5 and p[0].isdigit() and len(p[0]) == 4 and p[1].isdigit():
            out.append({"month": f"{int(p[0])}-{int(p[1]):02d}", "dmi": float(p[4])})
    return pd.DataFrame(out)


def indices(cfg, prov):
    u, t = get_text(cfg["oni_urls"])
    Path(cfg["oni"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["oni"]).write_text(t)
    prov["oni"] = {"url": u, "last": t.strip().splitlines()[-1]}
    u, t = get_text(cfg["dmi_urls"])
    dmi = parse_cpc_dmi(t) if "WTIO - SETIO" in t else parse_psl_monthly(t, "dmi")
    dmi.to_csv(cfg["dmi"], index=False)
    prov["dmi"] = {"url": u, "first": dmi["month"].min(), "last": dmi["month"].max()}
    need = cfg["panel_end"]
    for k, last in (("oni", None), ("dmi", dmi["month"].max())):
        if last is not None and last < need:
            print(f"WARNING: {k} only until {last}, panel until {need}. Find another source (see docs/REPRODUCE.md).")


def fwi(cfg, prov):
    import cdsapi
    import xarray as xr
    fp = pd.read_csv(cfg["forecast_cells"])["cell"].to_numpy()
    lat = (fp // 100000) * 0.01 - 90 + 0.005
    lon = (fp % 100000) * 0.01 - 180 + 0.005
    c = cdsapi.Client(url=cfg["fwi_cds"].get("url"))                # URL from the config (EWDS); the key still comes from ~/.cdsapirc
    raw = Path(cfg.get("fwi_raw", "data/raw/fwi")); raw.mkdir(parents=True, exist_ok=True)
    y0, y1 = int(cfg["panel_start"][:4]), int(cfg["panel_end"][:4])
    frames = []
    for y in range(y0, y1 + 1):
        target = raw / f"fwi_{y}.nc"
        if not target.exists():
            req = copy.deepcopy(cfg["fwi_cds"]["request"])
            req["year"] = [str(y)]
            if y == y1:                                              # current year: intermediate dataset
                req.update(cfg["fwi_cds"].get("current_year_overrides", {}))
            c.retrieve(cfg["fwi_cds"]["dataset"], req, str(target))
        ds = xr.open_dataset(target)
        var = [v for v in ds.data_vars if ds[v].ndim >= 3][0]
        tname = [d for d in ds[var].dims if "time" in d][0]
        la = [d for d in ds[var].dims if d.startswith("lat")][0]; lo = [d for d in ds[var].dims if d.startswith("lon")][0]
        da = ds[var]
        if float(da[lo].max()) > 180:
            da = da.assign_coords({lo: ((da[lo] + 180) % 360) - 180}).sortby(lo)
        mon = da.resample({tname: "MS"}).mean()
        pts = mon.interp({la: xr.DataArray(lat, dims="cell"), lo: xr.DataArray(lon, dims="cell")}, method="linear")
        df = pts.to_dataframe(name="fwi").reset_index()
        df["cell"] = fp[df["cell"].to_numpy()]                     # dimension "cell" = position in fp
        df["month"] = pd.to_datetime(df[tname]).dt.strftime("%Y-%m")
        frames.append(df[["cell", "month", "fwi"]])
        print(f"FWI {y}: {len(df):,} rows ({var})", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(cfg["fwi"], index=False, float_format="%.4g")
    prov["fwi"] = {"dataset": cfg["fwi_cds"]["dataset"], "variable_in_file": var,
                   "first": out["month"].min(), "last": out["month"].max(), "n_cells": int(len(fp))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--what", choices=["indices", "fwi"], required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    pp = Path(cfg.get("provenance", "data/provenance.json"))
    prov = json.load(open(pp)) if pp.exists() else {}
    (indices if a.what == "indices" else fwi)(cfg, prov)
    pp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(prov, open(pp, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
