"""Ambil lapisan Google Earth Engine langsung ke tabel sel (tanpa ekspor Google Drive) via ee.data.computePixels.

    python gee_fetch.py --config config.json --kind monthly     # -> cfg["gee_monthly"]
    python gee_fetch.py --config config.json --kind static      # -> cfg["static"] (tanpa peat; peat_fraction.py menambahkannya)

Grid output identik dengan grid_features.py: piksel 0,01 derajat yang tepinya berimpit dengan tepi sel.
Hanya sel kandidat (data/cube/candidate_cells.csv) yang disimpan. Setiap potongan (tile x tahun) disimpan
terpisah di data/gee_parts/ sehingga proses bisa dilanjutkan setelah terputus.

ID koleksi GEE ada di config["gee_ids"]; bila GEE menolak ID, periksa katalog GEE, perbarui config, dan catat
di data/provenance.json (bagian gee_ids bukan parameter analisis sehingga tidak mengubah hash registrasi).
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

RES = 0.01
NROW = 18000          # baris grid global (dari utara)


def tiles_for(bbox, tile_px):
    """Tile berindeks integer (kolom dari -180, baris dari +90)."""
    w, s, e, n = bbox
    c0, c1 = int(np.floor((w + 180) / RES)), int(np.ceil((e + 180) / RES))
    r0, r1 = int(np.floor((90 - n) / RES)), int(np.ceil((90 - s) / RES))
    for r in range(r0, r1, tile_px):
        for c in range(c0, c1, tile_px):
            yield r, c, min(tile_px, r1 - r), min(tile_px, c1 - c)


def cell_ids(r, c, ny, nx):
    rows = np.arange(r, r + ny)[:, None]
    cols = np.arange(c, c + nx)[None, :]
    iy = (NROW - 1 - rows)                         # sama dengan floor((lat_c + 90) / RES) di grid_features
    return (iy * 100000 + cols).astype(np.int64)   # (ny, nx)


def compute(ee, image, r, c, ny, nx, retries=6):
    req = {"expression": image, "fileFormat": "NUMPY_NDARRAY",
           "grid": {"dimensions": {"width": nx, "height": ny},
                    "affineTransform": {"scaleX": RES, "shearX": 0, "translateX": -180 + c * RES,
                                        "shearY": 0, "scaleY": -RES, "translateY": 90 - r * RES},
                    "crsCode": "EPSG:4326"}}
    for k in range(retries):
        try:
            return ee.data.computePixels(req)
        except Exception as ex:                    # kuota/timeout sementara -> tunggu lalu ulang
            msg = str(ex)
            if k == retries - 1:
                raise
            time.sleep(min(300, 10 * 2 ** k) if ("oo many" in msg or "ate" in msg or "imed out" in msg) else 5 * (k + 1))


# ---------------------------------------------------------------- definisi citra
def monthly_image(ee, ids, year, last_month=12):
    bands = []
    for m in range(1, last_month + 1):
        s = ee.Date.fromYMD(year, m, 1); e = s.advance(1, "month")
        ba_col = ee.ImageCollection(ids["mcd64a1"]).filterDate(s, e)
        # reduceResolution hanya di cabang yang punya data: citra konstan pengganti tidak punya proyeksi bawaan
        burned = ee.Image(ee.Algorithms.If(ba_col.size().gt(0),
                                           ee.Image(ba_col.first()).select("BurnDate").gt(0).unmask(0)
                                           .reduceResolution(ee.Reducer.mean(), maxPixels=1024),
                                           ee.Image.constant(0).updateMask(0)))
        burned = burned.rename(f"burned_frac_{m:02d}")
        fire = (ee.ImageCollection(ids["mod14a1"]).merge(ee.ImageCollection(ids["myd14a1"])).filterDate(s, e)
                .map(lambda i: i.select("FireMask").gte(7)).sum().rename(f"modis_fire_days_{m:02d}"))
        ch = ee.ImageCollection(ids["chirps"]).filterDate(s, e).select("precipitation").map(lambda i: i.resample("bilinear"))
        precip = ch.sum().rename(f"precip_mm_{m:02d}")
        dry = ch.map(lambda i: i.lt(1)).sum().rename(f"dry_days_{m:02d}")
        era = (ee.Image(ee.ImageCollection(ids["era5land_monthly"]).filterDate(s, e).first())
               .select(["volumetric_soil_water_layer_1", "temperature_2m", "dewpoint_temperature_2m"],
                       [f"swvl1_{m:02d}", f"t2m_{m:02d}", f"d2m_{m:02d}"]).resample("bilinear"))
        bands += [burned, fire, precip, dry, era]
    return ee.Image.cat(bands).toFloat()


def static_once_image(ee, ids):
    wc = ee.Image(ee.ImageCollection(ids["worldcover"]).first()).select("Map")
    classes = {"wc_tree": 10, "wc_shrub": 20, "wc_grass": 30, "wc_crop": 40, "wc_built": 50,
               "wc_bare": 60, "wc_water": 80, "wc_wetland": 90, "wc_mangrove": 95}
    wcf = [wc.eq(v).reduceResolution(ee.Reducer.mean(), maxPixels=65535).rename(k) for k, v in classes.items()]
    lsib = ee.FeatureCollection(ids["lsib"]).filter(ee.Filter.eq("country_na", "Indonesia"))
    fine = [0.001, 0, -180, 0, -0.001, 90]
    idn = (ee.Image.constant(0).byte().paint(lsib, 1).setDefaultProjection("EPSG:4326", fine)
           .reduceResolution(ee.Reducer.mean(), maxPixels=1024).rename("in_indonesia"))
    return ee.Image.cat(wcf + [idn]).toFloat()


def embedding_image(ee, ids, year, region_geom):
    col = ee.ImageCollection(ids["alphaearth"]).filterDate(f"{year}-01-01", f"{year + 1}-01-01").filterBounds(region_geom)
    img = col.mosaic().setDefaultProjection(ee.Image(col.first()).projection())
    names = [f"A{i:02d}" for i in range(64)]
    # Rerata per sel dari sampel sistematis 10 x 10 titik (kisi 0,001 derajat, tetangga terdekat), bukan semua piksel
    # 10 m: rerata penuh (~12.300 piksel x 64 band per sel) melampaui memori/kuota GEE project ini (lihat CHANGELOG
    # 2026-10-01; disetujui pengguna). Validasi vs rerata penuh: logs/emb_sample_validation.txt.
    sub = img.select(names).reproject(crs="EPSG:4326", crsTransform=[0.001, 0, -180, 0, -0.001, 90])
    return (sub.reduceResolution(ee.Reducer.mean(), maxPixels=256)
            .rename([f"emb_{n}" for n in names]).toFloat())


def susc_phys_image(ee, ids, y0=2012, y1=2018):
    """FOLLOWUP (kerentanan statis): SRTM dirata-rata ke sel seperti WorldCover (reduceResolution mean); CHIRPS dan
    ERA5-Land di-resample bilinear seperti monthly_image, lalu diringkas untuk y0..y1."""
    srtm = ee.Image(ids["srtm"]).select("elevation")
    elev = srtm.reduceResolution(ee.Reducer.mean(), maxPixels=65535).rename("elev")
    slope = ee.Terrain.slope(srtm).reduceResolution(ee.Reducer.mean(), maxPixels=65535).rename("slope")
    s, e, n = f"{y0}-01-01", f"{y1 + 1}-01-01", y1 - y0 + 1
    ch = (ee.ImageCollection(ids["chirps"]).filterDate(s, e).select("precipitation")
          .map(lambda i: i.resample("bilinear")))
    rain = ch.sum().divide(n).rename("rain_annual")                                   # total / jumlah tahun
    dry = (ch.filter(ee.Filter.calendarRange(8, 10, "month")).map(lambda i: i.lt(1)).sum()
           .divide(n).rename("dry_days_aso"))                                         # hari < 1 mm Agu-Okt per tahun
    era = ee.ImageCollection(ids["era5land_monthly"]).filterDate(s, e)
    sw = (era.filter(ee.Filter.calendarRange(8, 10, "month")).select("volumetric_soil_water_layer_1")
          .map(lambda i: i.resample("bilinear")).mean().rename("swvl1_aso"))
    t2m = era.select("temperature_2m").map(lambda i: i.resample("bilinear")).mean().rename("t2m_annual")
    return ee.Image.cat([elev, slope, rain, dry, sw, t2m]).toFloat()


def susc_keep(ids_grid, cand):
    """FOLLOWUP: sel kandidat + sampel sistematis 10 % sel lain: ((cell * 2654435761) mod 2^32) mod 10 == 0."""
    c = np.asarray(ids_grid).ravel().astype(np.uint64)
    h = ((c * np.uint64(2654435761)) % np.uint64(2 ** 32)) % np.uint64(10) == 0
    return np.unique(c[h | np.isin(c, cand)].astype(np.int64))


# ---------------------------------------------------------------- kerja per potongan
def to_rows(arr, ids_grid, keep):
    mask = np.isin(ids_grid, keep)
    if not mask.any():
        return None
    data = {"cell": ids_grid[mask]}
    for name in arr.dtype.names:
        v = np.asarray(arr[name], dtype=np.float32)[mask]
        v[~np.isfinite(v)] = np.nan                    # piksel ter-mask dari computePixels datang sebagai -inf
        data[name] = v
    return pd.DataFrame(data)


def monthly_long(df, year):
    """Lebar (var_MM) -> panjang (cell, month, var)."""
    parts = []
    for m in range(1, 13):
        sub = {"cell": df["cell"].to_numpy(), "month": f"{year}-{m:02d}"}
        for c in df.columns:
            if c.endswith(f"_{m:02d}"):
                sub[c[:-3]] = df[c].to_numpy()
        if len(sub) > 2:                               # bulan yang tidak diminta (tahun terakhir) dilewati
            parts.append(pd.DataFrame(sub))
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--kind", choices=["monthly", "static", "susc"], required=True)
    ap.add_argument("--parts", choices=["all", "once", "emb", "phys"], default="all",
                    help="static: hanya potongan 'once' (WorldCover/LSIB) atau 'emb'; penggabungan hanya bila 'all'")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    import ee
    ee.Initialize(project=cfg["gee_project"])
    ee.data.setDeadline(cfg.get("gee_deadline_ms", 600_000))       # permintaan yang menggantung tidak menahan proses
    ids = cfg["gee_ids"]
    cand = pd.read_csv(cfg["candidate_cells"])["cell"].to_numpy()
    parts_dir = Path(cfg.get("gee_parts", "data/gee_parts")) / a.kind
    parts_dir.mkdir(parents=True, exist_ok=True)
    # prediktor cukup sampai bulan sebelum panel_end (sama dengan stage_verify); bulan sesudahnya belum tentu ada di GEE
    pred_end = pd.Period(cfg["panel_end"], freq="M") - 1
    y_first, y_last = int(cfg["panel_start"][:4]), pred_end.year

    jobs = []
    for region, bbox in cfg["regions"].items():
        geom = ee.Geometry.Rectangle(bbox, None, False)
        if a.kind == "monthly":
            tpx = int(round(cfg.get("tile_deg_monthly", 2.0) / RES))
            for t in tiles_for(bbox, tpx):
                for y in range(y_first, y_last + 1):
                    last = pred_end.month if y == y_last else 12
                    sfx = f"m{last:02d}" if last < 12 else ""       # potongan tahun parsial tidak dipakai ulang bila panel_end berubah
                    jobs.append((f"{region}_{t[0]}_{t[1]}_{y}{sfx}", t, ("monthly", y, last)))
        elif a.kind == "susc":                                               # FOLLOWUP: kerentanan statis
            tpx = int(round(cfg.get("tile_deg_static", 0.5) / RES))
            for t in tiles_for(bbox, tpx):
                jobs.append((f"{region}_{t[0]}_{t[1]}_once", t, ("once", None)))
                jobs.append((f"{region}_{t[0]}_{t[1]}_phys", t, ("phys", None)))
                for y in cfg.get("followup_emb_years", [2017, 2018]):
                    jobs.append((f"{region}_{t[0]}_{t[1]}_emb{y}", t, ("emb", y, geom)))
        else:
            tpx = int(round(cfg.get("tile_deg_static", 0.5) / RES))
            y0, y1 = cfg["embedding_years"]
            for t in tiles_for(bbox, tpx):
                jobs.append((f"{region}_{t[0]}_{t[1]}_once", t, ("once", None)))
                for y in range(y0, y1 + 1):
                    jobs.append((f"{region}_{t[0]}_{t[1]}_emb{y}", t, ("emb", y, geom)))

    def run(job):
        key, (r, c, ny, nx), spec = job
        out = parts_dir / f"{key}.csv.gz"
        if out.exists():
            return key, "ada"
        grid = cell_ids(r, c, ny, nx)
        keep = susc_keep(grid, cand) if a.kind == "susc" else cand
        if not np.isin(grid, keep).any():
            return key, "lewati (tanpa sel kandidat)"
        if spec[0] == "monthly":
            img = monthly_image(ee, ids, spec[1], spec[2])
        elif spec[0] == "once":
            img = static_once_image(ee, ids)
        elif spec[0] == "phys":
            img = susc_phys_image(ee, ids)
        else:
            img = embedding_image(ee, ids, spec[1], spec[2])
        df = to_rows(compute(ee, img, r, c, ny, nx), grid, keep)
        if df is None:
            return key, "kosong"
        if spec[0] == "monthly":
            df = monthly_long(df, spec[1])
        elif spec[0] == "emb":
            df.insert(1, "year", spec[1])
        df.to_csv(out, index=False, float_format="%.5g")
        return key, f"{len(df)} baris"

    if a.parts != "all":
        jobs = [j for j in jobs if j[2][0] == a.parts]

    failed = []
    with ThreadPoolExecutor(cfg.get("gee_workers", 6)) as pool:
        futs = {pool.submit(run, j): j[0] for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                k, msg = f.result()
                if i % 25 == 0 or "baris" in msg:
                    print(f"[{i}/{len(jobs)}] {k}: {msg}", flush=True)
            except Exception as ex:
                failed.append({"part": futs[f], "error": str(ex)[:500]})
                print(f"[{i}/{len(jobs)}] GAGAL {futs[f]}: {str(ex)[:200]}", flush=True)
    json.dump(failed, open(parts_dir / "failed.json", "w"), indent=1)
    if failed:
        raise SystemExit(f"{len(failed)} potongan gagal; jalankan ulang perintah yang sama untuk melanjutkan "
                         f"(daftar di {parts_dir/'failed.json'})")

    # ---- gabungkan
    if a.parts != "all":
        print(f"potongan '{a.parts}' selesai; penggabungan dilewati (jalankan tanpa --parts untuk menggabung)")
        return
    if a.kind == "monthly":
        df = pd.concat((pd.read_csv(p) for p in sorted(parts_dir.glob("*.csv.gz"))), ignore_index=True)
        df = df.groupby(["cell", "month"], as_index=False).mean()        # sel di tepi dua tile
        df = df[(df["month"] >= cfg["panel_start"]) & (df["month"] <= cfg["panel_end"])]
        df.to_csv(cfg["gee_monthly"], index=False, float_format="%.5g")
        print(f"gee_monthly: {len(df):,} baris -> {cfg['gee_monthly']}")
    elif a.kind == "susc":
        once = pd.concat((pd.read_csv(p) for p in parts_dir.glob("*_once.csv.gz")), ignore_index=True)
        phys = pd.concat((pd.read_csv(p) for p in parts_dir.glob("*_phys.csv.gz")), ignore_index=True)
        emb = pd.concat((pd.read_csv(p) for p in parts_dir.glob("*_emb*.csv.gz")), ignore_index=True)
        emb = (emb.groupby(["cell", "year"], as_index=False).mean()                 # sel di tepi dua tile
               .drop(columns="year").groupby("cell", as_index=False).mean())       # rerata tahun embedding
        st = (once.groupby("cell", as_index=False).mean()
              .merge(phys.groupby("cell", as_index=False).mean(), on="cell", how="outer")
              .merge(emb, on="cell", how="outer"))
        Path(cfg["followup_susc_gee"]).parent.mkdir(parents=True, exist_ok=True)
        st.to_csv(cfg["followup_susc_gee"], index=False, float_format="%.5g")
        print(f"susc (tanpa peat): {len(st):,} sel -> {cfg['followup_susc_gee']}")
    else:
        once = pd.concat((pd.read_csv(p) for p in parts_dir.glob("*_once.csv.gz")), ignore_index=True)
        once = once.groupby("cell", as_index=False).mean()
        emb = pd.concat((pd.read_csv(p) for p in parts_dir.glob("*_emb*.csv.gz")), ignore_index=True)
        emb = emb.groupby(["cell", "year"], as_index=False).mean()
        st = emb.merge(once, on="cell", how="outer")
        st.to_csv(cfg["static_gee"], index=False, float_format="%.5g")
        print(f"static (tanpa peat): {len(st):,} baris -> {cfg['static_gee']}")


if __name__ == "__main__":
    main()
