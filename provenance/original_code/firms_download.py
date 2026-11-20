"""Unduh deteksi api NASA FIRMS (VIIRS 375 m / MODIS 1 km) untuk wilayah Indonesia.

    export FIRMS_MAP_KEY=xxxx          # gratis: https://firms.modaps.eosdis.nasa.gov/api/map_key/
    python firms_download.py --source VIIRS_SNPP_SP --start 2012-01-20 --end 2025-12-31 --region all
    python firms_download.py --source VIIRS_SNPP_NRT --start 2026-01-01 --end 2026-09-30 --region all

Catatan:
- Format endpoint mengikuti dokumentasi FIRMS "area" API:
  /api/area/csv/{MAP_KEY}/{SOURCE}/{west,south,east,north}/{DAY_RANGE}/{DATE}
  Batas DAY_RANGE dan kuota per 10 menit bisa berubah; cek dokumentasi FIRMS bila respons berisi pesan error.
- Arsip SP (standard processing) tertinggal beberapa bulan; bulan terbaru pakai sumber *_NRT.
- Alternatif untuk arsip besar: formulir "Archive Download" di situs FIRMS (hasil dikirim via email).
"""
import argparse
import os
import time
from datetime import date, timedelta
from pathlib import Path

import requests

API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{bbox}/{days}/{date}"
# kotak kasar (west, south, east, north); dipotong ke batas administrasi/peta gambut saat grid_features
REGIONS = {
    "sumatra": (95.0, -6.2, 106.3, 6.0),
    "kalimantan": (108.5, -4.3, 119.3, 4.5),
    "papua_selatan": (137.5, -9.2, 141.1, -5.5),
}


def daterange(start: date, end: date, step: int):
    d = start
    while d <= end:
        yield d, min(step, (end - d).days + 1)
        d += timedelta(days=step)


STATUS = "https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY={key}"


def quota_busy(key):
    """Transaksi 10 menit terakhir mendekati batas? (FIRMS kadang membalas 'Invalid MAP_KEY' saat kuota habis.)"""
    try:
        s = requests.get(STATUS.format(key=key), timeout=60).json()
        return int(s["current_transactions"]) >= 0.9 * int(s["transaction_limit"])
    except Exception:
        return False


def fetch(key, source, bbox, days, d, retries=4, quota_waits=15):
    url = API.format(key=key, source=source, bbox=",".join(map(str, bbox)), days=days, date=d.isoformat())
    attempt = waits = 0
    while True:
        try:
            r = requests.get(url, timeout=120)
        except requests.RequestException as ex:                  # galat jaringan sementara; pesan tanpa kunci
            attempt += 1
            if attempt >= retries:
                raise RuntimeError(f"FIRMS {source} {d}: {type(ex).__name__}: {str(ex).replace(key, '<KEY>')[:200]}") from None
            time.sleep(15 * attempt)
            continue
        text = r.text
        if r.status_code == 200 and text.startswith("latitude"):
            return text
        quota = "exceeding" in text.lower() or r.status_code == 429
        if not quota and "invalid map_key" in text.lower():
            quota = quota_busy(key)
        if quota:                                                  # kuota transaksi: tunggu jendela 10 menit bergeser
            waits += 1
            if waits > quota_waits:
                raise RuntimeError("kuota FIRMS terus terlampaui; coba lagi nanti")
            time.sleep(120)
            continue
        if r.status_code >= 500:                                   # server sibuk
            attempt += 1
            if attempt < retries:
                time.sleep(15 * attempt)
                continue
        raise RuntimeError(f"FIRMS {source} {d} {r.status_code}: {text[:200].replace(key, '<KEY>')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="VIIRS_SNPP_SP",
                    help="VIIRS_SNPP_SP | VIIRS_NOAA20_SP | VIIRS_SNPP_NRT | VIIRS_NOAA20_NRT | MODIS_SP | MODIS_NRT")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--region", default="all", choices=["all", *REGIONS])
    ap.add_argument("--day-range", type=int, default=5)
    ap.add_argument("--out", default="data/raw/firms")
    args = ap.parse_args()

    key = os.environ.get("FIRMS_MAP_KEY")
    if not key:
        raise SystemExit("Set FIRMS_MAP_KEY terlebih dahulu.")
    regions = REGIONS if args.region == "all" else {args.region: REGIONS[args.region]}
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    for name, bbox in regions.items():
        out = Path(args.out) / args.source / name
        out.mkdir(parents=True, exist_ok=True)
        for d, days in daterange(start, end, args.day_range):
            f = out / f"{d.isoformat()}_{days}d.csv"
            if f.exists():
                continue
            text = fetch(key, args.source, bbox, days, d)
            f.write_text(text)
            print(f"{name} {d} +{days}d: {max(0, text.count(chr(10)) - 1)} deteksi")
            time.sleep(1.0)


if __name__ == "__main__":
    main()
