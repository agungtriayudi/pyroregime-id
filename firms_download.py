"""Download NASA FIRMS fire detections (VIIRS 375 m / MODIS 1 km) for the Indonesian regions.

    export FIRMS_MAP_KEY=xxxx          # free: https://firms.modaps.eosdis.nasa.gov/api/map_key/
    python firms_download.py --source VIIRS_SNPP_SP --start 2012-01-20 --end 2025-12-31 --region all
    python firms_download.py --source VIIRS_SNPP_NRT --start 2026-01-01 --end 2026-09-30 --region all

Notes:
- The endpoint format follows the FIRMS "area" API documentation:
  /api/area/csv/{MAP_KEY}/{SOURCE}/{west,south,east,north}/{DAY_RANGE}/{DATE}
  The DAY_RANGE limit and the 10-minute quota can change; check the FIRMS documentation if a response contains an error.
- The SP (standard processing) archive lags by a few months; the latest months use the *_NRT sources.
- Alternative for large archives: the "Archive Download" form on the FIRMS website (results are sent by e-mail).
"""
import argparse
import os
import time
from datetime import date, timedelta
from pathlib import Path

import requests

API = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{bbox}/{days}/{date}"
# rough boxes (west, south, east, north); clipped to administrative and peat boundaries later in the pipeline
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
    """Are the transactions of the last 10 minutes close to the limit? (FIRMS sometimes answers 'Invalid MAP_KEY' when the quota is used up.)"""
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
        except requests.RequestException as ex:                  # transient network error; message without the key
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
        if quota:                                                  # transaction quota: wait until the 10-minute window has moved on
            waits += 1
            if waits > quota_waits:
                raise RuntimeError("FIRMS quota exceeded repeatedly; try again later")
            time.sleep(120)
            continue
        if r.status_code >= 500:                                   # server busy
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
        raise SystemExit("Set FIRMS_MAP_KEY first.")
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
            print(f"{name} {d} +{days}d: {max(0, text.count(chr(10)) - 1)} detections")
            time.sleep(1.0)


if __name__ == "__main__":
    main()
