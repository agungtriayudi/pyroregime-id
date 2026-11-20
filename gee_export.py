"""Export GEE layers on the same 0.01-degree grid as grid_features.py (pixel = cell).

    pip install earthengine-api && earthengine authenticate
    python gee_export.py --project YOUR_GCP_PROJECT --region sumatra --monthly 2012-01 2026-09 --folder pyroregime
    python gee_export.py --project YOUR_GCP_PROJECT --region sumatra --annual 2017 2025 --peat-asset users/xxx/peat_map

Monthly (bands): burned_frac (MCD64A1), modis_fire_days (MOD14A1+MYD14A1, FireMask>=7), precip_mm, dry_days (CHIRPS),
                swvl1, t2m, d2m (ERA5-Land).
Annual (bands): emb_A00..emb_A63 (AlphaEarth mean per cell), wc_* (WorldCover fractions), in_indonesia (LSIB),
                peat_frac (if an asset exists).
FWI is not in the GEE catalogue: download the CEMS fire danger reanalysis (Vitolo et al. 2020) from Copernicus EWDS/CDS,
aggregate it to months and save it as CSV (cell, month, fwi) for --fwi in run_all.py.
The GeoTIFF results go to Google Drive; convert them to cell tables with raster_to_table.py.
Fallback path, NOT run for the paper (no GEE access in the development environment); collection IDs as in the
GEE catalogue of 2025-2026.
"""
import argparse

import ee

from firms_download import REGIONS

CRS, TRANSFORM = "EPSG:4326", [0.01, 0, -180, 0, -0.01, 90]
WC = {"tree": 10, "shrub": 20, "grass": 30, "crop": 40, "built": 50, "bare": 60, "water": 80, "wetland": 90}


def to_grid(img, maxpx=65535):
    return img.reduceResolution(ee.Reducer.mean(), maxPixels=maxpx).reproject(crs=CRS, crsTransform=TRANSFORM)


def monthly(y, m, aoi):
    s = ee.Date.fromYMD(y, m, 1); e = s.advance(1, "month")
    burned = (ee.ImageCollection("MODIS/061/MCD64A1").filterDate(s, e).select("BurnDate").max()
              .gt(0).unmask(0).rename("burned_frac"))
    fire = (ee.ImageCollection("MODIS/061/MOD14A1").merge(ee.ImageCollection("MODIS/061/MYD14A1"))
            .filterDate(s, e).map(lambda i: i.select("FireMask").gte(7)).sum().rename("modis_fire_days"))
    ch = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY").filterDate(s, e).select("precipitation")
    precip, dry = ch.sum().rename("precip_mm"), ch.map(lambda i: i.lt(1)).sum().rename("dry_days")
    era = (ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterDate(s, e).first()
           .select(["volumetric_soil_water_layer_1", "temperature_2m", "dewpoint_temperature_2m"], ["swvl1", "t2m", "d2m"]))
    # fire_days summed at the native 1 km resolution, then aligned; the others averaged to the cell
    stack = ee.Image.cat([to_grid(burned), fire.reproject(crs=CRS, crsTransform=TRANSFORM),
                          precip.resample("bilinear").reproject(crs=CRS, crsTransform=TRANSFORM),
                          dry.reproject(crs=CRS, crsTransform=TRANSFORM),
                          era.resample("bilinear").reproject(crs=CRS, crsTransform=TRANSFORM)])
    return stack.toFloat().clip(aoi)


def annual(y, aoi, peat_asset=None):
    col = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").filterDate(f"{y}-01-01", f"{y + 1}-01-01").filterBounds(aoi)
    emb = col.mosaic().setDefaultProjection(col.first().projection())       # 10 m UTM projection of the first tile
    emb = to_grid(emb).rename([f"emb_A{i:02d}" for i in range(64)])
    wc = ee.Image("ESA/WorldCover/v200/2021")
    wc = ee.Image.cat([to_grid(wc.eq(v).rename(f"wc_{k}")) for k, v in WC.items()])
    lsib = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq("country_na", "Indonesia"))
    inid = to_grid(ee.Image.constant(0).paint(lsib, 1).rename("in_indonesia"))
    layers = [emb, wc, inid]
    if peat_asset:
        layers.append(to_grid(ee.Image(peat_asset).gt(0).unmask(0).rename("peat_frac")))
    return ee.Image.cat(layers).toFloat().clip(aoi)


def export(img, name, aoi, folder):
    t = ee.batch.Export.image.toDrive(image=img, description=name, folder=folder, fileNamePrefix=name,
                                      region=aoi, crs=CRS, crsTransform=TRANSFORM, maxPixels=1e11)
    t.start()
    print("task:", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--region", choices=list(REGIONS), required=True)
    ap.add_argument("--monthly", nargs=2, metavar=("YYYY-MM", "YYYY-MM"))
    ap.add_argument("--annual", nargs=2, type=int, metavar=("Y0", "Y1"))
    ap.add_argument("--peat-asset", default=None)
    ap.add_argument("--folder", default="pyroregime")
    a = ap.parse_args()
    ee.Initialize(project=a.project)
    aoi = ee.Geometry.Rectangle(list(REGIONS[a.region]), None, False)
    if a.monthly:
        (y0, m0), (y1, m1) = [map(int, v.split("-")) for v in a.monthly]
        y, m = y0, m0
        while (y, m) <= (y1, m1):
            export(monthly(y, m, aoi), f"m_{a.region}_{y}{m:02d}", aoi, a.folder)
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    if a.annual:
        for y in range(a.annual[0], a.annual[1] + 1):
            export(annual(y, aoi, a.peat_asset), f"a_{a.region}_{y}", aoi, a.folder)


if __name__ == "__main__":
    main()
