import os, sys, json, numpy as np, rasterio, warnings
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from pystac_client import Client
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2")
x0, y1, RES, W, H = np.load("rs/grid.npy"); W, H = int(W), int(H); T = from_origin(x0, y1, RES, RES)
dem, wc, cty = np.load("rs/dem.npy"), np.load("rs/wc.npy"), np.load("rs/cty.npy")
hills = (dem > 500) & (dem < 2300); plains = dem < 300
STR = {"hill_farm": hills & np.isin(wc, [20, 30, 40]), "hill_tree": hills & (wc == 10), "plains_crop": plains & (wc == 40)}
cat = Client.open("https://earth-search.aws.element84.com/v1")
def read(href, resamp, lvl):
    with rasterio.open(href, OVERVIEW_LEVEL=lvl) as src:
        with WarpedVRT(src, crs="EPSG:32645", transform=T, width=W, height=H, resampling=resamp, nodata=0) as v:
            return v.read(1)
def tile(it):
    try:
        r = read(it.assets["red"].href, Resampling.average, 2).astype("f4"); n = read(it.assets["nir"].href, Resampling.average, 2).astype("f4")
        s = read(it.assets["scl"].href, Resampling.nearest, 1)
        ok = np.isin(s, [4, 5]) & (r > 0) & (n > 0)
        return np.where(ok, (n - r) / (n + r), np.nan)
    except Exception as e:
        print("fail", it.id, str(e)[:60]); return None
rows = []
for yr in range(2018, 2026):
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=[87.85, 26.6, 88.55, 27.15], datetime=f"{yr}-10-15/{yr}-12-31", max_items=400).items())
    byd = {}
    for it in items: byd.setdefault(it.datetime.date().isoformat(), []).append(it)
    good = [(np.mean([i.properties["eo:cloud_cover"] for i in v]), d, v) for d, v in byd.items() if len(v) >= 4 and max(i.properties["eo:cloud_cover"] for i in v) < 30]
    good = sorted(good)[:6]
    stack = []
    for _, d, v in sorted(good, key=lambda g: g[1]):
        with ThreadPoolExecutor(4) as ex: arrs = [a for a in ex.map(tile, v) if a is not None]
        mos = np.nanmax(np.stack(arrs), axis=0)  # same-date tiles: take max over overlap (identical acquisition)
        stack.append(mos)
        rec = {"year": yr, "date": d}
        for k, m in STR.items():
            for c, cn in [(1, "IN"), (2, "NP")]:
                mm = m & (cty == c); vals = mos[mm]; rec[f"{k}_{cn}"] = float(np.nanmean(vals)) if np.mean(~np.isnan(vals)) > 0.5 else None
        rows.append(rec); print(rec, flush=True)
    np.save(f"rs/dated_ndvi_{yr}.npy", np.nanmedian(np.stack(stack), axis=0).astype("f4"))
json.dump(rows, open("rs/dated_rows.json", "w"))
