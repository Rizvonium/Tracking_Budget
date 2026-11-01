import os, sys, numpy as np, rasterio, warnings
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from pystac_client import Client
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings("ignore")
os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES", GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2",
                  CURL_CA_BUNDLE=os.environ.get("CURL_CA_BUNDLE", "/root/.ccr/ca-bundle.crt"), GDAL_HTTP_MULTIRANGE="YES", VSI_CACHE="TRUE")
OUT = sys.argv[1]; years = [int(y) for y in sys.argv[2].split(",")]; PER_TILE = int(sys.argv[3]) if len(sys.argv) > 3 else 6
from pyproj import Transformer
tr = Transformer.from_crs(4326, 32645, always_xy=True)
x0, y1 = tr.transform(87.85, 27.15); x1, y0 = tr.transform(88.55, 26.6)
RES = 80.0; W = int((x1 - x0) / RES); H = int((y1 - y0) / RES)
T = from_origin(x0, y1, RES, RES)
cat = Client.open("https://earth-search.aws.element84.com/v1")
def read(href, resamp):
    with rasterio.Env():
        with rasterio.open(href, OVERVIEW_LEVEL=2 if "SCL" not in href else 1) as src:
            with WarpedVRT(src, crs="EPSG:32645", transform=T, width=W, height=H, resampling=resamp) as v:
                return v.read(1)
def scene(it):
    try:
        r = read(it.assets["red"].href, Resampling.average).astype("f4"); n = read(it.assets["nir"].href, Resampling.average).astype("f4")
        s = read(it.assets["scl"].href, Resampling.nearest)
        ok = np.isin(s, [4, 5]) & (r > 0) & (n > 0)
        ndvi = np.where(ok, (n - r) / (n + r), np.nan)
        return ndvi
    except Exception as e:
        print("  fail", it.id, str(e)[:80]); return None
for yr in years:
    items = list(cat.search(collections=["sentinel-2-l2a"], bbox=[87.85, 26.6, 88.55, 27.15], datetime=f"{yr}-10-15/{yr}-12-31",
                            query={"eo:cloud_cover": {"lt": 25}}, max_items=300).items())
    by = {}
    for it in sorted(items, key=lambda i: i.properties["eo:cloud_cover"]):
        k = it.properties.get("grid:code"); by.setdefault(k, [])
        if len(by[k]) < PER_TILE: by[k].append(it)
    sel = [i for v in by.values() for i in v]
    with ThreadPoolExecutor(8) as ex: arrs = [a for a in ex.map(scene, sel) if a is not None]
    stack = np.stack(arrs); med = np.nanmedian(stack, axis=0); cnt = np.sum(~np.isnan(stack), axis=0)
    np.save(f"{OUT}/ndvi_{yr}.npy", med.astype("f4")); np.save(f"{OUT}/cnt_{yr}.npy", cnt.astype("u1"))
    print(yr, "scenes", len(arrs), "valid px %.1f%%" % (100 * np.mean(cnt > 0)), "median NDVI %.3f" % np.nanmedian(med), flush=True)
np.save(f"{OUT}/grid.npy", np.array([x0, y1, RES, W, H]))
