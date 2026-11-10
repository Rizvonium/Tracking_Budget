"""EMAG2v3 regional grid (NOAA) + Sentinel-2 L2A bands (Earth Search) + DEM mosaic for the Kevitsa box."""
import json, sys, urllib.parse, urllib.request
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.transform import from_origin

K = sys.argv[1]
BOX = (488000, 7499000, 510000, 7517000)  # EPSG:3067
RES = 20.0
W, H = int((BOX[2] - BOX[0]) / RES), int((BOX[3] - BOX[1]) / RES)
DST_T = from_origin(BOX[0], BOX[3], RES, RES)


def to_grid(src_path, band=1, resampling=Resampling.bilinear, src_window=None):
    out = np.full((H, W), np.nan, dtype="float32")
    with rasterio.open(src_path) as s:
        reproject(source=rasterio.band(s, band), destination=out, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=DST_T, dst_crs="EPSG:3067", resampling=resampling, dst_nodata=np.nan)
    return out


# --- DEM (Copernicus GLO-30), mosaic of two tiles
dem = np.full((H, W), np.nan, dtype="float32")
for t in ["N67_00_E026_00", "N67_00_E027_00"]:
    g = to_grid(f"{K}/dem/cop_{t}.tif")
    dem = np.where(np.isnan(dem), g, dem)
np.save(f"{K}/grid_dem.npy", dem)
print("DEM", np.nanmin(dem), np.nanmax(dem))

# --- EMAG2v3 regional (4 km grid) as float tiff, wider box for context
reg = (25.5, 67.2, 28.5, 68.2)
q = dict(bbox=",".join(map(str, reg)), bboxSR=4326, imageSR=3067, size="300,200", format="tiff",
         pixelType="F32", interpolation="RSP_BilinearInterpolation", f="image")
url = "https://gis.ngdc.noaa.gov/arcgis/rest/services/EMAG2v3/ImageServer/exportImage?" + urllib.parse.urlencode(q)
try:
    urllib.request.urlretrieve(url, f"{K}/raw/emag2_regional.tif")
    with rasterio.open(f"{K}/raw/emag2_regional.tif") as s:
        a = s.read(1); print("EMAG2", s.crs, s.bounds, np.nanmin(a), np.nanmax(a))
except Exception as e:
    print("EMAG2 failed", e)

# --- Sentinel-2 L2A: least-cloudy summer scene
lon0, lat0, lon1, lat1 = transform_bounds("EPSG:3067", "EPSG:4326", *BOX)
body = {"collections": ["sentinel-2-l2a"], "bbox": [lon0, lat0, lon1, lat1],
        "datetime": "2022-06-15T00:00:00Z/2025-08-31T23:59:59Z", "limit": 100,
        "query": {"eo:cloud_cover": {"lt": 3}}}
req = urllib.request.Request("https://earth-search.aws.element84.com/v1/search", data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
items = json.load(urllib.request.urlopen(req, timeout=90))["features"]
items = [i for i in items if i["properties"]["datetime"][5:7] in ("06", "07", "08")]
items.sort(key=lambda i: i["properties"]["eo:cloud_cover"])
print("S2 candidates", len(items))
from shapely.geometry import shape, box as sbox  # noqa
aoi = sbox(lon0, lat0, lon1, lat1)
items = [i for i in items if shape(i["geometry"]).contains(aoi)] or items
it = items[0]
print("S2 scene", it["id"], it["properties"]["datetime"], it["properties"]["eo:cloud_cover"])
bands = {}
for b in ["blue", "green", "red", "nir", "swir16", "swir22"]:
    bands[b] = to_grid(it["assets"][b]["href"])
np.savez_compressed(f"{K}/grid_s2.npz", **bands, scene=it["id"])
print("done")
