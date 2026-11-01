import numpy as np, rasterio, shapefile, math
from rasterio.transform import from_origin, from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
from shapely.geometry import shape
from shapely.ops import transform as stx
from pyproj import Transformer
x0, y1, RES, W, H = np.load("rs/grid.npy") if __import__("os").path.exists("rs/grid.npy") else (None,)*5
tr = Transformer.from_crs(4326, 32645, always_xy=True)
x0, y1 = tr.transform(87.85, 27.15); x1, y0 = tr.transform(88.55, 26.6); RES = 80.0; W = int((x1-x0)/RES); H = int((y1-y0)/RES)
T = from_origin(x0, y1, RES, RES)
# DEM
e = np.load("geo/dem_darj.npy"); bb = (87.5, 26.3, 89.0, 27.3)
def my(lat): return math.log(math.tan(math.pi/4 + math.radians(lat)/2))
Hd = e.shape[0]; lats = np.linspace(bb[3], bb[1], Hd); m0, m1 = my(bb[3]), my(bb[1])
idx = np.clip(np.round([(my(l)-m0)/(m1-m0)*(Hd-1) for l in lats]).astype(int), 0, Hd-1); e = e[idx].astype("f4")
dem = np.zeros((H, W), "f4")
reproject(e, dem, src_transform=from_bounds(*bb, e.shape[1], e.shape[0]), src_crs="EPSG:4326", dst_transform=T, dst_crs="EPSG:32645", resampling=Resampling.bilinear)
# WorldCover mode
wc = np.zeros((H, W), "u1")
for t in ["rs/wc_N24E087.tif", "rs/wc_N27E087.tif"]:
    with rasterio.open(t) as s:
        tmp = np.zeros((H, W), "u1")
        reproject(rasterio.band(s, 1), tmp, dst_transform=T, dst_crs="EPSG:32645", resampling=Resampling.mode)
        wc = np.where(tmp > 0, tmp, wc)
# country
r = shapefile.Reader("geo/ind/ne_10m_admin_0_countries_ind.shp"); f = [x[0] for x in r.fields[1:]]
shapes = []
for sr in r.iterShapeRecords():
    rec = dict(zip(f, sr.record)); n = rec.get("ADMIN")
    if n in ("India", "Nepal", "Bhutan", "Bangladesh"):
        g = stx(lambda x, y, z=None: tr.transform(x, y), shape(sr.shape.__geo_interface__))
        shapes.append((g, {"India": 1, "Nepal": 2}.get(n, 3)))
cty = rasterize(shapes, out_shape=(H, W), transform=T, fill=0, dtype="u1")
np.save("rs/dem.npy", dem); np.save("rs/wc.npy", wc); np.save("rs/cty.npy", cty)
print(W, H, "India px", (cty==1).sum(), "Nepal px", (cty==2).sum(), "dem range", dem.min(), dem.max(), "wc classes", np.unique(wc))
