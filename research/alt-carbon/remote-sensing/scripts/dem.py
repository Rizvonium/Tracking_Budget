# Fetch AWS terrarium tiles for a bbox and mosaic into an elevation array
import math, os, sys, urllib.request, numpy as np
from PIL import Image
def deg2num(lat, lon, z):
    n = 2**z; x = (lon+180)/360*n
    y = (1 - math.log(math.tan(math.radians(lat)) + 1/math.cos(math.radians(lat)))/math.pi)/2*n
    return x, y
def fetch(bbox, z, cache):
    lon0, lat0, lon1, lat1 = bbox
    x0, y0 = deg2num(lat1, lon0, z); x1, y1 = deg2num(lat0, lon1, z)
    xs, ys = range(int(x0), int(x1)+1), range(int(y0), int(y1)+1)
    rows = []
    for y in ys:
        row = []
        for x in xs:
            p = os.path.join(cache, f"{z}_{x}_{y}.png")
            if not os.path.exists(p):
                urllib.request.urlretrieve(f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png", p)
            a = np.asarray(Image.open(p).convert('RGB')).astype(float)
            row.append(a[...,0]*256 + a[...,1] + a[...,2]/256 - 32768)
        rows.append(np.hstack(row))
    elev = np.vstack(rows)
    # pixel extents in tile coords
    n = 2**z
    def num2deg(x, y):
        lon = x/n*360-180; lat = math.degrees(math.atan(math.sinh(math.pi*(1-2*y/n)))); return lat, lon
    top, left = num2deg(xs[0], ys[0]); bottom, right = num2deg(xs[-1]+1, ys[-1]+1)
    # crop to bbox
    H, W = elev.shape
    def px(lat, lon):
        fx, fy = deg2num(lat, lon, z); return int((fx-xs[0])*256), int((fy-ys[0])*256)
    cx0, cy0 = px(lat1, lon0); cx1, cy1 = px(lat0, lon1)
    elev = elev[cy0:cy1, cx0:cx1]
    return elev  # web-mercator rows; small bbox so treat as near-linear in lat
if __name__ == "__main__":
    bbox = tuple(map(float, sys.argv[1:5])); z = int(sys.argv[5]); out = sys.argv[6]
    e = fetch(bbox, z, os.path.join(os.path.dirname(out), 'tiles'))
    np.save(out, e); print(e.shape, e.min(), e.max())
