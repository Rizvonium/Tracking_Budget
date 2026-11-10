"""Render co-registered Kevitsa layers (EPSG:3067, 22 x 18 km, 1100 x 900 px) for the atlas viewer."""
import json, sys, os
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import rasterio

K = sys.argv[1]
OUT = f"{K}/atlas"; os.makedirs(OUT, exist_ok=True)
X0, Y0, X1, Y1 = 488000, 7499000, 510000, 7517000
W, H = 1100, 900
PAPER = (493830, 7504863, 503826, 7511861)  # paper's inversion area (YKJ box converted)


def canvas(transparent=False):
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.axis("off")
    if transparent:
        fig.patch.set_alpha(0); ax.patch.set_alpha(0)
    return fig, ax


def save(fig, name, transparent=False):
    p = f"{OUT}/{name}.png"
    fig.savefig(p, dpi=100, transparent=transparent); plt.close(fig)
    if not transparent:
        Image.open(p).convert("RGB").save(f"{OUT}/{name}.jpg", quality=82, optimize=True); os.remove(p)


def stretch(a, lo=2, hi=98):
    v = np.nanpercentile(a, [lo, hi]); return np.clip((a - v[0]) / (v[1] - v[0]), 0, 1)


ext = [X0, X1, Y0, Y1]
# --- Sentinel-2
s2 = np.load(f"{K}/grid_s2.npz")
rgb = np.dstack([stretch(s2[b]) for b in ["red", "green", "blue"]]) ** 0.8
fig, ax = canvas(); ax.imshow(rgb, extent=ext, origin="upper"); save(fig, "s2_truecolour")
swir = np.dstack([stretch(s2[b]) for b in ["swir22", "nir", "red"]])
fig, ax = canvas(); ax.imshow(swir, extent=ext, origin="upper"); save(fig, "s2_swir")
fe = s2["red"] / np.maximum(s2["blue"], 1)  # ferric iron ratio
fig, ax = canvas(); ax.imshow(stretch(fe, 5, 99), extent=ext, origin="upper", cmap="magma"); save(fig, "s2_iron_ratio")

# --- DEM hillshade + colour
dem = np.load(f"{K}/grid_dem.npy"); res = 20.0
gy, gx = np.gradient(dem, res)
az, alt = np.deg2rad(315), np.deg2rad(40)
slope = np.arctan(np.hypot(gx, gy)); aspect = np.arctan2(-gx, gy)
hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
fig, ax = canvas()
ax.imshow(dem, extent=ext, origin="upper", cmap="terrain", vmin=150, vmax=360)
ax.imshow(hs, extent=ext, origin="upper", cmap="gray", alpha=0.45)
save(fig, "dem")
print("DEM range", np.nanmin(dem), np.nanmax(dem))

# --- GTK raster layers (already rendered by GTK at the same box and size)
for n in ["mag", "mag_shaded", "bouguer", "aem_inphase", "aem_quad", "aem_resistivity", "rad_ternary", "rad_K",
          "rad_Th", "rad_U", "rad_total", "overburden", "bedrock_surface"]:
    im = Image.open(f"{K}/img/{n}.png").convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); bg.alpha_composite(im)
    bg.convert("RGB").save(f"{OUT}/{n}.jpg", quality=82, optimize=True)


# --- vectors
def load(n): return json.load(open(f"{K}/vec/{n}.geojson"))["features"]


def polys(geom):
    t, c = geom["type"], geom["coordinates"]
    if t == "Polygon": return [np.array(c[0])[:, :2]]
    if t == "MultiPolygon": return [np.array(p[0])[:, :2] for p in c]
    return []


CLASSES = [  # (label, colour, keywords)
    ("Ultramafic (peridotite, komatiite)", "#2e7d32", ["peridot", "komat", "dunit", "pyroxenit", "ultramafic"]),
    ("Gabbro and mafic intrusive", "#7cb342", ["gabbro", "granophyre"]),
    ("Mafic volcanic, tuff, basalt", "#26a69a", ["mafic tuff", "mafic volcanic", "basalt", "andesite", "intermediate volcanic"]),
    ("Felsic volcanic", "#ef9a9a", ["felsic"]),
    ("Graphite and sulphide schist", "#424242", ["graphite", "black schist", "sulph"]),
    ("Sedimentary and schist", "#d7b98e", ["schist", "quartzite", "arkose", "silt", "sandstone", "conglomer", "phyllite", "mica"]),
]


def rock_class(name):
    n = (name or "").lower()
    for lab, col, keys in CLASSES:
        if any(k in n for k in keys): return lab, col
    return "Other", "#bdbdbd"


geo = load("bedrock_200k")
fig, ax = canvas()
for f in geo:
    lab, col = rock_class(f["properties"].get("ROCK_NAME_"))
    pc = PolyCollection([np.array(p) for p in polys(f["geometry"])], facecolor=col, edgecolor="#333333", linewidth=0.4)
    ax.add_collection(pc)
save(fig, "geology_200k")
legend = {}
for f in geo:
    lab, col = rock_class(f["properties"].get("ROCK_NAME_")); legend[lab] = col
json.dump(legend, open(f"{OUT}/geology_legend.json", "w"))

# surficial (soil) 1:20k
SOIL = {"turve": "#8d6e63", "moreeni": "#c5b358", "hiekka": "#ffe082", "sora": "#ffcc80", "kallio": "#ef5350",
        "siltti": "#b0bec5", "savi": "#90a4ae", "vesi": "#64b5f6"}
soil_used = {}
fig, ax = canvas()
for f in load("surficial_20k"):
    nm = (f["properties"].get("PINTAMAALAJI") or "").lower()
    col = next((c for k, c in SOIL.items() if k in nm), "#e0e0e0")
    soil_used[f["properties"].get("PINTAMAALAJI")] = col
    ax.add_collection(PolyCollection([np.array(p) for p in polys(f["geometry"])], facecolor=col, edgecolor="none"))
save(fig, "surficial_20k")
json.dump(soil_used, open(f"{OUT}/soil_legend.json", "w"), ensure_ascii=False)

# --- overlays (transparent PNG)
fig, ax = canvas(True)
for f in geo:
    for p in polys(f["geometry"]):
        p = np.array(p); ax.plot(p[:, 0], p[:, 1], color="black", lw=0.7)
save(fig, "ov_contacts", True)


def pts(n):
    out = []
    for f in load(n):
        g = f["geometry"]
        if g and g["type"] == "Point": out.append(g["coordinates"][:2])
    return np.array(out) if out else np.zeros((0, 2))


for n, col, s, mk in [("drill_holes", "#ffeb3b", 10, "o"), ("core_archive", "#ff6f00", 18, "o"),
                      ("outcrops", "#e040fb", 8, "^"), ("petrophysics_samples", "#00e5ff", 60, "D")]:
    P = pts(n); fig, ax = canvas(True)
    if len(P): ax.scatter(P[:, 0], P[:, 1], s=s, c=col, marker=mk, edgecolors="black", linewidths=0.5)
    save(fig, f"ov_{n}", True); print(n, len(P))

fig, ax = canvas(True)
x0, y0, x1, y1 = PAPER
ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="white", lw=4)
ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="#d50000", lw=2, ls="--")
M = pts("metal_mines")
for x, y in M:
    ax.scatter([x], [y], s=380, marker="*", c="#d50000", edgecolors="white", linewidths=1.5)
save(fig, "ov_paper_area", True)

# --- EMAG2 regional context (separate extent)
with rasterio.open(f"{K}/raw/emag2_regional.tif") as s:
    a = s.read(1).astype(float); b = s.bounds
a[a < -1e5] = np.nan
fig = plt.figure(figsize=(8, 5.6), dpi=110); ax = fig.add_axes([0.08, 0.1, 0.8, 0.82])
im = ax.imshow(a, extent=[b.left, b.right, b.bottom, b.top], origin="upper", cmap="RdBu_r", vmin=-400, vmax=400)
ax.plot([X0, X1, X1, X0, X0], [Y0, Y0, Y1, Y1, Y0], color="black", lw=2)
ax.set_xlabel("Easting, EPSG:3067 (m)"); ax.set_ylabel("Northing (m)")
ax.set_title("EMAG2v3 regional magnetic anomaly (4 km grid); black box = atlas area", fontsize=10)
fig.colorbar(im, ax=ax, fraction=0.04, label="nT")
fig.savefig(f"{OUT}/emag2_regional.png", dpi=110); plt.close(fig)
Image.open(f"{OUT}/emag2_regional.png").convert("RGB").save(f"{OUT}/emag2_regional.jpg", quality=85); os.remove(f"{OUT}/emag2_regional.png")
print("ok")
