"""Fetch GTK vector layers (open data, CC BY 4.0) in the Kevitsa box as GeoJSON (EPSG:3067)."""
import json, sys, time, urllib.parse, urllib.request

OUT = sys.argv[1]
R = "https://gtkdata.gtk.fi/arcgis/rest/services"
BBOX = (488000, 7499000, 510000, 7517000)
LAYERS = {
    "bedrock_200k": "GTKWFS/MapServer/13",
    "drill_holes": "GTKWFS/MapServer/7",
    "till_geochem": "GTKWFS/MapServer/2",
    "rock_geochem": "GTKWFS/MapServer/3",
    "surficial_20k": "GTKWFS/MapServer/10",
    "mineral_occurrences": "Rajapinnat/GTK_Kalliopera_WFS/MapServer/1",
    "metal_mines": "Rajapinnat/GTK_Kalliopera_WFS/MapServer/13",
    "petrophysics_samples": "Rajapinnat/GTK_Kalliopera_WFS/MapServer/20",
    "outcrops": "Rajapinnat/GTK_Kalliopera_WFS/MapServer/16",
    "structures_100k": "Rajapinnat/GTK_Kalliopera_WFS/MapServer/39",
    "core_archive": "Rajapinnat/GTK_Kalliopera_WFS/MapServer/25",
}


def get(url):
    for k in range(4):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return json.load(r)
        except Exception as e:  # noqa
            time.sleep(2 ** k)
            err = e
    raise err


for name, path in LAYERS.items():
    feats, off = [], 0
    while True:
        q = dict(geometry=",".join(map(str, BBOX)), geometryType="esriGeometryEnvelope", inSR=3067,
                 spatialRel="esriSpatialRelIntersects", outFields="*", outSR=3067, returnGeometry="true",
                 resultOffset=off, resultRecordCount=1000, f="geojson")
        try:
            d = get(f"{R}/{path}/query?" + urllib.parse.urlencode(q))
        except Exception as e:
            print(name, "ERROR", e); break
        if "error" in d:
            print(name, "ERROR", d["error"].get("message")); break
        fs = d.get("features", [])
        feats += fs
        if len(fs) < 1000 and not d.get("exceededTransferLimit") and not d.get("properties", {}).get("exceededTransferLimit"):
            break
        off += len(fs)
        if not fs:
            break
    json.dump({"type": "FeatureCollection", "crs": "EPSG:3067", "features": feats}, open(f"{OUT}/{name}.geojson", "w"))
    keys = list(feats[0]["properties"].keys())[:14] if feats else []
    print(f"{name}: {len(feats)} features; fields: {keys}")
