import json, sys, urllib.request
import numpy as np, rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
K=sys.argv[1]; BOX=(488000,7499000,510000,7517000); RES=20.0
W,H=int((BOX[2]-BOX[0])/RES),int((BOX[3]-BOX[1])/RES); T=from_origin(BOX[0],BOX[3],RES,RES)
ids=["S2C_35WMR_20250714_0_L2A","S2C_35WNR_20250714_0_L2A"]
bands={}
for b in ["blue","green","red","nir","swir16","swir22"]:
    out=np.full((H,W),np.nan,dtype="float32")
    for i in ids:
        it=json.load(urllib.request.urlopen(f"https://earth-search.aws.element84.com/v1/collections/sentinel-2-l2a/items/{i}"))
        g=np.full((H,W),np.nan,dtype="float32")
        with rasterio.open(it["assets"][b]["href"]) as s:
            reproject(rasterio.band(s,1),g,src_transform=s.transform,src_crs=s.crs,src_nodata=0,dst_transform=T,dst_crs="EPSG:3067",resampling=Resampling.bilinear,dst_nodata=np.nan)
        out=np.where(np.isnan(out),g,out)
    bands[b]=out; print(b, np.isnan(out).mean())
np.savez_compressed(f"{K}/grid_s2.npz",**bands,scene="2025-07-14 tiles 35WMR+35WNR")
