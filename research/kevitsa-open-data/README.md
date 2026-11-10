# Kevitsa open geoscience data (Sodankylä, Finland)

Free, open layers for a 22 km x 18 km box around the Kevitsa Ni-Cu-PGE mine.
Box in EPSG:3067 (ETRS-TM35FIN): x 488000-510000, y 7499000-7517000.

| Script | What it downloads |
|---|---|
| `fetch_vectors.py OUT` | GTK WFS/REST: bedrock 1:200k, Quaternary deposits 1:20k, drill holes, core archive, outcrops, petrophysics sample sites, rock and till geochemistry, mineral occurrences, mines (GeoJSON) |
| `fetch_rasters.py K` | Copernicus DEM GLO-30 grid (needs the two DEM tiles in `K/dem/`), EMAG2v3 regional magnetic GeoTIFF (NOAA), Sentinel-2 L2A bands |
| `fetch_s2.py K` | Sentinel-2 L2A mosaic of tiles 35WMR + 35WNR, 14 Jul 2025 |
| `atlas.py K` | Renders all layers at 1100 x 900 px (20 m) in the same box |

GTK geophysics images (magnetics, Bouguer gravity, AEM, radiometrics) come from the
MDaE/Geophysics MapServer `export` endpoint (layers 58, 64, 65, 69-71, 73-77).
They are coloured images, not values. Numeric line data are in GTK Hakku (registration).

Licences: GTK data CC BY 4.0; Copernicus DEM and Sentinel-2 under the Copernicus licence; EMAG2v3 public domain.
