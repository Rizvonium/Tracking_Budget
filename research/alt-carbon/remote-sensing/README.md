# Independent remote-sensing check of Alt Carbon's Darjeeling deployment (pilot)

Run 1 Nov 2026. All data are open; no login needed.

## Question
Can public satellite data independently confirm Alt Carbon's claims (80,000 acres treated
with basalt, 15–50 % yield gains) without visiting the sites?

## Data
- Sentinel-2 L2A (Element 84 Earth Search, AWS), Oct–Dec 2018–2025, 39 cloud-free overpass dates, 80 m, SCL cloud mask.
- ESA WorldCover 2021 (land cover), SRTM-derived AWS Terrain Tiles (elevation),
  geoBoundaries district boundaries, Natural Earth India-POV country borders.

## Design
Difference-in-differences: Indian side (Darjeeling hills and Terai) vs. Nepalese side (Ilam/Jhapa),
which has the same tea, terrain and monsoon but no basalt. Three strata: hill farmland/tea,
hill tree cover (incl. shaded tea), plains cropland (incl. rice). Before = 2018–23, after = 2024–25.
India−Nepal NDVI differences are computed **per overpass date**. A placebo test uses a fake 2021 rollout.

## Results
1. **Land area:** Darjeeling district totals ~512,000 acres. WorldCover maps ~74,000 acres as cropland;
   most tea is mapped as tree cover. The claimed 80,000 acres (~16 % of the district) is ambitious but
   possible if it covers most tea estates and extends into Jalpaiguri.
2. **First attempt was an artifact.** Per-tile scene selection mixed acquisition dates, producing a false
   "India greener after 2024" signal with tile-shaped seams (`figures/rs_anomaly_map.png`). Kept as a cautionary example.
3. **Date-matched result: no detectable post-rollout greening on the Indian side** in any stratum
   (post−pre India−Nepal NDVI change: hill farmland −0.003, p=0.69; hill trees −0.008, p=0.24;
   plains cropland −0.035, p=0.13). None beat the placebo.

## Interpretation
This does **not** disprove Alt Carbon's carbon removal:
- CO₂ removal by weathering is chemical, so satellites cannot see it.
- Plot locations are not public. Signals from treated fields are diluted across the whole landscape.
- NDVI saturates over dense tea canopy, and plucking or pruning cycles affect it.

It does show that the claimed large yield gains do **not** appear as region-wide canopy greening.
A conclusive test needs plot boundaries and application dates, which Isometric requires in the
project documents.

## Next steps
1. Obtain plot polygons and dates (Isometric PDD, or from Alt Carbon via the collaboration). Then run
   treated-vs-control plot NDVI/NDRE DiD at 10 m, plus Sentinel-1 SAR for the monsoon months.
2. Check bare-soil darkening or SWIR change at spreading time in Terai rice fields (fallow season, Dec–Mar).
3. Use Sentinel-1 VV/VH time series for rice phenology and biomass, since radar sees through cloud.
