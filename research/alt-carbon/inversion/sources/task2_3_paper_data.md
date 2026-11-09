# Task 2/3: Paper identification and Finland dataset inventory

Compiled 2026-11-09 from web searches only. Anything not seen in a search result or fetched page is tagged **UNVERIFIED**. No DOIs were invented: every DOI or arXiv ID below appeared in a result.

---

## JOB 1: Candidate papers

Alt Carbon's description: a 3D magnetic (possibly gravity plus magnetic) inversion using GTK Hakku data. It has possible overfitting, attributes negative susceptibility to remanence, and does something notable for memory and compute efficiency. Probably 2022 to 2026.

**Bottom line:** no paper found matches every element. The strongest lead is #1. It has Indian authors (consistent with Alt Carbon being Indian), a Geological Survey of Finland co-author, and a memory and GPU efficiency focus. But it is a **gravity** paper, and its v1 field data are Indian, not Hakku. A magnetic companion paper or the revised v2 might be the actual paper, but that is **UNVERIFIED**. Ask the student for the title or arXiv ID.

### Candidate 1: Julia framework for GPU gravity inversion (match: MEDIUM, the best lead)
- **Citation:** Nimatullah, Pankaj K. Mishra, Jochen Kamm, Anand Singh. "Backend-agnostic Julia framework for 3D modeling and inversion of gravity data." arXiv:2602.03857 [physics.geo-ph]. v1 submitted 23 Jan 2026; v2 revised 1 Sep 2026. https://arxiv.org/abs/2602.03857
- **Affiliations:** Dept. of Earth Sciences, IIT Bombay, and the Geological Survey of Finland (GTK). Mishra and Kamm are GTK (Mishra: GTK Senior Scientist, https://gtk.fi/en/experts/mishra-pankaj).
- **Method:** depth-weighted, regularized 3D prism (voxel) gravity inversion.
- **Efficiency approach:**
  - A **data-space inversion formulation**, which shrinks the system to N_data x N_data and lowers memory.
  - A backend-agnostic kernel abstraction (KernelAbstractions-style), so one code base runs on multicore CPUs and GPUs (NVIDIA CUDA).
  - Tested up to about 3.3 million prisms and 178,797 observations. Beyond about 3.3 million prisms, runs hit a GPU memory error on CSC's Mahti.
- **Funding and compute:** Research Council of Finland (HPC 2023 project, grant no. 35926), with CSC LUMI and IIT Bombay compute.
- **Code:** https://github.com/naimat04/GravityInversionGPU.jl and Zenodo https://doi.org/10.5281/zenodo.18339626
- **Region and data (v1 HTML):** Tangarparha, Odisha (chromite, after Mohanty et al. 2011) and Bharatpur District, Rajasthan (Great Boundary Fault). **No Finnish or Hakku data and no magnetics in v1.**
- **Overfitting angle:** v1 does not report the regularization parameter, the stopping rule, or any chi-square or misfit values. That gap is exactly what would prompt an "overfitting" critique.
- **Remanence and negative susceptibility:** not in v1.
- **Why it still ranks first:** the GTK co-authors and the HPC 2023 Finland grant point to a Finnish application. Whether v2 (Sep 2026) or a sibling magnetic paper uses Hakku magnetics is **UNVERIFIED**. The first author's GitHub has 12 repos and no magnetic-inversion repo is visible.

### Candidate 2: Akanvaara layered intrusion magnetic inversion (match: MEDIUM)
- **Citation:** EAGE 2026 extended abstract, DOI 10.3997/2214-4609.202620160 (EarthDoc). The title is roughly "integrated geophysical workflow combining magnetic inversion, geological constraints and AEM-IP, Akanvaara layered intrusion, Finland."
- **Authors:** **UNVERIFIED** (the page returned 403). A related NSG 2025 paper on Akanvaara AEM is by Panepinto, Lucchelli, Chen, Dauti, Fernandez, Viezzoli and Fiandaca, so this may be the same group (**UNVERIFIED**).
- **Region:** Akanvaara Cr–V–PGE layered intrusion, Finnish Lapland.
- **Method and efficiency:** regularized magnetic susceptibility inversion with **SVD of the sensitivity matrix** to stabilize the solution, reduce artefacts and improve deeper recovery. The susceptibility model is then integrated with AEM-IP chargeability.
- **Match:** Finnish, magnetic, and the authors use an SVD or truncation trick. Use of Hakku data and any remanence or negative-susceptibility discussion are **UNVERIFIED**.
- A companion AEM-IP paper: DOI 10.3997/2214-4609.202640062 (AEM 2026 workshop; uses Source Parameter Imaging magnetic constraints).

### Candidate 3: Pirttijärvi, Zaher and Korja (match: LOW-MEDIUM; older, but the canonical GTK airborne-magnetics inversion)
- **Citation:** Pirttijärvi, M., Zaher, M.A., Korja, T. "Combined inversion of airborne electromagnetic and static magnetic field data." *Geophysica* 50(2), 65–87 (2014). https://www.geophysica.fi/pdf/geophysica_2015_50_2_065_pirttijarvi.pdf (DOI not found).
- **Data:** GTK low-altitude national airborne survey (frequency-domain AEM plus total-field magnetics), two Finnish test areas.
- **Method:** laterally constrained inversion. The magnetic part uses vertical magnetized prisms, and basement susceptibility is inverted as a 4th parameter because high susceptibility gives **negative in-phase AEM anomalies**.
- **Efficiency:** designed to process whole map sheets (many lines) semi-automatically.
- Outside 2022–2026, but relevant to the "negative" values theme. The negative values here are in-phase EM driven, not remanence.

### Candidate 4: Kevitsa probabilistic model-based magnetic inversion (match: LOW)
- Güdük, N., TU Delft MSc thesis, "Model-based probabilistic inversion using magnetic data" (https://repository.tudelft.nl/record/uuid:33ee0bfd-a6c5-4319-ae06-d454d3286ee5).
- There is also a 2021 *Geosciences* (MDPI) 11(4):150 paper on Kevitsa probabilistic 3D model-based inversion of airborne magnetics with a Finnish co-author (https://www.mdpi.com/2076-3263/11/4/150). Authors and DOI are **UNVERIFIED** (403).
- Region: Kevitsa Ni-Cu-PGE, Finnish Lapland. Efficiency comes from a parametric (model-based) rather than voxel parameterization (**UNVERIFIED** detail).

### Candidate 5: Tomofast-x 2.0 (match: LOW, method only)
- Tomofast-x 2.0, *Geoscientific Model Development* 17, 2325 (2024). https://gmd.copernicus.org/articles/17/2325/2024/
- An open-source, MPI-parallel Fortran code with **wavelet compression of the sensitivity matrix** (the classic memory trick) and topography support.
- **No Finnish application found.**

### Other notes from the search
- No result links the term "Hakku" to any inversion paper.
- The negative-susceptibility-to-remanence reading is classic. Reversed-polarity anomalies used to be treated as "apparent negative susceptibility" (CSIRO chapter). UBC-GIF codes avoid negative values through a positivity transform, and SimPEG examples bound susceptibility to be non-negative.
- So a paper that *reports* negative susceptibility probably ran an unbounded L2 inversion. Calling the negatives remanence without MVI, amplitude inversion, or paleomag or Q data is a weak inference. That is a likely reviewer critique.

---

## JOB 2: Dataset inventory (Finland; Lapland or Koillismaa focus if the target is Akanvaara or Kevitsa)

Licence context: GTK's open licence is compatible with CC BY 4.0 and valid from 1 Dec 2025 (https://www.gtk.fi/en/open-licence/). Most datasets are in Hakku (https://hakku.gtk.fi), which also has an online shop for charged products and its own order and delivery terms. GTK's site lists Hakku and Ground surveying (PTR) as services with exceptional terms. The 2026 price list is at https://www.gtk.fi/app/uploads/2026/05/gtk_paikkatietoaineistot_hinnasto_price_list_for_digital_data_products_2026_05_27.pdf

| # | Dataset | Provider | Type | Resolution / spacing | Years | Access | Format | Known issues | URL |
|---|---|---|---|---|---|---|---|---|---|
| 1 | National low-altitude airborne magnetics (aeromagnetic anomaly map) | GTK | Total-field magnetics (part of a combined mag + AEM + gamma system) | Nominal line spacing 200 m, flight altitude 30–40 m. Some projects flown at 50 m spacing. National grids resampled to 500 m x 500 m cells | 1972–2007 acquisition (full coverage about 2008) | Map/grid on avoindata.fi tagged CC BY 4.0. Line data via Hakku (login / possibly charged, **UNVERIFIED**) | Grids (GeoTIFF or ERMapper typical, **UNVERIFIED**). Line data XYZ (**UNVERIFIED**) | 35 years of mixed instruments. High-latitude diurnal (auroral) variation needs a base station in the survey area. Levelling and mis-ties. Terrain clearance varies. IGRF/DGRF epoch handling across surveys **UNVERIFIED** | https://www.avoindata.fi/data/en/dataset/activity/lentogeofysikaalinen-magneettinen-anomaliakartta ; ref. Airo (ed.) 2005, GTK Special Paper 39 |
| 2 | Airborne radiometrics (K, Th, U, total count) | GTK | Gamma-ray spectrometry | Same flights as #1 | 1972–2007 | Open (avoindata.fi, CC BY 4.0) | Grids | Soil moisture, snow and bog attenuation. Altitude variations | https://avoindata.suomi.fi/data/en_GB/dataset/aeroradiometrinen-uraanikartta |
| 3 | Airborne EM (frequency domain, in-phase and quadrature, 2 to 4 frequencies) | GTK | AEM | Same flights as #1 | 1972–2007 | Maps open. Lines via Hakku | Grids / XYZ | Negative in-phase over highly magnetic ground (Pirttijärvi 2014). Cultural noise. Drift | Hakku |
| 4 | Ground gravity, Finnish national gravity net plus GTK regional gravity | FGI (now NLS-FGI) and GTK | Bouguer anomaly | FGI net average station separation about 5 km (about 1 per 25 km²). GTK regional surveys are denser (layer "Regional gravity survey") | 1945–1996 (FGI), GTK ongoing | Bouguer data listed as **purchasable**, enquiries geodata@gtk.fi. Map layers via WMS | Point data / grids | Mixed reduction density (Mid-Norden used 2.670 g/cm³). Terrain corrections. Sparse in places | https://gtkdata.gtk.fi/arcgis/rest/services/MDaE/Geophysics/MapServer/21 |
| 5 | Petrophysical database | GTK | Density, susceptibility; subset with NRM / Q-ratio, resistivity, IP | About 129,252 density samples (mean 2725 kg/m³, sd 135). Remanence for a subset only | 60+ years | Indices visible in Hakku. Full data access **UNVERIFIED** | Point tables | Outcrop sampling bias toward surface. Weathering. Remanence sparse. Units and measurement-method changes | Pirttijärvi et al. 2013 Geophysica 49:33 https://www.geophysica.fi/pdf/geophysica_2013_49_1-2_033_pirttijarvi.pdf |
| 6 | Bedrock of Finland 1:200 000 (seamless) | GTK | Lithology and structure polygons | 1:200k | Current | Open licence, price 0.00, Hakku package, plus WMS/WFS | Shapefile / GeoPackage (**UNVERIFIED**), WMS/WFS | Generalized from scale-free data. Under cover, contacts are inferred from geophysics (circularity risk) | https://hakku.gtk.fi ; https://gtk.fi/en/services/data-sets-and-online-services-geo-fi/map-services |
| 7 | Bedrock 1:1 000 000 | GTK | Lithology | 1:1M | Current | Open | Vector | Coarse | Hakku |
| 8 | Bedrock drill holes | GTK | Collar, survey, lithology, ore minerals, alteration, assays | About 30,000 to 34,000 holes. National Drill Core Archive (Loppi; Rovaniemi for the north) holds over 3.6 Mm of core | Decades | WFS layer "kallioperakairaukset" open. Detail logs partly restricted (**UNVERIFIED**) | WFS / tables | Old collar coordinates (KKJ datum). Missing downhole surveys. Company data confidentiality | https://gtkdata.gtk.fi/arcgis/rest/services/GTKWFS/MapServer/7 ; https://tupa.gtk.fi/paikkatieto/meta/bedrock_drilling.html |
| 9 | FIRE deep seismic reflection profiles | GTK, Univ. Helsinki, Univ. Oulu | CMP reflection, 4 transects, 2,104 km | Crustal scale | 2001–2005 | CC BY 4.0 per research.fi. Some records restricted. Per-profile DOIs | SEG-Y and stacks via OpenFIRE / Etsin | Crooked-line geometry. Low fold in places | https://www.helsinki.fi/en/infrastructures/fin-epos/the-finnish-reflection-experiment ; GTK Special Paper 43 |
| 10 | Magnetotellurics | Univ. Oulu (MaSca: 79 sites, 15–30 km spacing, 0.001–10,000 s, 2014), BEAR array, LTU (324 sites, 2021) | MT | Regional | 1990s–2021 | No public download found (**UNVERIFIED**) | EDI | Extreme phases and strong 3D effects in Lapland (Autio et al.) | Autio et al. 2020 *Tectonophysics* (MT array central Finnish Lapland II) |
| 11 | DEM 2 m (lidar) | National Land Survey (MML) | DTM | 2 m grid. Laser density ≥0.5 pts/m². Accuracy class I 0.3 m, class II 0.3–1 m | 2008– | Open (NLS CC BY 4.0) | GeoTIFF | Not full national coverage. Use 10 m DEM elsewhere | https://maanmittauslaitos.fi/en/node/10715 |
| 12 | EMAG2v3 | NOAA NCEI | Global magnetic anomaly grid | 2 arc-min, at 4 km altitude (plus sea-level ocean product) | 2017 | Open | GeoTIFF / CSV | Far too coarse for deposit scale. Use only for regional and long-wavelength checks | https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ngdc.mgg.geophysical_models:EMAG2_V3/html |
| 13 | WGM2012 | BGI | Bouguer, isostatic and free-air grids | 2'x2'. From EGM2008 + DTU10, ETOPO1 terrain correction | 2012 | Open (BGI) | Grid | "Spherical Bouguer" differs from conventional Bouguer and is poorly matched to land data (Milsom critique) | https://bgi.obs-mip.fr (**UNVERIFIED** URL) ; https://johnmilsom.online/egm_wgm |
| 14 | GOCE / satellite gravity models (e.g. GOCO) | ESA / ICGEM | Satellite gravity | About 80–100 km half-wavelength (**UNVERIFIED**, not searched) | 2009–2013 mission | Open | Spherical harmonics | Only long wavelengths | http://icgem.gfz-potsdam.de (**UNVERIFIED**) |
| 15 | Fennoscandian Shield magnetic and Bouguer anomaly maps | GTK, SGU, NGU and others (Korhonen et al. 2002) | Compilation | 1:2M map. Digital grid 2 km (WDMAM) | 2002 | Ask GTK/SGU/NGU | Grid | Smoothed compilation | EGU2010-15532 abstract |
| 16 | Mineral deposit database / MDaE | GTK | Deposits, occurrences, prospects (33 tables, 216 fields). MIN4EU harmonized export | Point | Current | Hakku download. MDaE map app | Tables / WFS | Mixed-quality legacy resource figures | https://gtkdata.gtk.fi/MDaE/index.html ; https://www.gtk.fi/en/current/gtks-mineral-deposit-database-renewed-extractive-waste-now-on-a-map/ |
| 17 | 3D models | GTK (e.g. Vihanti 30x18x5 km; Pyhäsalmi–Vihanti regional / mine; Outokumpu RI 195, 2012; Koillismaa Common Earth Model 2025, SEEMS DEEP) | Geological / geophysical 3D | Various | 2011–2025 | No public download found (**UNVERIFIED**). Ask GTK | Gocad / Leapfrog (**UNVERIFIED**) | Interpretations built on the same potential-field data (circularity risk) | https://gtk.fi/en/services/interpretation-and-modelling-of-regional-data |

Coordinate systems: modern GTK data use ETRS-TM35FIN (EPSG:3067). Legacy data use KKJ / YKJ (EPSG:2393). Check before merging (**UNVERIFIED** for each dataset).

---

## QC checklist by data type

**Airborne magnetics (GTK line data)**
- [ ] Check flight-line and tie-line geometry. Report actual line spacing and the terrain clearance distribution (radar altimeter). Flag any segments above 60 m.
- [ ] Check diurnal correction (base station record). Look for auroral-storm days and herringbone or corrugation artefacts.
- [ ] Check levelling: tie-line mis-ties, micro-levelling. Look at the grid's 1st vertical derivative for line-parallel stripes.
- [ ] Identify the IGRF/DGRF model and epoch used per survey block. Check that block-to-block DC shifts are removed when merging 1972–2007 surveys.
- [ ] Remove cultural noise (power lines, railways, mines).
- [ ] Inversion inputs:
  - Upward continuation or a ≥ data-altitude cell size so 30 m clearance does not alias near-surface sources.
  - Downsample along-line data (spacing is much denser than the 200 m line spacing). State the decimation.
  - Assign data uncertainty as a floor plus a percentage, and justify it.
- [ ] Remanence screening:
  - Reversed or asymmetric anomalies.
  - ARRA or amplitude-data test.
  - Compare against GTK Q-ratios.
  - Run MVI or amplitude inversion before attributing negative susceptibility to remanence.
- [ ] Overfitting test:
  - Is the final χ² ≈ N, or is the misfit far below the noise level?
  - Show the L-curve or discrepancy principle.
  - Show a residual map: it should be spatially uncorrelated noise, with no geology left in it.
  - Run hold-out lines or cross-validation.

**Gravity**
- [ ] Station density vs target size (the FGI net at about 5 km is too sparse for deposit-scale work).
- [ ] Bouguer density (2.67 vs local values) and terrain correction using the NLS DEM.
- [ ] Datum and height system consistency (N60 / N2000; **UNVERIFIED** per dataset).
- [ ] Regional/residual separation method stated.
- [ ] Repeat-station and loop-closure errors.

**Petrophysics**
- [ ] Sample count per lithology and spatial clustering.
- [ ] Susceptibility units (SI vs cgs) and log-normal statistics.
- [ ] Remanence available? Report NRM direction and Q.
- [ ] Do lab densities and susceptibilities bound the inversion's recovered range? Negative susceptibility has no petrophysical support except diamagnetic rocks (about -1e-5 SI).

**Geology and drill holes**
- [ ] Datum conversion KKJ to ETRS89.
- [ ] Downhole survey present?
- [ ] Logging consistency over decades.
- [ ] Use holes as independent validation, not only as constraints.

**Seismic (FIRE) and MT**
- [ ] Depth conversion velocity model.
- [ ] Crooked-line migration artefacts.
- [ ] MT static shift and 3D distortion.
- [ ] Coordinate alignment with the inversion mesh.

**DEM**
- [ ] 2 m coverage gaps (use the 10 m DEM elsewhere).
- [ ] Vertical datum.
- [ ] Use the DEM for mesh topography and for sensor height (sensor height = altimeter + DEM).

**Global grids (EMAG2v3, WGM2012, GOCE)**
- [ ] Use only for long wavelengths.
- [ ] Do not mix with airborne data without spectral matching.
- [ ] Note the WGM2012 spherical-Bouguer definition.

---

## Sources (key)
- arXiv:2602.03857 https://arxiv.org/abs/2602.03857 ; https://arxiv.org/html/2602.03857v1 ; https://github.com/naimat04
- EAGE 2026 Akanvaara https://www.earthdoc.org/content/papers/10.3997/2214-4609.202620160
- Pirttijärvi et al. 2014 https://www.geophysica.fi/pdf/geophysica_2015_50_2_065_pirttijarvi.pdf
- Tomofast-x 2.0 https://gmd.copernicus.org/articles/17/2325/2024/
- GTK MPM data description (1972–2007, 30–40 m altitude, 200 m spacing, 500 m grids) https://gtkdata.gtk.fi/mpm/data.htm
- GTK open licence https://www.gtk.fi/en/open-licence/ ; data policy https://www.gtk.fi/en/data-policy/
- FGI gravity (Geophysica 1997) https://archive.geophysica.fi/pdf/geophysica_1997_33_1_051_elo.pdf
- Petrophysics https://www.geophysica.fi/pdf/geophysica_2013_49_1-2_033_pirttijarvi.pdf
- FIRE https://www.helsinki.fi/en/infrastructures/fin-epos/the-finnish-reflection-experiment
- NLS DEM https://maanmittauslaitos.fi/en/node/10715
- EMAG2v3 https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ngdc.mgg.geophysical_models:EMAG2_V3/html
- WGM2012 critique https://johnmilsom.online/egm_wgm
- Koillismaa seismic preprint https://egusphere.copernicus.org/preprints/2026/egusphere-2026-3292/
