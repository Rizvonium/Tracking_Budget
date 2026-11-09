# Task 2: Analysis and reproduction of the attached paper

## 2.1 Status

We do not have the attached paper. A web search found no paper that matches the whole description (a magnetic inversion on GTK Hakku data that explains negative susceptibility as remanence and has a memory-efficiency focus). The candidates are below. We need the PDF to finish this task.

| Candidate | Match | Why |
|---|---|---|
| Nimatullah, Mishra, Kamm & Singh, *Backend-agnostic Julia framework for 3D modeling and inversion of gravity data*, arXiv:2602.03857 (v1 Jan 2026, v2 Sep 2026) | Medium | Two co-authors are from GTK, and the paper has a Finnish HPC grant. Data-space inversion plus a CPU/GPU code (about 3.3 M prisms and 179 k data before the memory limit). But v1 is gravity only, with Indian field data, and reports no β or χ² values. Code: github.com/naimat04/GravityInversionGPU.jl |
| EAGE 2026 abstract, Akanvaara layered intrusion, Finland, DOI 10.3997/2214-4609.202620160 | Medium | Finnish magnetic inversion stabilised with SVD of the sensitivity matrix. Authors UNVERIFIED. |
| Pirttijärvi, Zaher & Korja (2014), *Geophysica* 50(2):65–87 | Low–medium | GTK airborne magnetic + EM inversion. The negative values here come from in-phase EM, not from remanence. |
| Kevitsa probabilistic magnetic inversion (*Geosciences* 2021, 11(4):150) | Low | Finnish, probabilistic, parametric. Details UNVERIFIED. |
| Tomofast-x 2.0 (*GMD* 17:2325, 2024) | Low | Wavelet-compressed sensitivity matrix. No Finnish case found. |

## 2.2 Critical points to check in any such paper

**Over-fitting.** Look for:

- the final misfit compared with the number of data (χ² ≈ N),
- how β was selected (L-curve, discrepancy principle, GCV),
- a residual map (it must look like random noise, with no geology in it),
- validation on independent data (hold-out lines, drill holes).

A paper that gives none of these cannot show that it does not over-fit.

**Negative susceptibility explained as remanence.** Natural rocks have almost no negative susceptibility: diamagnetic minerals such as quartz and calcite are near −1×10⁻⁵ SI. A negative value in an inversion comes from one of these causes:

1. Remanent magnetisation at an angle to the present field.
2. An error in the regional field or the IGRF (wrong epoch, levelling, survey merge).
3. No positivity bound, combined with smoothness regularisation. The smoothing makes side-lobes around real bodies.
4. Over-fitting of noise.
5. Depth weighting and a reference model that move the anomaly.
6. Self-demagnetisation in very high-susceptibility bodies (above about 0.1 SI).

To claim cause 1, the paper must exclude causes 2–6. The standard tools are MVI, inversion of amplitude data, and measured Koenigsberger ratios (Q) from the GTK petrophysical database.

**Memory and compute.** Record the exact method (data-space solve, compression, octree mesh, on-the-fly kernels, GPU) and its costs: accuracy loss, time per iteration, and the limits it puts on which regularisation is possible.

## 2.3 Synthetic stress tests (done)

We built a controlled test in SimPEG 0.26 (Python) on a 4-CPU cloud machine. All code and results are in the repository.

**Setup:**

- **Mesh:** 24 × 24 × 12 cells of 25 m (6,912 cells).
- **Data:** total-field magnetic data at 30 m height on a 24 × 24 grid (576 points). The field is B₀ = 52,000 nT, inclination 75°, declination 10°, similar to Finland.
- **Hold-out lines:** every 4th flight line (144 data) is used only for testing. The inversion uses the remaining 432 data.
- **Bodies:**
  - Body A: χ = 0.05, induced magnetisation only.
  - Body B: χ = 0.03. In the remanence case, body B also has reversed remanence (Q = 3, inclination −60°, declination 190°).
- **Noise:** Gaussian, with standard deviation 2 % of the value + 2 nT.
- **Inversion:** scalar susceptibility, L2 smallness + smoothness, sensitivity weighting, projected Gauss–Newton.

### Test E1/E2: remanence or something else?

![Cross-sections at y = 0. Top left: true model. Top right: a scalar inversion without bounds turns the remanent body B negative. Bottom left: with χ ≥ 0, the inversion cannot fit the data and puts false material near the surface. Bottom right: the same inversion without remanence gives only small negative side-lobes.](fig/fig_remanence.png)

| Case | φ<sub>d</sub>/N (train) | RMS on hold-out lines (nT) | min χ | % cells < −0.002 | mean χ in body B (true 0.03) |
|---|---|---|---|---|---|
| Remanent, no bounds | 0.18 | 3.4 | −0.020 | 21 % | **−0.014** |
| Remanent, χ ≥ 0 | **68.8** | **86.1** | 0 | 0 % | 0.000 |
| Induced, no bounds | 0.20 | 3.0 | −0.004 | 3 % | 0.007 |
| Induced, χ ≥ 0 | 0.24 | 2.9 | 0 | 0 % | 0.008 |
| Induced, −25 nT regional error, no bounds | 0.20 | 3.1 | **−0.021** | **32 %** | 0.007 |
| Induced, 0.1 nT/m trend not removed, no bounds | 0.20 | 3.1 | −0.020 | 13 % | 0.007 |

![Regional-field errors without remanence also give strong negative susceptibility.](fig/fig_regional.png)

**Conclusions:**

1. A negative-susceptibility body by itself does not separate remanence from processing errors. In this test, a regional error gave **more** negative cells (32 %) than real remanence (21 %).
2. **The positivity test discriminates.** With χ ≥ 0, the remanent data cannot be fit (φ<sub>d</sub>/N rose from 0.18 to 69), but the induced data still fit (0.24). Recommended protocol: run the bounded inversion, and look for areas where it cannot fit. Then test those areas with MVI or amplitude inversion and with Q values.
3. Note that a smooth L2 inversion in this test also under-estimates susceptibility (0.015 against a true 0.05) and spreads the body in depth. The paper's depth estimates need the same check.

### Test E3: over-fitting

![A β sweep. As β decreases, the misfit on training lines continues to decrease. The error on hold-out lines is lowest near β = 10⁴ (φd/N ≈ 0.5) and then increases. Negative cells increase from 0 % to 31 %.](fig/fig_beta_sweep.png)

| β | φ<sub>d</sub>/N (train) | RMS on hold-out lines vs noise-free field (nT) | % negative cells |
|---|---|---|---|
| 10⁻¹ | 0.00002 | 5.7 | 31 % |
| 10¹ | 0.001 | 5.0 | 27 % |
| 10³ | 0.20 | 3.0 | 3 % |
| **10⁴** | **0.53** | **1.9** | **0.2 %** |
| 10⁵ | 2.3 | 7.4 | 0 % |

Over-fitting and negative susceptibility increase together. If the paper's model shows many negative cells and a misfit well below N, this is a clear sign of over-fitting.

### Test E4: memory

![The dense sensitivity matrix size is N × M × 4 bytes (float32). Measured points agree with the formula. A 100 km region with 50 m cells needs about 2.6 PB.](fig/fig_scaling.png)

| Scale | Data N | Cells M | Dense G (float32) |
|---|---|---|---|
| This test | 576 | 6,912 | 16 MB |
| 10 km block, 50 m cells | ~2×10⁵ | ~3.2×10⁶ | ~2.6 TB |
| 100 km region, 50 m cells | ~4×10⁶ | ~1.6×10⁸ | ~2.6 PB |

The common solutions are:

- **Wavelet or adaptive compression of G:** 10–100× smaller, with some accuracy loss.
- **Octree meshes:** fewer cells at depth.
- **Data-space formulation:** solves an N × N system, which helps when N ≪ M.
- **Matrix-free GPU kernels:** compute G on the fly instead of storing it.
- **Equivalent sources and tiling:** cut the problem into tiles.

The reproduction must record the memory, time and accuracy for each approach against a dense reference on a small tile.

## 2.4 Reproduction plan (when the paper and data arrive)

1. **Week 1:** Fill the parameter checklist from the paper: data decimation, noise model, mesh, β rule, bounds, depth weighting, IGRF epoch. Mark each missing value and ask the authors.
2. Make a dense reference inversion on a small tile in SimPEG. Make the paper's efficient version on the same tile, and compare them.
3. Run the full region with the paper's settings. Compare with each figure.
4. Run the stress tests E1–E4 on the real data:
   - hold-out lines,
   - a β sweep,
   - a bounded vs unbounded comparison,
   - MVI or amplitude inversion,
   - sensitivity to the regional removal,
   - validation against drill holes and GTK petrophysics.
5. Write a 6–8 page technical note with the results.
