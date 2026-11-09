---
title: "Inversion Assignment: First Report"
subtitle: "Joint and probabilistic inversion for Alt Carbon. Tasks 1 to 4 from Shrey Agarwal's email of 3 November 2026"
author: "Rizwan Mir, PhD scholar, Ashoka University (supervisor: Prof. Aalok Thakkar)"
date: "9 November 2026"
---

# Summary

This report is a first pass at all four tasks in Shrey Agarwal's email. It was prepared with AI research assistance (web search plus code). It is a starting point for discussion, not a finished review.

| Task | Status | Main result |
|---|---|---|
| 1. Literature review and ranking | Draft complete | 22 methods in 8 families are scored on 6 criteria, with 85 references. Combined methods rank highest. Shortlist: (1) PGI extended to three properties, with randomise-then-optimise for uncertainty; (2) an implicit geological model sampled with HMC or parallel tempering; (3) structural coupling with ensemble uncertainty, checked against a small HMC reference. |
| 2. Analyse and reproduce the attached paper | **Blocked: we do not have the paper** | Synthetic stress tests are done. They show that negative susceptibility is **not** evidence of remanence by itself: regional-field errors and over-fitting give the same result. A test protocol is ready for the real paper. |
| 3. Dataset report and better methods | Draft complete | 17 Finnish datasets are listed with resolution, access and known problems, with a QC checklist for each data type. 15 ideas from medical imaging, astrophysics and machine learning are ranked. |
| 4. Third physical property | Formulations drafted | Three Bayesian formulations are written out. Recommended first prototype: a latent lithology field with a petrophysical Gaussian mixture, with DC resistivity/IP as the third property. |

## Five key points

1. **Negative susceptibility needs a test, not an assumption.** In our synthetic test, a remanent body inverts to negative susceptibility (mean −0.014 SI against a true +0.03). But a −25 nT regional-field error with **no** remanence gave a larger negative extreme (−0.021 SI) and more negative cells (32 % against 21 %). Under-regularisation gives the same effect. The deciding test is this: with a positivity bound, remanent data **cannot be fit** (φ<sub>d</sub>/N = 69). Without remanence, the data fit at φ<sub>d</sub>/N = 0.24. Magnetisation vector inversion (MVI), amplitude inversion and GTK Q-ratios then confirm the cause.
2. **Over-fitting is easy to detect with hold-out flight lines.** As the regularisation weight β falls from 10⁴ to 10⁻¹, the training misfit falls by about 4 orders of magnitude. Over the same range the error on held-out lines increases from 1.9 nT to 5.7 nT. The best β gives φ<sub>d</sub>/N ≈ 0.5–1, which agrees with the discrepancy principle.
3. **Memory is the real limit at regional scale.** A dense sensitivity matrix for a 100 km region with 50 m cells is about 2.6 PB. Each published efficiency approach (wavelet compression, data-space solves, octree meshes, on-the-fly GPU kernels) has a cost. The report lists these costs so that Alt Carbon's preferred approach can be compared fairly.
4. **The paper is not identified.** The best public candidate is arXiv:2602.03857, a GPU and data-space gravity inversion with GTK co-authors. Its version 1 uses Indian gravity data, not Hakku magnetics, so it is probably **not** the attached paper. We need the PDF.
5. **A research gap exists for Task 4.** We found no published work that does a fully sampled 3D joint inversion of gravity, magnetics and a third physics with a latent lithology field. This can be the core of a PhD chapter.

## What we need from Alt Carbon

1. The attached paper (PDF) and any code.
2. A Hakku account, or the line data for the study area.
3. A technical contact for weekly questions.
4. A short, permitted description of the current gravity–magnetic coupling (for Task 4).
5. Rules for publication and IP, and for the use of AI tools with confidential material.

## How to read the citations

Research agents collected the references through web search. Each reference has a DOI or arXiv ID that appeared in a search result. Items tagged **UNVERIFIED** need a manual check. Scores in the ranking tables are our **assessment**, not measured results.
