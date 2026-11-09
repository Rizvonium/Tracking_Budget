# AI Prompts for the Alt Carbon Inversion Assignment

Source: email from Shrey Agarwal (Alt Carbon) to Prof. Aalok Thakkar, 3 Nov 2026, "Inversion Assignment".

How to use:
1. Paste the **Shared Context** block first in every new AI chat.
2. Then paste the prompt for the task.
3. Replace every `[[...]]` placeholder with real data.
4. Use one chat for each sub-step. Long chats lose detail.
5. Verify every citation, number and equation yourself. AI tools can invent references.

---

## 0. Shared Context (paste first, every time)

```text
CONTEXT
I am Rizwan Mir, a PhD scholar in Computer Science at Ashoka University (India),
supervised by Prof. Aalok Thakkar. My background: geospatial modelling, remote
sensing, Python, and machine learning. I am new to applied geophysics.

I am doing a research assignment for Alt Carbon, an Indian climate-tech company.
Their team (Shrey Agarwal, CEO; Abhishek Kumar; Dr. Sourav Ganguly) works on
geophysical inversion: estimating subsurface physical properties (density,
magnetic susceptibility, and later conductivity/chargeability/seismic velocity)
from surface or airborne measurements (gravity, magnetics, IP/EM, seismic).
They have built their own method that couples gravity and magnetic inversion
probabilistically. They want to add a third physical property.

KEY TECHNICAL TERMS
- Forward problem: d = G(m). Inverse problem: estimate m from noisy d.
- The problem is ill-posed and non-unique (especially potential fields:
  gravity and magnetics have no intrinsic depth resolution).
- Deterministic inversion: minimise phi_d + beta * phi_m (Tikhonov), with
  depth weighting, bounds, and reference models.
- Joint inversion: several datasets/properties at once, linked by structural
  coupling (cross-gradient, joint total variation), petrophysical coupling
  (clustering, Gaussian mixture models), or shared geology (level sets).
- Probabilistic inversion: Bayesian posterior p(m|d); MCMC, variational
  inference, ensemble methods, trans-dimensional methods, and learned
  priors (deep generative models, normalising flows, diffusion models).
- Magnetic complications: remanent magnetisation, self-demagnetisation,
  amplitude inversion, magnetisation vector inversion (MVI).

RULES FOR YOUR ANSWER
1. Do not invent references. For every paper give authors, year, title,
   venue and DOI or arXiv ID. If you are not sure that a paper exists,
   say "UNVERIFIED".
2. Separate facts from your own opinion. Mark opinions as "ASSESSMENT".
3. Use standard notation and define every symbol.
4. When you suggest code, use Python and open libraries (SimPEG,
   Fatiando a Terra / Harmonica / Verde, PyGIMLi, NumPy, SciPy, PyTorch,
   JAX). Say which version you assume.
5. If information is missing, list your assumptions and ask me.
6. Be concise. Use tables where a comparison helps.
```

---

## Task 1. Literature Review: Joint and Probabilistic Inversion

**Shrey's aim:** a full review of the joint and probabilistic inversion literature (papers, preprints and whitepapers). Rank the methods by strengths, weaknesses, mathematical assumptions, computational efficiency and how well they apply to Alt Carbon's problems.

**Target time:** 1–2 weeks.

### Prompt 1A: Map the field

```text
TASK
Build a taxonomy of joint inversion and probabilistic inversion methods for
geophysical data (gravity, magnetics, IP, EM, magnetotellurics, seismic).

Cover at least these families:
1. Structural coupling: cross-gradient (Gallardo & Meju), joint total
   variation, Gramian constraints (Zhdanov), structure-guided methods.
2. Petrophysical coupling: Gaussian mixture model / PGI (Astic & Oldenburg),
   fuzzy c-means clustering, known petrophysical relations.
3. Geology-based: level-set / parametric shape inversion, implicit
   geological models (e.g. GemPy, Loop3D) as priors.
4. Bayesian sampling: MCMC (Metropolis-Hastings, HMC, NUTS), trans-
   dimensional rjMCMC, parallel tempering, Stein variational gradient
   descent.
5. Variational and amortised inference: mean-field VI, normalising flows,
   simulation-based inference.
6. Learned priors: VAE/GAN priors, diffusion-model priors, plug-and-play
   priors, deep image prior.
7. Ensemble and uncertainty methods: ensemble Kalman inversion,
   randomise-then-optimise (RTO), Laplace approximation, null-space
   shuttles.
8. Large-scale computation: sparse/compressed sensitivity matrices
   (wavelet compression, octree meshes, ZFP), matrix-free methods,
   GPU implementations, equivalent-source methods.

For each family give:
- Core idea in 2-3 sentences, with the key equation.
- 3-5 foundational or most-cited papers (with DOI).
- 2-3 recent papers (2021 or later).
- Open-source code, if any.

OUTPUT
A markdown table and a short bibliography. Mark unverified items.
```

### Prompt 1B: Rank the methods

```text
TASK
Using the taxonomy below, rank the methods for Alt Carbon's use case:
3D joint inversion of airborne/ground gravity and magnetic data with
uncertainty, and a planned extension to a third property (IP, EM or
seismic). The survey area can be regional (10-100 km), so models can
have 10^5 to 10^7 cells.

[[PASTE TAXONOMY FROM 1A]]

Score each method from 1 to 5 on:
a) Mathematical assumptions (fewer and weaker = higher score). State them:
   linearity, Gaussian noise, known petrophysics, stationarity, etc.
b) Uncertainty quantification quality.
c) Computational cost and memory for 10^6 cells. Give big-O where possible.
d) Ease of adding a third physical property.
e) Robustness to remanent magnetisation.
f) Maturity: open code, field case studies.

OUTPUT
1. A ranking table with scores and one-line reasons.
2. A top-3 shortlist "worth experimenting with", each with a small
   experiment plan (data, software, success metric, time estimate).
3. A list of 5 open research gaps.
Mark all scores as ASSESSMENT.
```

### Prompt 1C: Read one paper in depth (repeat for each key paper)

```text
TASK
Read the paper below and fill this template.

[[PASTE PAPER TEXT OR ABSTRACT + METHOD SECTION]]

TEMPLATE
- Citation (with DOI):
- Problem and data types:
- Forward model(s):
- Objective function / posterior (write the equation):
- Coupling mechanism between properties:
- Assumptions (explicit and hidden):
- Optimiser or sampler:
- Mesh size, runtime, memory, hardware:
- Validation (synthetic? field? drilling?):
- Uncertainty output:
- Code availability:
- Weaknesses and failure cases:
- Relevance to Alt Carbon (1-5) and why:
```

---

## Task 2. Analyse and Reproduce the Attached Paper

**Shrey's aim:** analyse the attached paper in depth and reproduce its results. Alt Carbon already sees some problems: overfitting, and negative susceptibility explained as remanent magnetisation. They like the paper's memory and compute efficiency. The data is on the Geological Survey of Finland's (GTK) Hakku portal, and it needs a login. Test the limits of the method, not only the theory.

**Target time:** 2–4 weeks.

### Prompt 2A: Critical analysis

```text
TASK
Do a critical technical review of the paper below. Alt Carbon has
already flagged: (1) possible overfitting, (2) attributing negative
susceptibility values to remanent magnetisation, and (3) they like the
memory/computational efficiency approach.

[[PASTE FULL PAPER TEXT]]

ANSWER THESE
1. Summarise the method step by step, with all equations.
2. Explain the memory/compute efficiency trick exactly. What is its
   complexity in cells (M) and data (N)? What does it trade off?
3. Overfitting: what chi-squared / misfit target did they use? Is the
   noise model justified? Is there cross-validation or hold-out data?
   How would I test for overfitting (e.g. L-curve, discrepancy principle,
   GCV, hold-out lines, residual spatial autocorrelation)?
4. Negative susceptibility: list all physical and numerical causes
   (remanence, self-demagnetisation, diamagnetic minerals, regional
   field removal errors, levelling errors, IGRF errors, lack of
   positivity bounds, depth weighting). Which explanation is best
   supported? What test would separate them (e.g. MVI, amplitude
   inversion, Koenigsberger ratio from petrophysical samples, Helbig
   integrals)?
5. List every parameter I need to reproduce the work, and mark the ones
   the paper does not report.
6. List other weaknesses.

OUTPUT
Structured report with headings 1-6. Mark ASSESSMENT where needed.
```

### Prompt 2B: Reproduction plan and code

```text
TASK
Help me reproduce the paper's results in Python.

PAPER SUMMARY
[[PASTE OUTPUT OF 2A, SECTIONS 1, 2 AND 5]]

DATA
I downloaded from GTK Hakku: [[LIST FILES, FORMATS, CRS, LINE SPACING,
FLIGHT HEIGHT, UNITS]].

MY COMPUTE
Laptop/cloud: [[CPU cores, RAM, GPU yes/no]].

DO THIS
1. A step-by-step pipeline: data loading -> QC -> gridding/levelling ->
   regional removal -> mesh -> forward operator -> inversion -> plots.
2. For each step, recommend the library function (SimPEG / Harmonica /
   Verde) and the parameters that match the paper.
3. Write clean, modular code for steps 1-3 first. Use a config file
   (YAML) for all parameters. Set random seeds. Log runtime and peak
   memory.
4. Give a synthetic test first: a known prism model, forward modelled,
   noise added, then inverted. I must recover it before using real data.
5. Give me a checklist that compares my results against each figure and
   table in the paper.
```

### Prompt 2C: Stress-test the method

```text
TASK
Design experiments that find the limits of this method.

METHOD
[[PASTE METHOD SUMMARY]]

Design 6-10 experiments. For each give: hypothesis, setup, what to vary,
metric, expected failure sign. Include:
- Synthetic models with known truth: deep vs shallow bodies, dipping
  dykes, remanently magnetised bodies, two close bodies.
- Noise level and noise model mismatch.
- Regularisation parameter sweep (overfit vs underfit).
- Data decimation: line spacing, hold-out flight lines.
- Mesh size scaling: runtime and memory vs number of cells.
- Removing positivity bounds; comparing with MVI.
- Sensitivity to regional-field removal.

OUTPUT
A table of experiments, then the code skeleton for one experiment loop
that runs all of them and saves results to JSON + figures.
```

---

## Task 3. Dataset Report and Better Inversion

**Shrey's aim:** a detailed report on the quality, coverage and types of geological and geophysical data for the region in the paper. Then find out whether newer inversion methods from Task 1, or ideas from astrophysics, medical imaging and other inverse problems, give better results. The goal is a list of improvements worth implementing, and possibly a new whitepaper on the same dataset with stronger results.

**Target time:** 4–8 weeks (start the dataset report in week 2).

### Prompt 3A: Dataset inventory

```text
TASK
Help me build a dataset inventory for this study region:
[[REGION NAME, BOUNDING BOX lat/lon, COUNTRY = Finland]].

Search knowledge of public Finnish and European sources, including:
GTK Hakku (airborne magnetics, radiometrics, EM; ground gravity;
bedrock maps; drill-core and petrophysical databases), GTK
petrophysical database (density, susceptibility, remanence, Q ratio),
Finnish Geodetic Institute / NLS gravity, EMAG2/WDMAM, satellite
gravity (GOCE, WGM2012), DEM (NLS Finland 2 m / 10 m LiDAR DEM),
seismic profiles (FIRE), MT surveys, borehole logs, mineral occurrence
databases, and published 3D geological models.

For each dataset give: name, provider, data type, spatial resolution or
line spacing, flight height, year, coverage of the region, format,
access (open/login/paid), known quality issues, and URL.
Mark items as UNVERIFIED if you are not sure.

Then give me a QC checklist for each data type (e.g. magnetics: diurnal
correction, levelling, tie-line mismatch, heading error, IGRF epoch;
gravity: terrain correction, density used, station spacing).
```

### Prompt 3B: Cross-domain ideas

```text
TASK
Find inverse-problem ideas from other fields that could improve 3D
gravity/magnetic (and later multi-physics) inversion. The current
baseline is: [[ONE-PARAGRAPH SUMMARY OF THE PAPER'S METHOD AND ITS
WEAKNESSES FROM TASK 2]].

Look at:
- Medical imaging: CT/MRI compressed sensing, total variation, PET
  multi-modal priors (e.g. parallel level sets), learned unrolled
  reconstruction (LEARN, variational networks), electrical impedance
  tomography (very close to DC/IP).
- Astrophysics: radio interferometric imaging (CLEAN, sparsity in wavelet
  dictionaries, Bayesian imaging such as Event Horizon Telescope methods),
  gravitational lensing mass reconstruction (similar potential-field
  non-uniqueness), Hamiltonian Monte Carlo at scale.
- Signal processing / ML: score-based diffusion priors for inverse
  problems (DPS, DDRM), neural fields / implicit neural representations,
  physics-informed neural networks, operator learning (FNO, DeepONet) as
  fast surrogates, simulation-based inference.
- Other geoscience: full-waveform inversion tricks, ensemble methods
  from weather data assimilation.

For each idea give: source field, key paper (DOI), what it solves,
how to transfer it to potential-field inversion, expected gain,
risks, and implementation effort (days). Rank the top 5.
```

### Prompt 3C: Experiment and whitepaper plan

```text
TASK
Plan the experiments and a whitepaper that improves on the baseline
paper using the same dataset.

INPUTS
Baseline results: [[KEY NUMBERS/FIGURES FROM TASK 2]]
Shortlisted methods: [[TOP 3 FROM 1B AND TOP 5 FROM 3B]]
Data inventory: [[SUMMARY FROM 3A]]
Compute: [[RESOURCES]]

DO THIS
1. Define fair comparison metrics: data misfit on hold-out lines, model
   recovery on synthetics built from the real geology, agreement with
   drill-hole/petrophysical data, uncertainty calibration (coverage of
   credible intervals), runtime, memory.
2. Propose 2-3 candidate improvements with a minimal experiment each.
3. Give a 6-week timeline with weekly deliverables.
4. Write a whitepaper outline: title, abstract (draft), sections,
   figures list, and the claims each figure must support.
5. List threats to validity.
```

---

## Task 4. Third Property in Alt Carbon's Coupled Inversion

**Shrey's aim:** Alt Carbon has its own method for coupling gravity and magnetic inversion probabilistically. They cannot yet add a third physical property (IP, EM or seismic). Work with the team on the mathematical formulation and the implementation. This is open research.

**Note:** Do not paste Alt Carbon's confidential method or data into a public AI tool without permission. Ask the team first which AI tools are allowed.

### Prompt 4A: Formulation options

```text
TASK
Help me formulate a probabilistic joint inversion that adds a third
physical property to an existing coupled gravity-magnetic inversion.

CURRENT METHOD (permitted summary only)
[[SUMMARY OF ALT CARBON'S METHOD: parameterisation, coupling term,
sampler/optimiser, scale]]

THIRD PROPERTY CANDIDATES
- DC resistivity / IP (chargeability): nonlinear, PDE forward model,
  strong near-surface sensitivity.
- EM (frequency or time domain, airborne): nonlinear, often 1D layered
  approximations.
- Seismic (velocity or reflectivity): different resolution and physics.

DO THIS
1. For each candidate, write the forward model, its cost, and its
   sensitivity pattern compared with gravity/magnetics.
2. Give 3 ways to couple it in a Bayesian framework:
   a) Shared latent geology (lithology classes) + petrophysical
      likelihood p(density, susceptibility, conductivity | class):
      a hierarchical model / Gaussian mixture.
   b) Structural coupling prior (cross-gradient or joint TV) as a
      Gibbs/energy term.
   c) Learned joint prior (normalising flow or diffusion model trained
      on geological simulations).
   Write the full posterior for each, define every symbol, and state the
   assumptions.
3. Discuss mesh mismatch (octree for potential fields vs PDE meshes for
   EM/IP) and how to map between meshes.
4. Discuss data weighting between datasets of different size and
   noise, and how to avoid one dataset dominating.
5. Recommend the most practical option for a first prototype and why.
```

### Prompt 4B: Prototype implementation

```text
TASK
Write a prototype for the chosen formulation:
[[CHOSEN FORMULATION FROM 4A]]

REQUIREMENTS
- Start in 2D (a cross-section) with synthetic data for all three
  properties, generated from a known lithology model.
- Use SimPEG for forward operators (gravity, magnetics, DC/IP) or
  PyGIMLi for DC/IP.
- Implement the posterior and a sampler or optimiser (start with MAP +
  Laplace, then MCMC/HMC if time allows). JAX or PyTorch for gradients
  is fine.
- Tests: recover the known model; show that adding the third property
  reduces posterior uncertainty in a target zone, compared with
  gravity+magnetics alone.
- Report runtime, memory, and convergence diagnostics (R-hat, ESS) for
  sampling.
- Modular code, config file, seeds, saved results.
```

---

## Prompt for the First Reply to Shrey and Prof. Thakkar

```text
TASK
Draft a short, professional email from me (Rizwan Mir, PhD scholar,
Ashoka University) to Shrey Agarwal, cc Prof. Aalok Thakkar.

Content:
- Thank him for the tasks.
- Proposed timeline: Week 1 Hakku access + paper reading + literature
  database; Week 2 Task 1 ranked review (v1) + Task 3 dataset inventory;
  Weeks 3-4 Task 2 reproduction and stress tests; Weeks 5-8 Task 3
  improvements and whitepaper draft. Task 4 after that, with the team.
- Ask: (1) the paper PDF and any code, (2) the main technical contact,
  (3) which data/code of theirs I may access, (4) compute available,
  (5) publication and IP terms, (6) which AI tools are acceptable for
  their confidential material.
Keep it under 180 words. Polite and direct.
```
