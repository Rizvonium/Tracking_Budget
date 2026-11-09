# Joint and Probabilistic Geophysical Inversion: A Ranked Review for 3-D Gravity–Magnetic(+X) Inversion

*Prepared for: Alt Carbon (3-D gravity + magnetic probabilistic joint inversion; adding a third property: IP / EM / seismic)*
*Compiled: 2026-11-09. Every reference was checked against a web search result; anything not checked is tagged **UNVERIFIED** (Section E).*

---

## 0. How to read this document

- **Problem framing.** We assume a voxel (or octree) model with about $10^6$ cells. Each cell holds a property vector $\mathbf{m}(x) = [\rho, \kappa\ \text{or}\ \mathbf{M}, \sigma\ \text{or}\ \eta\ \text{or}\ v_p]$. The data are $d_g$ (gravity), $d_m$ (TMI), and a third dataset $d_3$. The general objective or posterior is
  $$p(\mathbf{m}\mid \mathbf{d}) \propto \prod_{i} p(\mathbf{d}_i\mid F_i(\mathbf{m}_i))\; p_{\text{coupling}}(\mathbf{m}_1,\dots,\mathbf{m}_K)\; \prod_i p(\mathbf{m}_i),$$
  or, deterministically, $\phi(\mathbf{m}) = \sum_i \phi_d^{(i)} + \sum_i \beta_i \phi_m^{(i)} + \lambda\, \phi_c(\mathbf{m}_1,\dots,\mathbf{m}_K)$.
  The families below differ in **what $\phi_c$ / $p_{\text{coupling}}$ is** (sections 1–3, 6) and in **how the posterior is explored** (sections 4, 5, 7). Section 8 covers how the computation scales.
- **Scale reality check.** Gravity and magnetic forward operators are linear but dense. With $N=10^5$ data and $M=10^6$ cells, a dense double-precision sensitivity matrix needs $8NM \approx 800$ GB. A dense $M\times M$ posterior covariance is out of reach ($8\times10^{12}$ bytes). So at Alt Carbon's scale every UQ method has to be matrix-free, compressed, low-rank, or reduced-dimensional.
- **Scores in Section B are ASSESSMENTS** (my judgement from the literature), not measured benchmarks.

---

## A. Method families

### A1. Structural coupling (cross-gradient, joint total variation, Gramian, structure-guided)

**Core idea.** Different properties are assumed to change *at the same places*, so the coupling term rewards spatial alignment of gradients rather than any particular numerical relationship between the properties. That makes the approach the safe default when the petrophysical link is unknown or non-unique. Moorkamp et al. (2011) found that cross-gradient beat direct parameter coupling when the assumed relationship was violated. Gramian constraints generalise this: they penalise linear dependence between models, or between their gradients and other attributes, and can couple any number of properties.

**Key equations.**
- Cross-gradient (Gallardo & Meju 2003, 2004): $\mathbf{t}(m_i,m_j) = \nabla m_i \times \nabla m_j$, with $\phi_c = \sum_{i<j}\lVert \mathbf{t}(m_i,m_j)\rVert_2^2$.
- Joint total variation (Haber & Holtzman Gazit 2013): $\mathrm{JTV}(m_1,\dots,m_K) = \int_\Omega \sqrt{\textstyle\sum_{k=1}^K \lvert\nabla m_k\rvert^2 + \epsilon}\;dV$. It is convex, which makes it easier to optimise.
- Gramian (Zhdanov et al. 2012): $\phi_c = G(T m_1,\dots,T m_K) = \det\big[\langle T m_i, T m_j\rangle\big]_{i,j=1}^K$. Here $T$ is the identity (correlation coupling) or $\nabla$ (structural coupling).

**Foundational papers.**
- Haber & Oldenburg (1997), *Inverse Problems* 13(1):63–77. This is the original structural joint inversion.
- Gallardo & Meju (2003), *GRL* 30(13):1658, and Gallardo & Meju (2004), *JGR* 109:B03311. These introduced cross-gradients.
- Fregoso & Gallardo (2009), *Geophysics* 74(4):L31–L42. This is the 3-D cross-gradient method for gravity and magnetic data, the closest match to Alt Carbon's current pair.
- Gallardo & Meju (2011), *Rev. Geophys.* 49. A review of the structural approach.
- Moorkamp et al. (2011), *GJI* 184(1):477–493. A 3-D MT + gravity + seismic framework (jif3D) that compares direct coupling with cross-gradient coupling.
- Zhdanov, Gribenko & Wilson (2012), *GRL* 39:L09301. Introduced Gramian constraints.
- Haber & Holtzman Gazit (2013), *Surv. Geophys.* 34:675–695. Covers model fusion and JTV.

**Recent (2021+).**
- Giraud et al. (2021), *GMD* 14:6681–6709. Tomofast-x: open-source cross-gradient and petrophysical joint gravity–magnetic inversion.
- Vatankhah et al. (2022), *GJI* 230(3):1585–1611. Large-scale focusing gravity–magnetic joint inversion with a Gramian constraint.
- A 2024 *Surveys in Geophysics* review argues for combining Gramian terms to link the petrophysical and structural domains (DOI 10.1007/s10712-024-09832-0; authors UNVERIFIED).

**Open source.** Tomofast-x (Fortran/MPI, cross-gradient + clustering), jif3D (Moorkamp), SimPEG (`regularization.CrossGradient`, `JointTotalVariation`), pyGIMLi (structurally coupled joint inversion).

---

### A2. Petrophysical coupling (PGI / Gaussian mixture, fuzzy c-means, explicit relations)

**Core idea.** Each cell's property vector is assumed to come from one of a small number of rock units. Each unit has a characteristic (multivariate) distribution of physical properties. The coupling is a mixture prior, either a Gaussian mixture (PGI) or fuzzy clusters (FCM). Its parameters can be learned during the inversion, so the result is a property model plus a quasi-geology (unit membership) model. Explicit relations such as Gardner, Archie or a regression are the special case of one unit with a known function. They are powerful when correct and harmful when wrong (Moorkamp et al. 2011).

**Key equations.**
- PGI (Astic & Oldenburg 2019; Astic, Heagy & Oldenburg 2020). The prior is $p(\mathbf{m}) = \prod_{x}\sum_{k=1}^{K}\pi_k\,\mathcal{N}\big(\mathbf{m}(x)\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k\big)$. The smallness term is approximately $\phi_s = \tfrac12\sum_x \lVert \mathbf{W}^{1/2}(\mathbf{m}(x)-\boldsymbol{\mu}_{z^*(x)})\rVert^2_{\boldsymbol{\Sigma}^{-1}_{z^*(x)}}$, where $z^*(x)=\arg\max_k \pi_k\mathcal{N}(\cdot)$. The GMM parameters $\{\pi,\mu,\Sigma\}$ are refit after each model update (MAP-EM with conjugate priors).
- FCM (Sun & Li 2016, 2017): $\phi_c = \sum_{j=1}^{M}\sum_{k=1}^{K} u_{jk}^{q}\,\lVert \mathbf{m}_j - \mathbf{v}_k\rVert^2$, subject to $\sum_k u_{jk}=1$ and $q>1$.
- Explicit relation: $m_2 = f(m_1)$, imposed either through $\phi_c = \lVert m_2 - f(m_1)\rVert^2$ or by substitution.

**Foundational papers.**
- Bosch & McGaughey (2001), *The Leading Edge* 20(8). Lithology-constrained gravity–magnetic joint inversion.
- Bosch et al. (2006), *Geophysics*. 3-D gravity–magnetic joint inversion by Monte Carlo (DOI 10.1190/1.2209952).
- Sun & Li (2016), *Geophysics* 81(3):ID37–ID57. Guided FCM joint inversion. The brief says 2015; the journal issue is dated 2016.
- Sun & Li (2017), *GJI* 208(2):1201–1216. Generalised fuzzy clustering.
- Astic & Oldenburg (2019), *GJI* 219(3):1989–2012. PGI framework.
- Astic, Heagy & Oldenburg (2020), *GJI* 224(1):40–68. Multi-physics PGI, demonstrated on gravity + magnetics over a kimberlite. The brief says 2021; the published volume is 2020.

**Recent (2021+).**
- Joint inversion of potential-field data over the DO-27 kimberlite with a GMM prior, *Interpretation*, DOI 10.1190/int-2019-0283.1. The search attributes it to Astic and co-authors; the exact author list and year are UNVERIFIED. The hypabyssal kimberlite there is *remanent*, a useful remanence test case.
- Heagy, Astic, Capriotti, Weis & Oldenburg (2022), arXiv:2203.13894. Gravity + magnetic inversion to delineate rocks for **CO₂ mineralisation**, which is directly on Alt Carbon's topic. The author list comes from a mirror and should be confirmed on arXiv.
- Giraud et al. (2021), *GMD*. Petrophysical clustering constraints in Tomofast-x.

**Open source.** SimPEG PGI (`simpeg.utils.WeightedGaussianMixture`, `regularization.PGI`); companion notebooks `simpeg-research/Astic-2019-PGI` and `Astic-2020-JointInversion`; Tomofast-x (ADMM clustering).

---

### A3. Geology-based parameterisations (level sets, parametric/shape inversion, implicit geomodels as priors)

**Core idea.** Instead of inverting for $10^6$ free voxel values, invert for the parameters of a *geological model*. Those can be level-set functions that define unit boundaries, wireframe or parametric surfaces, or the inputs of an implicit geological modeller (interface points, orientations, fault parameters). Physical properties are then assigned per unit, which couples all physics *by construction*: one geometry feeds every forward model. The parameter space shrinks to $10^1$–$10^4$ dimensions, which makes full Bayesian sampling feasible. The cost is a strong modelling assumption: the unit topology and the per-unit property distributions must be right.

**Key equations.**
- Multiple level sets: $\mathbf{m}(x) = \sum_{k} \mathbf{p}_k\,\chi_k(x)$ with $\chi_k$ built from $H(\varphi_k(x))$. The level-set functions evolve by $\partial_t\varphi = -v_n\lvert\nabla\varphi\rvert$, where $v_n$ comes from the misfit gradient.
- Implicit geomodel prior: $\mathbf{m} = \mathcal{P}\big(\mathcal{G}(\boldsymbol{\theta})\big)$ and $p(\boldsymbol{\theta}\mid\mathbf{d}) \propto \prod_i p\big(\mathbf{d}_i \mid F_i(\mathcal{P}_i(\mathcal{G}(\boldsymbol{\theta})))\big)\,p(\boldsymbol{\theta})$. Here $\mathcal{G}$ is the implicit (co-kriging or potential-field) interpolator that maps geological inputs to a unit field, and $\mathcal{P}_i$ assigns property $i$ to each unit, possibly drawn from a per-unit distribution.

**Foundational papers.**
- Zheglova, Farquharson & Hurich (2013), *GJI* 192(2):688–698. Level-set traveltime inversion.
- Zheglova, Lelièvre & Farquharson (2018), *Geophysics* 83(1):R13–R30. Multiple level-set *joint* inversion of traveltime and gravity data.
- Lelièvre, Farquharson & Hurich (2012), *Geophysics*. Joint traveltime–gravity inversion on unstructured meshes (DOI 10.1190/1.3513182).
- de la Varga, Schaaf & Wellmann (2019), *GMD* 12:1–32. GemPy: differentiable implicit modelling with gravity forward modelling.
- Wellmann & Caumon (2018), *Adv. Geophys.* 59:1–121. Review of 3-D structural model uncertainty.
- Wellmann et al. (2018), *Geol. Soc. Spec. Publ.* 453:41–56. Bayesian integration of geology and geophysics, Sandstone greenstone belt.

**Recent (2021+).**
- Grose et al. (2021), *GMD* 14:3915–3937. LoopStructural.
- Scalzo et al. (2022), *GMD* 15:3641–3662. Blockworlds: Bayesian gravity/magnetic inversion of kinematic implicit models. Shows that naive voxelisation distorts the likelihood.
- Giraud et al. (2024), *Solid Earth* 15:63–89. Automatic implicit geological modelling inside deterministic potential-field inversion.
- Liang (2023), RWTH Aachen PhD thesis. Derivative-informed Bayesian inference (gpCN, SVGD) for trainable geomodels with gravity.

**Open source.** GemPy (Python, autodiff), LoopStructural / map2loop (Loop3D), Blockworlds (Scalzo), Obsidian (Scalzo et al. 2019; parallel-tempered MCMC over parametric multi-sensor geology).

---

### A4. Bayesian sampling (MCMC, HMC, rjMCMC, parallel tempering, SVGD)

**Core idea.** Draw samples from $p(\mathbf{m}\mid\mathbf{d})$ instead of computing a single optimum. Random-walk Metropolis–Hastings is exact in the limit but scales badly with dimension. Gradient-based samplers such as HMC (and MALA) use adjoint gradients to take long, high-acceptance moves. Trans-dimensional rjMCMC lets the data choose the model complexity (for example the number of Voronoi cells). Parallel tempering handles multimodality. SVGD moves a set of interacting particles deterministically toward the posterior, and sits between MCMC and variational inference.

**Key equations.**
- Metropolis–Hastings: $\alpha = \min\!\Big(1, \frac{p(\mathbf{m}'\mid\mathbf{d})\,q(\mathbf{m}\mid\mathbf{m}')}{p(\mathbf{m}\mid\mathbf{d})\,q(\mathbf{m}'\mid\mathbf{m})}\Big)$. The dimension-robust pCN proposal is $\mathbf{m}' = \sqrt{1-\beta^2}\,\mathbf{m} + \beta\,\boldsymbol{\xi}$ with $\boldsymbol{\xi}\sim\mathcal{N}(0,\mathbf{C}_{\text{prior}})$.
- HMC: $H(\mathbf{m},\mathbf{p}) = -\log p(\mathbf{m}\mid\mathbf{d}) + \tfrac12\mathbf{p}^\top\mathbf{M}^{-1}\mathbf{p}$, integrated with leapfrog steps.
- rjMCMC: $\alpha = \min\!\Big(1,\frac{p(k',\mathbf{m}'\mid\mathbf{d})\,q(k\mid k')}{p(k,\mathbf{m}\mid\mathbf{d})\,q(k'\mid k)}\,\lvert\mathbf{J}\rvert\Big)$.
- Parallel tempering: chains target $p(\mathbf{d}\mid\mathbf{m})^{1/T_i}p(\mathbf{m})$ and swap states with probability $\min\!\big(1,\big[\tfrac{L(\mathbf{m}_j)}{L(\mathbf{m}_i)}\big]^{1/T_i-1/T_j}\big)$.
- SVGD: $\mathbf{m}_i \leftarrow \mathbf{m}_i + \epsilon\,\frac1N\sum_{j=1}^N\big[k(\mathbf{m}_j,\mathbf{m}_i)\nabla_{\mathbf{m}_j}\log p(\mathbf{m}_j\mid\mathbf{d}) + \nabla_{\mathbf{m}_j}k(\mathbf{m}_j,\mathbf{m}_i)\big]$.

**Foundational papers.**
- Mosegaard & Tarantola (1995), *JGR* 100(B7):12431–12447.
- Sambridge et al. (2006), *GJI* 167(2):528–542. Trans-dimensional inversion and the evidence.
- Bodin & Sambridge (2009), *GJI* 178(3):1411–1436. rjMCMC tomography.
- Sambridge (2014), *GJI* 196(1):357–374. Parallel tempering.
- Fichtner, Zunino & Gebraad (2019), *GJI* 216(2):1344–1363. HMC for tomography.

**Recent (2021+).**
- Boehm, Gebraad, Zunino & Fichtner (2021), *GJI* 227(2):941–968. Autotuned HMC: L-BFGS mass matrix, orders-of-magnitude gain in effective sample size.
- Zhang & Curtis (2020), *JGR* 125:e2019JB018589. ADVI and SVGD tomography, close to MCMC accuracy at lower cost.
- Gebraad, Boehm & Fichtner (2020), *JGR*, DOI 10.1029/2019JB018428. Elastic FWI with HMC.
- Ghalehnoei (2022), Univ. Calgary PhD thesis. **Probabilistic joint gravity–magnetic inversion with 3-D trans-dimensional earth and noise models.**
- Scalzo et al. (2019), *GMD* 12:2941–2960 (arXiv:1812.00318). Parallel-tempered MCMC with pCN for multi-sensor 3-D inversion (Obsidian). pCN beat random walk.

**Open source.** HMCLab (Gebraad/Fichtner, `larsgeb/hmclab`), Obsidian, BayesBay / Espresso (InLab, UNVERIFIED in this search), PyMC / NumPyro / BlackJAX (generic HMC/NUTS), `rjmcmc` libraries (Sambridge group, UNVERIFIED).

---

### A5. Variational and amortised inference (VI, normalising flows, simulation-based inference)

**Core idea.** Turn sampling into optimisation: choose $q_\phi(\mathbf{m})$ from a flexible family, such as a mean-field Gaussian (ADVI) or a normalising flow, to minimise $\mathrm{KL}(q\Vert p)$. *Amortised* or simulation-based inference (SBI) goes further. It trains a conditional density $q_\phi(\mathbf{m}\mid\mathbf{d})$ on simulated $(\mathbf{m},\mathbf{d})$ pairs, so a new survey is inverted in milliseconds. The catch is that it is only as good as the training prior, and it fails under distribution shift unless corrected with physics (Siahkoohi et al. 2023).

**Key equations.**
- ELBO: $\mathcal{L}(\phi) = \mathbb{E}_{q_\phi}\big[\log p(\mathbf{d}\mid\mathbf{m}) + \log p(\mathbf{m}) - \log q_\phi(\mathbf{m})\big]$.
- Flow: $\mathbf{m} = f_\phi(\mathbf{z})$ with $\mathbf{z}\sim\mathcal{N}(0,I)$ and $\log q_\phi(\mathbf{m}) = \log q_0(\mathbf{z}) - \log\lvert\det \partial f_\phi/\partial\mathbf{z}\rvert$.
- Amortised neural posterior estimation: $\min_\phi\ \mathbb{E}_{(\mathbf{m},\mathbf{d})\sim p(\mathbf{m})p(\mathbf{d}\mid\mathbf{m})}\big[-\log q_\phi(\mathbf{m}\mid\mathbf{d})\big]$.

**Foundational papers.**
- Cranmer, Brehmer & Louppe (2020), *PNAS* 117(48):30055–30062. Review of SBI.
- Zhang & Curtis (2020), *JGR*. VI tomography.
- Zhao, Curtis & Zhang (2022), *GJI* 228(1):213–239. Normalising-flow tomography.
- Zhang & Curtis (2021), *JGR* 126:e2021JB022320. Invertible neural networks.

**Recent (2021+).**
- Siahkoohi, Rizzuti, Orozco & Herrmann (2023), *Geophysics* 88. Amortised VI with physics-based latent correction.
- Rakoczi & Hammond (2023), arXiv:2311.07200. **Normalising flows for Bayesian gravity inversion**, the closest potential-field example.
- Levy, Laloy & Linde (2023), *Computers & Geosciences*. Inverse autoregressive flows with geostatistical priors (DOI not seen).
- Saoulis et al., *GJI*. SBI for full-waveform source inversion, which shows that Gaussian-likelihood assumptions give overconfident posteriors. Year and DOI UNVERIFIED.

**Open source.** `sbi` (Mackelab), `nflows` / `zuko`, Pyro / NumPyro (SVI), SLIM InvertibleNetworks.jl. I found no gravity- or magnetic-specific SBI code; this is a gap.

---

### A6. Learned priors (VAE/GAN latent priors, diffusion priors, deep image prior, supervised DL inversion)

**Core idea.** Replace hand-made regularisation with a prior learned from realistic geological realisations. With a VAE or GAN, the model is $\mathbf{m}=G(\mathbf{z})$ and the inversion or sampling happens in a low-dimensional latent space; Laloy et al. report compression ratios of 200–500. Diffusion models supply a score-based prior that can be combined with a likelihood gradient for posterior sampling. Deep image prior (DIP) needs *no training data*: it reparameterises $\mathbf{m}$ as the output of a randomly initialised CNN and relies on the network's implicit regularisation. Supervised end-to-end networks map data straight to models, which is fast but offers weak guarantees and little UQ.

**Key equations.**
- Latent prior: $p(\mathbf{z}\mid\mathbf{d}) \propto p\big(\mathbf{d}\mid F(G(\mathbf{z}))\big)\,\mathcal{N}(\mathbf{z};0,I)$.
- Diffusion posterior sampling: $\nabla_{\mathbf{m}_t}\log p_t(\mathbf{m}_t\mid\mathbf{d}) \approx s_\theta(\mathbf{m}_t,t) + \nabla_{\mathbf{m}_t}\log p\big(\mathbf{d}\mid \hat{\mathbf{m}}_0(\mathbf{m}_t)\big)$.
- DIP: $\mathbf{m}=f_\theta(\mathbf{z}_0)$ and $\theta^* = \arg\min_\theta \lVert F(f_\theta(\mathbf{z}_0))-\mathbf{d}\rVert^2_{\Gamma^{-1}}$.

**Foundational papers.**
- Laloy et al. (2017), *Adv. Water Resour.* 110:387–405. VAE parameterisation.
- Laloy et al. (2018), *WRR* 54(1):381–406. Spatial GAN with MCMC in latent space.
- Mosser, Dubrule & Blunt (2020), *Math. Geosci.* 52:53–79. GAN prior with MALA for seismic.
- Ulyanov, Vedaldi & Lempitsky, arXiv:1711.10925. Deep image prior. The CVPR 2018 venue is UNVERIFIED; an IJCV version has DOI 10.1007/s11263-020-01303-4.

**Recent (2021+).**
- Zhang et al. (2021/2022), *IEEE TGRS*, DOI 10.1109/TGRS.2021.3110606. Deep learning for 3-D gravity inversion (encoder–decoder, Vinton salt dome).
- DIP-Inv (2023), arXiv:2312.04752. CNN reparameterisation of DC-resistivity inversion with no training data (authors UNVERIFIED).
- Ravasi (2025), arXiv:2501.04881. Measurement-guided diffusion models (DPS/PGDM) for geophysical inverse problems.
- A 2026 preprint, arXiv:2603.19629, studies memorisation in learned (diffusion) priors and shows that a memorised prior collapses to a Gaussian mixture. This is a warning about over-confident posteriors.

**Open source.** `elaloy/SGANinv`, `lukasmosser/stochastic_seismic_waveform_inversion`, DIP reference code (Ulyanov), diffusers / score-SDE codebases (generic).

---

### A7. Ensemble and approximate UQ (EKI, randomise-then-optimise, Laplace, null-space)

**Core idea.** These methods approximate the posterior with *many optimisations*, *an ensemble of forward runs* or *local curvature*, so they can reuse a deterministic inversion engine.
- EKI is derivative-free. It updates an ensemble with sample covariances, and its solution lies in the span of the initial ensemble.
- Randomise-then-optimise (RTO) and randomised maximum likelihood (RML) solve perturbed MAP problems in parallel. RTO adds a Metropolis or importance-sampling correction.
- The Laplace approximation uses a Gaussian built on a low-rank Hessian at the MAP point.
- Null-space shuttles explore models that fit the data equally well without full sampling.

**Key equations.**
- EKI: $\mathbf{m}_j^{n+1} = \mathbf{m}_j^n + \mathbf{C}^{md}_n\big(\mathbf{C}^{dd}_n+\Gamma\big)^{-1}\big(\mathbf{d}+\boldsymbol{\eta}_j - F(\mathbf{m}^n_j)\big)$.
- Randomised MAP (RML-style; RTO adds a projection and an MH/IS weight): $\mathbf{m}_j = \arg\min_{\mathbf{m}} \lVert F(\mathbf{m})-(\mathbf{d}+\boldsymbol{\epsilon}_j)\rVert^2_{\Gamma^{-1}} + \lVert \mathbf{m}-(\mathbf{m}_0+\boldsymbol{\delta}_j)\rVert^2_{\mathbf{C}^{-1}}$.
- Laplace: $p(\mathbf{m}\mid\mathbf{d})\approx\mathcal{N}\big(\mathbf{m}_{\text{MAP}},\,(\mathbf{J}^\top\Gamma^{-1}\mathbf{J}+\mathbf{C}^{-1})^{-1}\big)$, with the low-rank form $\mathbf{C}^{1/2}(\mathbf{I}-\mathbf{V}_r\mathbf{D}_r\mathbf{V}_r^\top)\mathbf{C}^{1/2}$ and $\mathbf{D}_r=\mathrm{diag}\big(\lambda_i/(1+\lambda_i)\big)$.
- Null-space shuttle (linear case): $\mathbf{m}' = \mathbf{m}_{\text{MAP}} + \mathbf{P}_{\mathcal{N}}\,\delta\mathbf{m}$, with $\mathbf{P}_{\mathcal{N}}$ the projector onto the (effective) null space.

**Foundational papers.**
- Deal & Nolet (1996), *GJI* 124(2):372–380. Null-space shuttles.
- Iglesias, Law & Stuart (2013), *Inverse Problems* 29(4):045001. Ensemble Kalman inversion.
- Bui-Thanh, Ghattas, Martin & Stadler (2013), *SIAM J. Sci. Comput.* 35(6):A2494–A2523. Low-rank Laplace approximation at scale.
- Bardsley, Solonen, Haario & Laine (2014), *SIAM J. Sci. Comput.* 36(4):A1895–A1910. Randomise-then-optimise.
- Fichtner & Zunino (2019), *GRL* 46:644–651. Hamiltonian null-space shuttles, demonstrated on a magnetic example.

**Recent (2021+).**
- Tso et al. (2021), *GJI* 225(2):887–905. EKI with level sets for ERT.
- A 2024 *GJI* paper (vol. 236) applies EKI with level sets to **induced polarisation** data, which bears directly on a third property (Tso et al., authors partly verified).
- Boehm et al. (2021), *GJI*. Generalised null-space exploration.
- A 2025 *Scientific Reports* paper on regularised EKI for gravity (PMC12689625; authors UNVERIFIED).

**Open source.** hIPPYlib (Laplace and low-rank Hessian; UNVERIFIED in this search), `EnsembleKalmanProcesses.jl` (UNVERIFIED), pyGIMLi + EKI scripts (Tso), HMCLab (null-space shuttle).

---

### A8. Large-scale computation (compression, octree, matrix-free, GPU, equivalent sources, distributed)

**Core idea.** The dense potential-field sensitivity $\mathbf{G}\in\mathbb{R}^{N\times M}$ is the bottleneck. There are several ways to break it:
- compress it (wavelets: Li & Oldenburg 2003);
- shrink $M$ (adaptive octree meshes: Davis & Li 2011; Haber & Heldmann 2007 for EM);
- never store it (on-the-fly or matrix-free products on GPU);
- shrink $N$ or regularise the data first (equivalent sources: Soler & Uieda 2021);
- distribute it (Dask: SimPEG).

Every UQ method in sections A4–A7 inherits this cost per forward or adjoint call, multiplied by the number of samples, particles or ensemble members.

**Key equations.**
- $\mathbf{d}=\mathbf{G}\mathbf{m}$, with dense memory $8NM$ bytes.
- Wavelet compression: $\tilde{\mathbf{G}} = \mathcal{T}_\tau(\mathbf{G}\mathbf{W}^\top)$ and $\mathbf{G}\mathbf{m}\approx\tilde{\mathbf{G}}(\mathbf{W}\mathbf{m})$, keeping about $1$–$5\%$ of the coefficients.
- Equivalent sources: $\mathbf{d}\approx\mathbf{A}\mathbf{c}$ with sources fitted window by window (gradient boosting) at $\mathcal{O}(\text{window})$ memory.

**Foundational papers.**
- Li & Oldenburg (1996), *Geophysics* 61(2):394–408. 3-D magnetic inversion with depth weighting.
- Li & Oldenburg (2003), *GJI* 152(2):251–265. Wavelet compression with a log-barrier method.
- Haber & Heldmann (2007), *J. Comput. Phys.* Octree multigrid for quasi-static Maxwell's equations (volume and DOI UNVERIFIED).
- Davis & Li (2011), *GJI* 185:157–166. Adaptive mesh, space-filling curves and wavelets.
- Cockett et al. (2015), *Computers & Geosciences*. SimPEG.
- Heagy et al. (2017), *Computers & Geosciences* 107:1–19. SimPEG EM framework.

**Recent (2021+).**
- Soler & Uieda (2021), *GJI* 227(3):1768–1783. Gradient-boosted equivalent sources.
- Vatankhah et al. (2022), *GJI*. Large-scale Gramian joint inversion using reweighted CG.
- A GPU joint FTG inversion at the million-cell scale with compressed sensitivity, *IEEE TGRS*, DOI 10.1109/TGRS.2022.3147028 (authors UNVERIFIED).
- Dask-parallel SimPEG potential fields (ParCFD 2020 paper; Mira Geoscience `simpeg-drivers`, octree meshes).

**Open source.** SimPEG + discretize (octree, Dask), Harmonica / Fatiando (equivalent sources, prism/tesseroid forward), Tomofast-x (MPI, wavelet compression), pyGIMLi.

---

### A9. Remanent magnetisation (cross-cutting; it decides how the magnetic leg can be coupled)

**Core idea.** Strong remanence breaks the induced-only assumption $\mathbf{M}=\kappa\mathbf{H}_0$ of Li & Oldenburg (1996). There are two standard responses:
1. **MVI**: invert for a 3-component magnetisation vector per cell, in Cartesian or sparse spherical coordinates.
2. **Amplitude inversion**: invert $\lvert\mathbf{B}\rvert$, which is only weakly sensitive to magnetisation direction, for an effective susceptibility.

For joint inversion, a *scalar* that is safe to couple is the magnetisation amplitude $\lVert\mathbf{M}\rVert$. Coupling raw susceptibility to density is biased whenever remanence is present.

**Key equations.**
- MVI: $\mathbf{d}_m = \sum_{c\in\{x,y,z\}}\mathbf{G}_c\,\mathbf{M}_c$, with $\mathbf{M}=\kappa_{\text{eff}}\hat{\mathbf{H}}_0 + \mathbf{M}_\perp$.
- Amplitude: $\lvert\mathbf{B}\rvert = \sqrt{B_x^2+B_y^2+B_z^2}$, computed from TMI through equivalent sources.

**Papers.**
- Lelièvre & Oldenburg (2009), *Geophysics* 74(3):L21–L30. Total magnetisation vector inversion. The actual title is "A 3D total magnetization inversion applicable when significant, complicated remanence is present."
- Li, Shearer, Haney & Dannemiller (2010), *Geophysics* 75(1):L1–L11. Amplitude inversion and related approaches.
- Ellis, de Wet & Macleod (2012), ASEG Extended Abstracts. MVI, the basis of the commercial Geosoft method.
- Fournier, Heagy & Oldenburg (2020), *Geophysics* 85(3):J33–J49. Sparse MVI in spherical coordinates. This is the SimPEG implementation.

---

## B. Ranking table (scores are ASSESSMENTS)

Scale is 1 (poor) to 5 (best) in every column; higher is always better.
- **Assumptions:** 5 = mild or few assumptions.
- **UQ:** quality and calibration of the uncertainty the method yields.
- **Cost@1e6:** feasibility of time and memory with $10^6$ cells and $10^4$–$10^5$ data.
- **+3rd prop:** ease of adding IP, EM or seismic.
- **Remanence:** robustness when magnetic remanence is strong.
- **Maturity:** field use, documentation and open code.

| # | Method | Assump. | UQ | Cost@1e6 | +3rd prop | Remanence | Maturity | Total /30 |
|---|---|---|---|---|---|---|---|---|
| 1 | Cross-gradient (det.) | 4 | 1 | 4 | 3 | 4 | 5 | 21 |
| 2 | Joint total variation | 4 | 1 | 4 | 5 | 4 | 3 | 21 |
| 3 | Gramian | 3 | 1 | 4 | 5 | 3 | 3 | 19 |
| 4 | PGI (GMM, det.) | 3 | 2 | 4 | 5 | 2 | 4 | 20 |
| 5 | Fuzzy c-means | 3 | 1 | 4 | 4 | 2 | 3 | 17 |
| 6 | Explicit petrophysical relation | 1 | 1 | 5 | 3 | 1 | 4 | 15 |
| 7 | Level-set / parametric shape | 2 | 3 | 4 | 4 | 3 | 3 | 19 |
| 8 | Implicit-geomodel Bayesian (GemPy/Loop + MCMC/HMC) | 2 | 4 | 3 | 5 | 3 | 2 | 19 |
| 9 | Random-walk / pCN MCMC + parallel tempering (voxel) | 4 | 5 | 1 | 3 | 3 | 4 | 20 |
| 10 | HMC (voxel, adjoint gradients) | 4 | 5 | 2 | 3 | 3 | 3 | 20 |
| 11 | rjMCMC (trans-dimensional) | 5 | 5 | 1 | 2 | 3 | 3 | 19 |
| 12 | SVGD / particle VI | 4 | 3 | 2 | 3 | 3 | 2 | 17 |
| 13 | VI / normalising flow (per survey) | 3 | 3 | 2 | 3 | 3 | 2 | 16 |
| 14 | Amortised SBI / conditional NF | 2 | 3 | 3 | 2 | 2 | 1 | 13 |
| 15 | VAE/GAN latent prior + MCMC | 2 | 3 | 3 | 3 | 2 | 2 | 15 |
| 16 | Diffusion prior (DPS) | 2 | 3 | 2 | 3 | 2 | 1 | 13 |
| 17 | Deep image prior | 4 | 1 | 3 | 4 | 3 | 1 | 16 |
| 18 | Supervised DL end-to-end | 1 | 1 | 5 | 2 | 1 | 2 | 12 |
| 19 | Ensemble Kalman inversion | 3 | 3 | 3 | 4 | 3 | 3 | 19 |
| 20 | Randomise-then-optimise / RML | 3 | 4 | 3 | 5 | 4 | 3 | 22 |
| 21 | Low-rank Laplace | 2 | 3 | 4 | 4 | 3 | 3 | 19 |
| 22 | Null-space shuttle | 4 | 2 | 4 | 3 | 3 | 2 | 18 |

**One-line reasons (ASSESSMENT).**
1. **Cross-gradient.** Only assumes co-located boundaries. No UQ. Pairwise terms grow as $\binom{K}{2}$ and suffer from gradient degeneracy in smooth regions. MVI amplitude couples cleanly. Most widely used, with code in Tomofast-x, jif3D and SimPEG.
2. **JTV.** Similar assumptions; convex. A single term handles any $K$, so adding a third property is trivial. Less field history than cross-gradient.
3. **Gramian.** Can impose a correlation that does not exist. Generalises naturally to $K$ properties. Mostly the Zhdanov/TechnoImaging line and Vatankhah's large-scale work.
4. **PGI.** Needs a unit count and petrophysical samples. Gives unit-membership probabilities but not a calibrated posterior. Scales like a Tikhonov inversion. The GMM extends directly to 3-D property vectors. Scalar $\kappa$ is biased under remanence unless $\lVert\mathbf{M}\rVert$ is used. Mature in SimPEG.
5. **FCM.** Like PGI but non-probabilistic: memberships are not likelihoods. Fewer public implementations.
6. **Explicit relation.** Fragile, and fails silently when the relation does not hold (Moorkamp 2011). Remanence breaks density–susceptibility relations.
7. **Level sets.** Assume a discrete number of units with known properties. A low-dimensional state can be sampled. Topology changes are awkward.
8. **Implicit geomodel Bayesian.** Assumes the unit topology and stratigraphy. Best-calibrated *geological* uncertainty. Each forward model is fast but needs geology-to-mesh resampling. One geometry feeds every physics, so the third property is close to free. Research-grade.
9. **MCMC + PT (voxel).** Gold-standard UQ, but $10^5$–$10^7$ forward calls on $10^6$ cells is impractical without reduction. Fine on reduced parameterisations.
10. **HMC.** Gradients make it far better than random walk, and autotuning helps further, but it is still costly at $10^6$ cells. Each property adds an adjoint.
11. **rjMCMC.** Weakest assumptions (parsimony emerges from the data). Prohibitive in 3-D voxels. Cross-physics trans-dimensional moves are complex.
12. **SVGD.** Kernel collapse in high dimensions underestimates variance. 50–500 particles, each a full gradient.
13. **Per-survey VI / normalising flow.** Flow capacity limits $10^6$-cell posteriors. Mean-field ADVI under-disperses.
14. **Amortised SBI.** Inherits the training prior and is brittle under distribution shift. Training needs a huge number of 3-D simulations. Little potential-field literature.
15. **VAE/GAN prior.** Prior misspecification is the main risk. Latent MCMC is cheap. 3-D training sets are scarce in Indian geological settings.
16. **Diffusion prior.** Approximate likelihood guidance, plus memorisation risk. Expensive 3-D score networks. Very new for geophysics.
17. **DIP.** Implicit bias only; a multi-channel output head gives joint properties. No UQ unless ensembled.
18. **Supervised DL.** Strong train/test assumptions, little physics consistency, essentially no UQ; inference is instant.
19. **EKI.** Gaussian-ish updates and ensemble-subspace collapse. Derivative-free, so any black-box third solver works. Proven for ERT and IP.
20. **RTO / RML.** Exact only for linear-Gaussian problems, approximate otherwise. Embarrassingly parallel MAP solves that reuse the existing coupled objective, with an MH/IS correction available. Works with MVI.
21. **Laplace.** Local Gaussian only, poor for multimodal posteriors. A low-rank Hessian is feasible at $10^6$ cells.
22. **Null-space shuttle.** Explores equivalent models rather than giving probabilities. Cheap and deterministic.

**Ranking headline.** No single method wins every column. Petrophysical/structural coupling gives the *physics link*, sampling or ensembles give the *UQ*, and compression or reduced parameterisations make it *affordable*. The best designs combine one from each group (Section C).

---

## C. Top-3 shortlist with experiment plans

*Assumed context (not confirmed with Alt Carbon): exploration for mafic/ultramafic rocks suitable for CO₂ mineralisation (cf. Heagy et al. 2022, arXiv:2203.13894). The third property is most likely conductivity or chargeability from DC/IP or EM (serpentinisation, magnetite, sulphides), or $v_p$ from seismic refraction.*

### C1. Three-property PGI, made probabilistic with randomise-then-optimise

**Why.** It extends the gravity–magnetic GMM coupling directly to a 3-vector $[\rho, \lVert\mathbf{M}\rVert, \log\sigma\ \text{or}\ \eta]$ per cell. It reuses SimPEG at $10^6$ cells. Wrapping it in RTO/RML adds calibrated-ish UQ through parallel perturbed MAP solves.

**Experiment.**
1. Build a synthetic 3-D ultramafic intrusion with 4 units: host basalt, serpentinite, peridotite, and remanent magnetite-rich gabbro. Simulate gravity, TMI and DC/IP on an octree mesh ($\sim10^6$ cells). SimPEG + discretize.
2. Compare four configurations: (a) separate Tikhonov; (b) two-property PGI (gravity + magnetic); (c) three-property PGI with $\lVert\mathbf{M}\rVert$ from MVI; (d) the same with raw $\kappa$, to quantify remanence bias.
3. Run 32–64 RTO/RML solves on a Dask cluster. Compute the posterior mean and std per cell and the unit-membership probability.
4. **Metrics:** unit classification accuracy (confusion matrix), property RMSE, 90% interval coverage of true values, volume-of-reactive-rock estimate with credible interval, and wall time and memory.
5. **Go/no-go:** three-property PGI improves classification of the serpentinite unit by at least 15% over two-property PGI, and coverage stays between 0.8 and 0.95.

### C2. Implicit geological model as the joint parameterisation, sampled with HMC or parallel-tempered pCN

**Why.** Every physics is coupled through one geometry, which makes adding a property essentially free. Dimension drops to about $10^2$–$10^3$, so full MCMC with calibrated *geometric* UQ becomes feasible. This matches the Alt Carbon question "how much reactive rock, and where are its boundaries?"

**Experiment.**
1. Build the geology in GemPy (autodiff) or LoopStructural from a few interface points and orientations, plus per-unit property distributions $\mathcal{N}(\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)$ for $(\rho,\lVert\mathbf{M}\rVert,\sigma)$.
2. Run forward models: prism gravity and magnetics (Harmonica or SimPEG) and DC/IP or EM (SimPEG) on a fixed octree. Use anti-aliased unit-to-mesh assignment (the Blockworlds lesson).
3. Sample with (a) parallel-tempered pCN (Obsidian-style) and (b) NUTS/HMC through JAX or PyTorch autodiff. Compare effective sample size per CPU-hour.
4. **Metrics:** posterior of contact depth and dip, posterior volume of the target unit, $\hat{R}$, and ESS. Also run a misspecification test in which the true model has an extra unit missing from the parameterisation.
5. **Deliverable:** probability-of-unit maps and volume distributions, which feed directly into CO₂-storage capacity estimates.

### C3. Structurally coupled voxel inversion (JTV or cross-gradient) with ensemble UQ (EKI vs SVGD vs low-rank Laplace)

**Why.** This is the fallback when petrophysics is poorly known; structure-only coupling is the safest assumption. The experiment benchmarks the three cheapest scalable UQ routes against a small-scale reference posterior.

**Experiment.**
1. Use the same synthetic as C1, but give the conductivity–density relation *no* consistent pattern, to stress-test petrophysical methods.
2. Joint objective with JTV across $(\rho, \lVert\mathbf{M}\rVert, \log\sigma)$, solved with SimPEG, using on-the-fly or compressed sensitivities.
3. Compute UQ three ways: EKI (derivative-free, 100–200 members), SVGD (50–100 particles, adjoint gradients) and low-rank Laplace at the MAP point (randomised eigensolver, $r\approx200$–$500$).
4. Reference posterior: on a $64^3$ downscaled version, run long HMC (HMCLab) as ground truth. Compare marginal variances, KL or Wasserstein distance per cell, and calibration.
5. **Metrics:** variance ratio against HMC (target 0.7–1.3), compute cost (forward-equivalents), and how the error grows from $64^3$ to $10^6$ cells.

*Watch-list (not top-3 yet):* VAE/GAN or diffusion latent priors trained on GemPy/Loop ensembles. These are promising for speed, but the 3-D potential-field evidence is thin and there is a memorisation and prior-misspecification risk.

---

## D. Five open research gaps

1. **Calibrated UQ for 3-property joint inversion at $10^6$ cells.** No published work combines petrophysical (GMM) coupling, ≥3 physics, remanence-aware magnetics and a validated posterior at production scale. PGI is deterministic; Ghalehnoei's (2022) Bayesian gravity–magnetic work is two-physics.
2. **Probabilistic coupling under remanence.** The joint posterior over $(\rho,\mathbf{M}\ \text{vector},\sigma)$ is non-Gaussian. Amplitude/MVI parameterisations add non-linearity, and no standard prior on remanence direction exists for joint Bayesian schemes.
3. **Uncertain coupling strength and form.** The weights $\lambda$, the GMM component count and the relevance of the cross-gradient term are tuned by hand. Hierarchical Bayes or evidence-based model selection between coupling types (structural, petrophysical, none) is largely unexplored for potential fields. The evidence itself is contested (Mosegaard 2023, arXiv:2308.05858; not read in full).
4. **Bridging geology-parameterised and voxel inversions.** Implicit-model posteriors are well calibrated but misspecified when unit topology is wrong. Voxel posteriors are flexible but poorly calibrated. Hybrid "geometry + residual field" models, with consistent likelihoods (Blockworlds aliasing) and gradients through the geomodeller, are an open design space.
5. **Learned priors and SBI for potential fields with distribution-shift guarantees.** Gravity and magnetic SBI or diffusion work is nearly absent (Rakoczi & Hammond 2023 is an exception). Methods are needed for physics-corrected amortisation (à la Siahkoohi et al. 2023), memorisation diagnostics, and training-set design for under-mapped terranes such as Indian cratons and the Deccan basalts.

---

## E. Bibliography

**Tags.**
- **VERIFIED**: authors, year, title, venue and DOI or arXiv ID all appeared in a search result seen during compilation.
- **PARTIAL**: the core record was seen, but some field (named) was not confirmed.
- **UNVERIFIED**: not confirmed. Do not cite without checking.

DOIs are given only where one appeared in a search result.

### Structural coupling
1. Haber, E. & Oldenburg, D.W. (1997). Joint inversion: a structural approach. *Inverse Problems* 13(1), 63–77. doi:10.1088/0266-5611/13/1/006. **VERIFIED**
2. Gallardo, L.A. & Meju, M.A. (2003). Characterization of heterogeneous near-surface materials by joint 2D inversion of dc resistivity and seismic data. *GRL* 30(13), 1658. doi:10.1029/2003GL017370. **VERIFIED**
3. Gallardo, L.A. & Meju, M.A. (2004). Joint two-dimensional DC resistivity and seismic travel time inversion with cross-gradients constraints. *JGR* 109, B03311. doi:10.1029/2003JB002716. **VERIFIED**
4. Fregoso, E. & Gallardo, L.A. (2009). Cross-gradients joint 3D inversion with applications to gravity and magnetic data. *Geophysics* 74(4), L31–L42. doi:10.1190/1.3119263. **VERIFIED**
5. Gallardo, L.A. & Meju, M.A. (2011). Structure-coupled multiphysics imaging in geophysical sciences. *Rev. Geophys.* 49. doi:10.1029/2010RG000330. **VERIFIED**
6. Moorkamp, M., Heincke, B., Jegen, M., Roberts, A.W. & Hobbs, R.W. (2011). A framework for 3-D joint inversion of MT, gravity and seismic refraction data. *GJI* 184(1), 477–493. doi:10.1111/j.1365-246X.2010.04856.x. **VERIFIED**
7. Zhdanov, M.S., Gribenko, A. & Wilson, G. (2012). Generalized joint inversion of multimodal geophysical data using Gramian constraints. *GRL* 39, L09301. doi:10.1029/2012GL051233. **VERIFIED**
8. Haber, E. & Holtzman Gazit, M. (2013). Model fusion and joint inversion. *Surv. Geophys.* 34, 675–695. doi:10.1007/s10712-013-9232-4. **VERIFIED**
9. Meju, M.A. & Gallardo, L.A. (2016). Structural coupling approaches in integrated geophysical imaging. In *Integrated Imaging of the Earth*, AGU Geophys. Monogr. 218, ch. 4. DOI not seen. **PARTIAL**
10. Vatankhah, S., Renaut, R.A., Huang, X., Mickus, K. & Gharloghi, M. (2022). Large-scale focusing joint inversion of gravity and magnetic data with Gramian constraint. *GJI* 230(3), 1585–1611. doi:10.1093/gji/ggac138; arXiv:2112.15562. **VERIFIED**
11. Vatankhah, S. et al. (2020 preprint). Generalized Lp-norm joint inversion of gravity and magnetic data using cross-gradient constraint. arXiv:2001.03579. **PARTIAL** (full author list and journal version not confirmed)
12. Anonymous review (2024). Gramian coupling terms in joint inversion. *Surv. Geophys.* doi:10.1007/s10712-024-09832-0. **UNVERIFIED** (authors and title not seen)

### Petrophysical coupling
13. Bosch, M. & McGaughey, J. (2001). Joint inversion of gravity and magnetic data under lithologic constraints. *The Leading Edge* 20(8). doi:10.1190/1.1487299. **VERIFIED**
14. Bosch, M. et al. (2006). Joint gravity and magnetic inversion in 3D using Monte Carlo methods. *Geophysics*. doi:10.1190/1.2209952. **PARTIAL** (co-authors and pages not seen)
15. Sun, J. & Li, Y. (2016). Joint inversion of multiple geophysical data using guided fuzzy c-means clustering. *Geophysics* 81(3), ID37–ID57. doi:10.1190/geo2015-0457.1. **VERIFIED**
16. Sun, J. & Li, Y. (2017). Joint inversion of multiple geophysical and petrophysical data using generalized fuzzy clustering algorithms. *GJI* 208(2), 1201–1216. doi:10.1093/gji/ggw442. **VERIFIED**
17. Astic, T. & Oldenburg, D.W. (2019). A framework for petrophysically and geologically guided geophysical inversion using a dynamic Gaussian mixture model prior. *GJI* 219(3), 1989–2012. doi:10.1093/gji/ggz389. **VERIFIED**
18. Astic, T., Heagy, L.J. & Oldenburg, D.W. (2020). Petrophysically and geologically guided multi-physics inversion using a dynamic Gaussian mixture model. *GJI* 224(1), 40–68. doi:10.1093/gji/ggaa378; arXiv:2002.09515. **VERIFIED** (the brief's "2021" should read 2020)
19. [Astic, T. et al.] Joint inversion of potential-fields data over the DO-27 kimberlite pipe using a Gaussian mixture model prior. *Interpretation*. doi:10.1190/int-2019-0283.1. **PARTIAL** (authors and year not confirmed)
20. Heagy, L.J., Astic, T., Capriotti, J., Weis, J. & Oldenburg, D.W. (2022). Geophysical inversions to delineate rocks with CO₂ sequestration potential through carbon mineralization. arXiv:2203.13894. **PARTIAL** (author list from a mirror)

### Geology-based parameterisations
21. Lelièvre, P.G., Farquharson, C.G. & Hurich, C.A. (2012). Joint inversion of seismic traveltimes and gravity data on unstructured grids with application to mineral exploration. *Geophysics*. doi:10.1190/1.3513182. **PARTIAL** (volume and pages not seen)
22. Zheglova, P., Farquharson, C.G. & Hurich, C.A. (2013). 2-D reconstruction of boundaries with level set inversion of traveltimes. *GJI* 192(2), 688–698. doi:10.1093/gji/ggs035. **VERIFIED**
23. Zheglova, P., Lelièvre, P.G. & Farquharson, C.G. (2018). Multiple level-set joint inversion of traveltime and gravity data with application to ore delineation: a synthetic study. *Geophysics* 83(1), R13–R30. doi:10.1190/geo2016-0675.1. **VERIFIED**
24. Wellmann, F. & Caumon, G. (2018). 3-D structural geological models: concepts, methods, and uncertainties. *Adv. Geophys.* 59, 1–121. doi:10.1016/bs.agph.2018.09.001. **VERIFIED**
25. Wellmann, F., de la Varga, M., Murdie, R.E., Gessner, K. & Jessell, M. (2018). Uncertainty estimation for a geological model of the Sandstone greenstone belt, Western Australia: insights from integrated geological and geophysical inversion in a Bayesian inference framework. *Geol. Soc. London Spec. Publ.* 453, 41–56. DOI not seen. **PARTIAL**
26. de la Varga, M., Schaaf, A. & Wellmann, F. (2019). GemPy 1.0: open-source stochastic geological modeling and inversion. *GMD* 12, 1–32. doi:10.5194/gmd-12-1-2019. **VERIFIED**
27. Giraud, J., Lindsay, M., Ogarko, V., Jessell, M., Martin, R. & Pakyuz-Charrier, E. (2019). Integration of geoscientific uncertainty into geophysical inversion by means of local gradient regularization. *Solid Earth* 10(1), 193–. DOI not displayed (URL se-10-193-2019). **PARTIAL**
28. Giraud, J., Ogarko, V., Lindsay, M., Pakyuz-Charrier, E., Jessell, M. & Martin, R. (2019). Sensitivity of constrained joint inversions to geological and petrophysical input data uncertainties with posterior geological analysis. *GJI* 218(1), 666–688. DOI not seen. **PARTIAL**
29. Giraud, J., Lindsay, M., Jessell, M. & Ogarko, V. (2020). Towards plausible lithological classification from geophysical inversion: honouring geological principles in subsurface imaging. *Solid Earth* 11, 419–436. doi:10.5194/se-11-419-2020. **VERIFIED**
30. Giraud, J., Ogarko, V., Martin, R., Jessell, M. & Lindsay, M. (2021). Structural, petrophysical, and geological constraints in potential field inversion using the Tomofast-x v1.0 open-source code. *GMD* 14, 6681–6709. doi:10.5194/gmd-14-6681-2021. **VERIFIED**
31. Grose, L., Ailleres, L., Laurent, G. & Jessell, M. (2021). LoopStructural 1.0: time-aware geological modelling. *GMD* 14, 3915–3937. doi:10.5194/gmd-14-3915-2021. **VERIFIED**
32. Scalzo, R., Lindsay, M., Jessell, M., Pirot, G., Giraud, J., Cripps, E. & Cripps, S. (2022). Blockworlds 0.1.0 [full subtitle not seen]. *GMD* 15(9), 3641–3662. doi:10.5194/gmd-15-3641-2022. **VERIFIED** (subtitle not captured)
33. Giraud, J., Caumon, G., Grose, L., Ogarko, V. & Cupillard, P. (2024). Integration of automatic implicit geological modelling in deterministic geophysical inversion. *Solid Earth* 15, 63–89. doi:10.5194/se-15-63-2024. **VERIFIED**
34. Liang, Z. (2023). Derivative-informed Bayesian inference for trainable geological modeling in modern machine-learning framework. PhD thesis, RWTH Aachen (advisers Wellmann and Ghattas). **PARTIAL** (no DOI)
35. Liang, Z. & Wellmann, F. (2021). Hessian-informed MCMC (gpCN) for 3-D model-based gravity Bayesian inversion. **UNVERIFIED** (exact title and venue not seen)

### Bayesian sampling
36. Mosegaard, K. & Tarantola, A. (1995). Monte Carlo sampling of solutions to inverse problems. *JGR* 100(B7), 12431–12447. doi:10.1029/94JB03097. **VERIFIED**
37. Sambridge, M., Gallagher, K., Jackson, A. & Rickwood, P. (2006). Trans-dimensional inverse problems, model comparison and the evidence. *GJI* 167(2), 528–542. DOI not seen. **PARTIAL**
38. Bodin, T. & Sambridge, M. (2009). Seismic tomography with the reversible jump algorithm. *GJI* 178(3), 1411–1436. doi:10.1111/j.1365-246X.2009.04226.x. **VERIFIED**
39. Sambridge, M. (2014). A parallel tempering algorithm for probabilistic sampling and multimodal optimization. *GJI* 196(1), 357–374. doi:10.1093/gji/ggt342. **VERIFIED**
40. Fichtner, A., Zunino, A. & Gebraad, L. (2019). Hamiltonian Monte Carlo solution of tomographic inverse problems. *GJI* 216(2), 1344–1363. doi:10.1093/gji/ggy496. **VERIFIED**
41. Scalzo, R., Kohn, D., Olierook, H., Houseman, G., Chandra, R., Girolami, M. & Cripps, S. (2019). Efficiency and robustness in Monte Carlo sampling for 3-D geophysical inversions with Obsidian v0.1.2: setting up for success. *GMD* 12(7), 2941–2960. arXiv:1812.00318. **VERIFIED** (DOI not displayed; Copernicus URL gmd-12-2941-2019)
42. Gebraad, L., Boehm, C. & Fichtner, A. (2020). Bayesian elastic full-waveform inversion using Hamiltonian Monte Carlo. *JGR Solid Earth*. doi:10.1029/2019JB018428. **VERIFIED** (volume not checked)
43. Zhang, X. & Curtis, A. (2020). Seismic tomography using variational inference methods. *JGR Solid Earth* 125, e2019JB018589. doi:10.1029/2019JB018589; arXiv:1908.08356. **VERIFIED**
44. Boehm, C., Gebraad, L., Zunino, A. & Fichtner, A. (2021). Autotuning Hamiltonian Monte Carlo for efficient generalized nullspace exploration. *GJI* 227(2), 941–968. doi:10.1093/gji/ggab270. **VERIFIED**
45. Ghalehnoei, E. (2022). Probabilistic joint inversion of gravity and magnetic data with 3D trans-dimensional earth and noise models. PhD thesis, University of Calgary (PRISM repository). **PARTIAL** (author name spelled "Ghaleh Noei" in the search result)
46. Moyer, [first name not seen] (2023/2024). Improvements to a probabilistic joint inversion of magnetic and gravimetric data. Krell Institute CSGF abstract. **UNVERIFIED** (full name and venue)

### Variational / amortised / SBI
47. Cranmer, K., Brehmer, J. & Louppe, G. (2020). The frontier of simulation-based inference. *PNAS* 117(48), 30055–30062. doi:10.1073/pnas.1912789117; arXiv:1911.01429. **VERIFIED**
48. Zhang, X. & Curtis, A. (2021). Bayesian geophysical inversion using invertible neural networks. *JGR Solid Earth* 126, e2021JB022320. doi:10.1029/2021JB022320. **VERIFIED**
49. Zhao, X., Curtis, A. & Zhang, X. (2022). Bayesian seismic tomography using normalizing flows. *GJI* 228(1), 213–239. doi:10.1093/gji/ggab298. **VERIFIED**
50. Siahkoohi, A., Rizzuti, G., Orozco, R. & Herrmann, F.J. (2023). Reliable amortized variational inference with physics-based latent distribution correction. *Geophysics* 88. doi:10.1190/geo2022-0472.1; arXiv:2207.11640. **VERIFIED**
51. Rakoczi, H. & Hammond, G. (2023). Normalising flows for Bayesian gravity inversion. arXiv:2311.07200. **PARTIAL** (first names and the final title capitalisation not confirmed)
52. Levy, S., Laloy, E. & Linde, N. (2023). Variational Bayesian inference with complex geostatistical priors using inverse autoregressive flows. *Computers & Geosciences*. **PARTIAL** (DOI and first names not seen)
53. Saoulis, A. et al. Full-waveform earthquake source inversion using simulation-based inference. *GJI*; arXiv:2410.23238. **PARTIAL** (year, volume and DOI not seen)

### Learned priors
54. Laloy, E., Hérault, R., Lee, J., Jacques, D. & Linde, N. (2017). Inversion using a new low-dimensional representation of complex binary geological media based on a deep neural network. *Adv. Water Resour.* 110, 387–405. doi:10.1016/j.advwatres.2017.09.029; arXiv:1710.09196. **VERIFIED**
55. Laloy, E., Hérault, R., Jacques, D. & Linde, N. (2018). Training-image based geostatistical inversion using a spatial generative adversarial neural network. *WRR* 54(1), 381–406. doi:10.1002/2017WR022148; arXiv:1708.04975. **VERIFIED**
56. Ulyanov, D., Vedaldi, A. & Lempitsky, V. Deep image prior. arXiv:1711.10925 (2017). IJCV version doi:10.1007/s11263-020-01303-4. **PARTIAL** (CVPR 2018 venue UNVERIFIED)
57. Mosser, L., Dubrule, O. & Blunt, M.J. (2020). Stochastic seismic waveform inversion using generative adversarial networks as a geological prior. *Math. Geosci.* 52, 53–79. doi:10.1007/s11004-019-09832-6; arXiv:1806.03720. **VERIFIED**
58. Zhang, [initials not seen] et al. (2021/2022). Deep learning for 3-D inversion of gravity data. *IEEE TGRS*. doi:10.1109/TGRS.2021.3110606. **PARTIAL** (author list not seen)
59. [Authors not seen] (2023). A test-time learning approach to reparameterize the geophysical inverse problem with a convolutional neural network (DIP-Inv). arXiv:2312.04752. **PARTIAL** (authors UNVERIFIED)
60. Ravasi, M. (2025). Solving geophysical inverse problems with measurement-guided diffusion models. arXiv:2501.04881. **PARTIAL** (single-author attribution from the search summary)
61. [Authors not seen] (2026). On the role of memorization in learned priors for geophysical inverse problems. arXiv:2603.19629. **PARTIAL**

### Ensemble / approximate UQ
62. Deal, M.M. & Nolet, G. (1996). Nullspace shuttles. *GJI* 124(2), 372–380. doi:10.1111/j.1365-246X.1996.tb07027.x. **VERIFIED**
63. Iglesias, M.A., Law, K.J.H. & Stuart, A.M. (2013). Ensemble Kalman methods for inverse problems. *Inverse Problems* 29(4), 045001. doi:10.1088/0266-5611/29/4/045001; arXiv:1209.2736. **VERIFIED**
64. Bui-Thanh, T., Ghattas, O., Martin, J. & Stadler, G. (2013). A computational framework for infinite-dimensional Bayesian inverse problems. Part I: the linearized case, with application to global seismic inversion. *SIAM J. Sci. Comput.* 35(6), A2494–A2523. doi:10.1137/12089586X; arXiv:1308.1313. **VERIFIED**
65. Bardsley, J.M., Solonen, A., Haario, H. & Laine, M. (2014). Randomize-then-optimize: a method for sampling from posterior distributions in nonlinear inverse problems. *SIAM J. Sci. Comput.* 36(4), A1895–A1910. doi:10.1137/140964023. **VERIFIED**
66. Fichtner, A. & Zunino, A. (2019). Hamiltonian nullspace shuttles. *GRL* 46, 644–651. doi:10.1029/2018GL080931. **VERIFIED**
67. Tso, C.-H.M., Iglesias, M., Wilkinson, P., Kuras, O., Chambers, J. & Binley, A. (2021). Efficient multiscale imaging of subsurface resistivity with uncertainty quantification using ensemble Kalman inversion. *GJI* 225(2), 887–905. doi:10.1093/gji/ggab013. **VERIFIED**
68. Tso, C.-H.M. et al. (2024). Ensemble Kalman inversion of induced polarization data. *GJI* 236. **PARTIAL** (co-authors and DOI not seen)
69. [Authors not seen] (2025). Regularized ensemble Kalman inversion for robust and efficient gravity data modeling to identify mineral and ore deposits. *Scientific Reports* (PMC12689625). **UNVERIFIED** (authors and DOI)
70. Mosegaard, K. (2023). Critique of the Bayesian evidence for model selection in physical inversion. arXiv:2308.05858. **PARTIAL** (exact title not seen)

### Large-scale computation and software
71. Li, Y. & Oldenburg, D.W. (1996). 3-D inversion of magnetic data. *Geophysics* 61(2), 394–408. DOI not seen. **PARTIAL**
72. Li, Y. & Oldenburg, D.W. (2003). Fast inversion of large-scale magnetic data using wavelet transforms and a logarithmic barrier method. *GJI* 152(2), 251–265. doi:10.1046/j.1365-246X.2003.01766.x. **VERIFIED**
73. Haber, E. & Heldmann, S. (2007). An octree multigrid method for quasi-static Maxwell's equations with highly discontinuous coefficients. *J. Comput. Phys.* **PARTIAL** (volume, pages and DOI not seen)
74. Davis, K. & Li, Y. (2011). Fast solution of geophysical inversion using adaptive mesh, space-filling curves and wavelet compression. *GJI* 185, 157–166. DOI not seen. **PARTIAL**
75. Cockett, R., Kang, S., Heagy, L.J., Pidlisecky, A. & Oldenburg, D.W. (2015). SimPEG: an open source framework for simulation and gradient based parameter estimation in geophysical applications. *Computers & Geosciences*. DOI not seen. **PARTIAL**
76. Heagy, L.J., Cockett, R., Kang, S., Rosenkjaer, G.K. & Oldenburg, D.W. (2017). A framework for simulation and inversion in electromagnetics. *Computers & Geosciences* 107, 1–19. doi:10.1016/j.cageo.2017.06.018; arXiv:1610.00804. **VERIFIED**
77. Rücker, C., Günther, T. & Wagner, F.M. (2017). pyGIMLi: an open-source library for modelling and inversion in geophysics. *Computers & Geosciences* 109, 106–123. doi:10.1016/j.cageo.2017.07.011. **VERIFIED**
78. Soler, S.R. & Uieda, L. (2021). Gradient-boosted equivalent sources. *GJI* 227(3), 1768–1783. doi:10.1093/gji/ggab297. **VERIFIED**
79. [Authors not seen] (2022). Joint nonlinear inversion of full tensor gravity gradiometry data and its parallel algorithm. *IEEE TGRS*. doi:10.1109/TGRS.2022.3147028. **PARTIAL**
80. Caprioti/Capriotti, Heagy & Fournier (2020). Dask-parallel SimPEG for large-scale potential-field inversion. ParCFD 2020. **UNVERIFIED** (title and author spellings)
81. Fatiando a Terra / Harmonica v0.6.0 (2023). Zenodo record 7690145. **PARTIAL** (software citation)

### Remanence
82. Lelièvre, P.G. & Oldenburg, D.W. (2009). A 3D total magnetization inversion applicable when significant, complicated remanence is present. *Geophysics* 74(3), L21–L30. doi:10.1190/1.3103249. **VERIFIED**
83. Li, Y., Shearer, S.E., Haney, M.M. & Dannemiller, N. (2010). Comprehensive approaches to 3D inversion of magnetic data affected by remanent magnetization. *Geophysics* 75(1), L1–L11. doi:10.1190/1.3294766. **VERIFIED**
84. Ellis, R.G., de Wet, B. & Macleod, I.N. (2012). Inversion of magnetic data for remanent and induced sources. *ASEG Extended Abstracts* 2012(1), 1–4. doi:10.1071/ASEG2012ab117. **VERIFIED**
85. Fournier, D., Heagy, L.J. & Oldenburg, D.W. (2020). Sparse magnetic vector inversion in spherical coordinates. *Geophysics* 85(3), J33–J49. doi:10.1190/geo2019-0244.1. **VERIFIED** (pages given inconsistently across sources)

---

### Corrections to the brief (found during verification)
- Astic, Heagy & Oldenburg's multi-physics PGI is **GJI 2020** (224(1):40–68), not 2021.
- Sun & Li's guided-FCM joint inversion is **Geophysics 2016** (81(3)); the GJI generalised-FCM paper is 2017.
- Lelièvre & Oldenburg (2009) is titled "A 3D total magnetization inversion applicable when significant, complicated remanence is present."
- Ellis et al. (2012) is an ASEG *extended abstract*, not a journal paper.
- A Zhdanov-authored "large-scale focusing Gramian" gravity–magnetic paper was not found. The verified large-scale Gramian paper is Vatankhah et al. (2022).
- No peer-reviewed gravity/magnetic SBI paper and no 2024 diffusion-prior potential-field paper were found. Treat both as research gaps, not established methods.
