# Tasks 3 and 4: Ideas from other inverse-problem fields, and how to add a third physical property

Prepared for the Alt Carbon PhD project on 3D gravity and magnetic joint inversion with probabilistic coupling.
Date: 2026-11-09.

**How citations were checked.** Every reference below appeared in a web search result during this research. Each has a DOI or arXiv ID taken from those results.
- Items tagged **UNVERIFIED** have a bibliographic detail (DOI, title or venue) that the search results did not fully confirm.
- Effort estimates assume one PhD student who already has a working SimPEG-style gravity and magnetic joint inversion.

---

## Part A: Ideas from other inverse-problem fields

### Notation used throughout
- $\mathbf m_\rho$: density contrast in each mesh cell.
- $\mathbf m_\kappa$: magnetic susceptibility in each mesh cell.
- $\mathbf d_g = \mathbf G_g\mathbf m_\rho + \mathbf e_g$: gravity data. The forward operator is linear.
- $\mathbf d_m = \mathbf G_m\mathbf m_\kappa + \mathbf e_m$: magnetic data. The forward operator is linear if remanence and self-demagnetisation are ignored.
- The current coupling is assumed to be probabilistic, for example a petrophysical Gaussian mixture or a correlation prior.

Two facts about potential fields drive most of the transfer decisions below:
1. **The forward operators are linear, dense and cheap.** Methods that need many forward solves, such as HMC, ensembles and diffusion guidance, are affordable here. Surrogates such as FNOs and PINNs bring little gain.
2. **The problem has a very large null space and no intrinsic depth resolution.** Most of the gain comes from better priors, such as structural, sparse or learned priors. More data-fit machinery helps less.

### A1. Medical imaging

#### A1.1 Compressed sensing and total variation (CT and MRI)
- **Key paper:** Lustig, Donoho and Pauly (2007), Sparse MRI, *Magn. Reson. Med.* 58(6):1182-1195. DOI 10.1002/mrm.21391.
- **What it solves:** recovering images from heavily undersampled data by promoting sparsity in a transform domain, using ℓ1, TV or wavelet penalties.
- **How to transfer:**
  - Potential-field codes already use ℓp "compact" and "blocky" norms. Heagy et al. used sparse and compact norms for carbon-mineralisation targets (arXiv 2203.13894).
  - The new elements are three things:
    1. Sparsity in a learned or multiscale dictionary, such as 3D wavelets on the octree, instead of in cell values.
    2. Isotropic 3D TV, which favours sharp-edged bodies, instead of separate anisotropic ℓp-on-gradient terms.
    3. Proper convex solvers (ADMM or primal-dual) instead of IRLS.
  - Tomofast-x v2 already uses wavelet compression (Ogarko et al. 2024, *GMD* 17:2325, DOI 10.5194/gmd-17-2325-2024).
- **Expected gain:** sharper boundaries and better volume estimates for compact bodies such as serpentinised or ultramafic lenses. There is no gain in depth resolution unless depth weighting is kept.
- **Risk:** low. The solutions are MAP estimates only, and the results depend on the regularisation weights.
- **Effort:** 5-10 days.

#### A1.2 Parallel level sets and structure-guided TV (PET-MRI)
- **Key papers:**
  - Ehrhardt et al. (2015), Joint reconstruction of PET-MRI by exploiting structural similarity, *Inverse Problems* 31:015001. DOI 10.1088/0266-5611/31/1/015001.
  - Ehrhardt et al. (2016), PET reconstruction with an anatomical MRI prior using parallel level sets, *IEEE TMI* 35(9). DOI 10.1109/TMI.2016.2549601.
  - Ehrhardt and Betcke (2016), Multicontrast MRI reconstruction with structure-guided total variation, *SIAM J. Imaging Sci.* 9(3):1084. DOI 10.1137/15M1047325.
- **What it solves:** coupling two images that share edges, but not intensities, without a petrophysical relationship.
- **How the two coupling terms work:**
  - **Parallel level sets (PLS):** penalises $\sum_i \big(\|\nabla u\|\|\nabla v\| - |\langle\nabla u,\nabla v\rangle|\big)$, or a smoothed version. Unlike the cross-gradient $\|\nabla u\times\nabla v\|^2$, it is scale-aware and does not vanish where one gradient is zero.
  - **Directional TV (dTV):** the projection $\big(I-\xi\xi^\top\big)\nabla u$ removes the gradient component along the reference edge direction $\xi=\nabla v/\|\nabla v\|_\eta$.
- **How to transfer:**
  - Drop-in replacement for the cross-gradient between $\mathbf m_\rho$ and $\mathbf m_\kappa$, and later for the third property.
  - It extends directly to 3 or more properties through joint TV (A1.2b below) or pairwise PLS.
  - Background on cross-gradient and structural joint inversion in geophysics:
    - Haber and Oldenburg (1997), *Inverse Problems* 13:63. DOI 10.1088/0266-5611/13/1/006.
    - Gallardo and Meju (2003), *GRL* 30. DOI 10.1029/2003GL017370.
    - Haber and Holtzman Gazit (2013), Model fusion and joint inversion, *Surv. Geophys.* 34:675. DOI 10.1007/s10712-013-9232-4. This paper covers joint TV as a convex structural coupling.
- **Expected gain:**
  - Joint TV is convex, so it gives better-conditioned and less local-minimum-prone coupling than the cross-gradient, which is non-convex.
  - The coupling weight is easier to tune.
  - Edges are preserved where one property has no contrast.
- **Risk:** low to medium. These terms enforce co-located edges, which is wrong where magnetite-destructive alteration decouples density and susceptibility boundaries.
- **Effort:** 4-8 days. Implement it as a SimPEG regularisation object with gradient and Hessian approximation.

#### A1.3 Learned unrolled reconstruction (learned primal-dual)
- **Key paper:** Adler and Öktem (2018), Learned primal-dual reconstruction, *IEEE TMI* 37(6):1322. DOI 10.1109/TMI.2018.2799231; arXiv 1707.06474.
- **What it solves:** fast, high-quality reconstruction by unrolling an iterative solver. CNNs replace the proximal operators, and the forward operator stays inside the network.
- **How to transfer:**
  - Unroll about 10 iterations of the primal-dual algorithm using $\mathbf G_g,\mathbf G_m$ and their adjoints.
  - Train on synthetic geology, for example the Noddyverse dataset used by arXiv 2603.06829.
  - The dense $\mathbf G$ for 1e5 cells is heavy but can be compressed or computed matrix-free.
- **Expected gain:**
  - Fast amortised inversion: seconds per survey.
  - The output is a strong initial model or proposal for a sampler.
- **Risk:** high.
  - The result is only as good as the training geology, which needs Indian targets such as Deccan basalt and ultramafics.
  - The network is mesh-specific and gives no calibrated uncertainty.
- **Effort:** 20-30 days.

#### A1.4 Electrical impedance tomography (EIT)
- **Key paper:** Hamilton and Hauptmann (2018), Deep D-bar: real-time EIT imaging with deep neural networks, *IEEE TMI* 37(10). arXiv 1711.03180. The DOI was not shown in the results.
- **What it solves:** EIT is mathematically the same as DC resistivity. The paper post-processes a fast, blurry direct reconstruction with a U-Net to sharpen it.
- **How to transfer:**
  - Mostly relevant to Task 4: DC/IP as the third property.
  - The idea of a "cheap blurry reconstruction plus learned deblurring" also applies to potential fields. Feed a smooth Tikhonov or L2 inversion into a U-Net trained on synthetic pairs.
- **Expected gain:** cheap sharpening, and a good starting model.
- **Risk:** it hallucinates structure and gives no uncertainty.
- **Effort:** 8-12 days.

### A2. Astrophysics

#### A2.1 CLEAN (radio interferometry)
- **Key paper:** Högbom (1974), *A&AS* 15:417. This is a classic paper with no DOI; it was cited in the results via Cornwell (2009), *A&A*, DOI 10.1051/0004-6361/200912148.
- **What it solves:** greedy deconvolution. The method repeatedly finds the brightest residual peak and subtracts a scaled point-spread function.
- **How to transfer:**
  - A greedy "source-body CLEAN" for potential fields.
  - Iteratively place the cell or prism with the largest normalised correlation to the residual, after depth weighting, and subtract its response. This is matching pursuit with $\mathbf G$ columns.
  - The result is a fast sparse initial model and an estimate of the number of bodies.
- **Expected gain:** a fast, interpretable compact-body starting model, useful as an initialiser for the sampler.
- **Risk:** depth ambiguity makes greedy selection biased toward shallow cells unless depth-weighted.
- **Effort:** 3-5 days.

#### A2.2 Sparsity Averaging Reweighted Analysis (SARA)
- **Key paper:** Carrillo, McEwen and Wiaux (2012), *MNRAS* 426:1223. DOI 10.1111/j.1365-2966.2012.21605.x; arXiv 1205.3123.
- **What it solves:** a reweighted ℓ1 penalty in a concatenation of several wavelet bases (average sparsity), solved by convex splitting.
- **How to transfer:**
  - Replace single-basis ℓp regularisation with reweighted analysis sparsity over a concatenation of 3D Daubechies wavelets plus a gradient (TV) basis.
  - Works on a tensor mesh, or on octree leaves via a wavelet-on-tree.
- **Expected gain:** better multi-scale structure than a pure TV or ℓ0-like penalty: broad basins plus compact bodies together.
- **Risk:** medium. There is solver complexity and the result is still a MAP point estimate.
- **Effort:** 6-10 days. Can be combined with A1.1.

#### A2.3 Event Horizon Telescope imaging and Bayesian imaging with normalising flows
- **Key papers:**
  - EHT Collaboration (2019), Paper IV, *ApJL* 875:L4. DOI 10.3847/2041-8213/ab0e85; arXiv 1906.11241.
  - Sun and Bouman, Deep Probabilistic Imaging (DPI), arXiv 2010.14462 (AAAI 2021).
  - α-DPI, arXiv 2201.08506.
- **What it solves:**
  - Paper IV calibrates imaging choices by running many independent pipelines on synthetic data and surveying the parameter space.
  - DPI trains a normalising flow to approximate the posterior by variational inference.
- **How to transfer:**
  1. **EHT-style "imaging-parameter survey".** Run the joint inversion over a grid of regularisation and coupling hyperparameters on synthetic look-alikes, and keep only settings that pass on all of them. This is a cheap and honest way to estimate uncertainty due to user choices.
  2. **DPI-style variational posterior.** Train a flow $q_\phi(\mathbf m)$ to minimise $\mathrm{KL}(q\,\|\,p(\mathbf m\mid\mathbf d))$. The geophysics analogue is Zhao, Curtis and Zhang (2022), Bayesian seismic tomography using normalizing flows, *GJI* 228:213, DOI 10.1093/gji/ggab298.
- **Expected gain:** posterior samples for a 1e4-1e5-cell model at much lower cost than MCMC.
- **Risk:**
  - The flow can miss modes; α-DPI notes trouble with disconnected modes.
  - Flows over a full 3D mesh are memory-heavy, so use a low-dimensional parameterisation.
- **Effort:** survey 3-5 days; DPI 15-25 days.

#### A2.4 Gravitational-lensing mass reconstruction
- **Key papers:**
  - Kaiser and Squires (1993), *ApJ* 404:441. DOI 10.1086/172297.
  - Remy et al., Probabilistic mass mapping with neural score estimation, arXiv 2201.05561 (A&A 2023). This paper combines a Gaussian (Wiener) analytic prior with a learned score correction, and samples with annealed HMC. Code: jax-lensing.
- **What it solves:** recovering a projected mass density from shear data. It is very close to potential-field inversion, being linear, Fourier-diagonal and noisy.
- **How to transfer:**
  - The Remy recipe transfers almost directly to 2D or 2.5D gravity.
  - The prior score is (Gaussian-process score) plus (learned residual score), sampled by annealed HMC.
  - For 3D, apply it in a reduced space: depth slabs or a latent space.
- **Expected gain:** principled posterior samples with a physically motivated Gaussian backbone. The learned part only corrects non-Gaussian features such as sharp bodies.
- **Risk:** medium. Training data are needed for the residual score.
- **Effort:** 12-20 days.

#### A2.5 Hamiltonian Monte Carlo at scale
- **Key paper:** Fichtner, Zunino and Gebraad (2019), HMC solution of tomographic inverse problems, *GJI* 216:1344. DOI 10.1093/gji/ggy496. The abstract reports thousands of unknowns on a laptop, with tuned mass matrices.
- **What it solves:** gradient-based MCMC that scales to high dimensions.
- **How to transfer:**
  - Potential fields are an ideal case. Gradients $\mathbf G^\top\mathbf C_d^{-1}(\mathbf G\mathbf m-\mathbf d)$ are a single matrix-vector product.
  - A good mass matrix is $\mathbf M\approx \mathbf G^\top\mathbf C_d^{-1}\mathbf G + \mathbf C_m^{-1}$ (Gauss-Newton Hessian), or a diagonal or low-rank version of it.
  - Use HMC for the continuous properties inside a Gibbs loop, with discrete lithology labels updated separately (see Part B (a)).
  - Use bounded or transformed variables, such as log-susceptibility.
- **Expected gain:** true posterior samples for 1e4-1e5 cells, including a non-Gaussian mixture prior. These enable volume-at-depth probability estimates of reactive rock, which matter directly for carbon-mineralisation decisions.
- **Risk:** medium.
  - Mixture priors create multimodality.
  - Mass matrix and step tuning are needed.
- **Effort:** 8-15 days with JAX or PyTorch autodiff.

### A3. Machine learning

#### A3.1 Diffusion posterior sampling (DPS) and DDRM
- **Key papers:**
  - Chung et al. (2023), Diffusion Posterior Sampling for General Noisy Inverse Problems, ICLR. arXiv 2209.14687.
  - Kawar, Elad, Ermon and Song (2022), DDRM, NeurIPS. arXiv 2201.11793.
- **What they solve:** using a pretrained diffusion prior to sample $p(\mathbf m\mid\mathbf d)$ for noisy linear or nonlinear forward models, without retraining for each problem.
  - DPS adds the likelihood gradient evaluated at the denoised estimate $\hat{\mathbf m}_0(\mathbf m_t)$.
  - DDRM works in the SVD basis of a linear operator.
- **How to transfer:**
  - The gravity and magnetic operators are linear, so both methods apply.
  - DDRM needs the SVD of $\mathbf G$, which is feasible after compression or with a randomised SVD.
  - DPS needs only $\mathbf G$ and $\mathbf G^\top$.
  - To run on a 3D mesh, train the diffusion model in a VAE latent space (latent diffusion), as done for gravity and magnetics below.
- **Expected gain:**
  - Geologically realistic samples with sharp contacts.
  - The posterior spread reflects the prior's notion of plausible geology.
- **Risk:** high.
  - DPS is known to be miscalibrated. Its likelihood approximation is crude and it under- or over-estimates the spread.
  - Results are dominated by how representative the training geology is.
  - **Mitigation:** run a few MCMC correction steps (Metropolis-adjusted) after DPS, or check calibration on synthetics.
- **Effort:** 25-40 days, including generating a training set.

#### A3.2 Existing applications to gravity and magnetic inversion (2022-2026)
1. **Joint 3D Gravity and Magnetic Inversion via Rectified Flow and Ginzburg-Landau Guidance.** arXiv 2603.06829 (March 2026).
   - Trains a 3D VAE on Noddyverse density and susceptibility volumes.
   - Runs latent rectified-flow posterior sampling, with a Ginzburg-Landau (phase-field) regulariser that pushes toward ore-like binary bodies.
   - This is the closest existing work to the Alt Carbon problem, and it is the paper to reproduce first.
2. **Three-dimensional inversion of gravity data using implicit neural representations.** Mishra et al., arXiv 2510.17876. A v2 in 2026 adds "and scientific machine learning" to the title.
   - Density is represented as a coordinate MLP with positional encoding, fitted directly through the forward operator. No mesh is predefined.
3. **Depth-Aware Implicit Neural Representation Priors for 3D Gravity Inversion.** arXiv 2608.08959 (2026).
   - Uses per-depth-slab INRs with separate Fourier-feature bandwidths, plus TV, sparsity and inter-slab consistency penalties.
4. **Regularized ensemble Kalman inversion for gravity data, mineral and ore deposits.** *Sci. Rep.* (2025). DOI 10.1038/s41598-025-30141-y.
5. **Diffusion posterior sampling for multivariate subsurface property generation.** arXiv 2507.15809 (2025).
   - Adapts DPS (Chung et al.) for probabilistic inversion of several properties together.
   - **UNVERIFIED:** the exact title and whether it targets potential fields; the results only described it as multivariate subsurface generation.
6. **GMNet, CNN joint gravity-magnetic inversion.** *Remote Sensing* (2024). **UNVERIFIED** (no DOI shown).
7. **Enhanced 3D Gravity Inversion Using ResU-Net with Density Logging Constraints.** arXiv 2601.02890.
8. **Diffusion-based posterior sampling in full waveform inversion.** arXiv 2512.12797. This is seismic, not potential-field, but it is relevant to a seismic third property.

#### A3.3 Neural fields and implicit neural representations (INRs)
- **Key paper:** arXiv 2510.17876; extension arXiv 2608.08959. See A3.2.
- **What it solves:** a mesh-free, continuous model parameterisation with implicit smoothness controlled by Fourier-feature bandwidth.
- **How to transfer:**
  - Use one MLP with several outputs: $f_\theta(x,y,z)\to(\rho,\kappa,\eta)$.
  - The shared hidden layers give implicit structural coupling.
  - Each physics queries the field at its own mesh's cell centres or quadrature points. **This removes the octree-versus-PDE-mesh mismatch** (see Part B).
  - For uncertainty, use deep ensembles, or Bayesian last-layer or SWAG.
- **Expected gain:**
  - Natural multi-physics coupling.
  - Resolution independence.
  - Compact parameters, about 1e5 weights, which are independent of the size of the domain.
- **Risk:**
  - Spectral bias toward smooth fields, and the results are sensitive to the bandwidth hyperparameter.
  - Uncertainty from ensembles is not a true posterior.
  - Optimisation is non-convex.
- **Effort:** 8-12 days for gravity and magnetics; plus 5 days for a third property.

#### A3.4 Physics-informed neural networks (PINNs)
- **Key paper:** Raissi, Perdikaris and Karniadakis (2019), *J. Comput. Phys.* 378:686. DOI 10.1016/j.jcp.2018.10.045. A gravity PINN that embeds the Poisson equation also exists in the CEUR-WS Vol-4224 short paper (Nurtas and Kenzhebayeva); it has no DOI and is **UNVERIFIED** as to venue details.
- **Transfer:** low value for gravity and magnetics. The integral forward model is exact and cheap, so enforcing $\nabla^2\phi=4\pi G\rho$ softly is worse.
- **Where it helps:** only for a PDE-based third physics (DC/EM) where meshing is painful, or for continuation of data to irregular surfaces.
- **Risk:** high, because PINN inverse problems are hard to train.
- **Effort:** 15+ days. **Not recommended.**

#### A3.5 Fourier neural operators (FNOs) as surrogates
- **Key paper:** Li et al. (2021), Fourier Neural Operator for Parametric PDEs, ICLR. arXiv 2010.08895.
- **Geophysics examples:**
  - EFKAN, MT forward modelling, arXiv 2502.02195.
  - Extended FNO for MT, *IEEE TGRS* 2022. **UNVERIFIED** DOI.
- **Transfer:**
  - No need for gravity and magnetics, whose forward models are already cheap.
  - **Very useful for the third property.** A DC/IP, EM or MT forward solve on a 3D mesh costs seconds to minutes, and an FNO surrogate makes HMC, EKI or DPS loops affordable.
  - Use the surrogate only inside the sampler, with periodic true-solver corrections. Delayed-acceptance MCMC uses the surrogate for screening and the true solver for acceptance.
- **Risk:** surrogate error biases the posterior.
- **Effort:** 15-25 days, including training data.

#### A3.6 Simulation-based inference (SBI)
- **Key paper:** Cranmer, Brehmer and Louppe (2020), *PNAS* 117(48):30055. DOI 10.1073/pnas.1912789117; arXiv 1911.01429.
- **What it solves:** amortised posteriors via neural posterior or likelihood estimation when the likelihood is intractable but simulation is easy.
- **Transfer:** best for **low-dimensional parametric** questions, not voxel models. Examples:
  - Depth to top, thickness, dip and volume of an ultramafic body.
  - Fraction of reactive rock above 300 m.
  - Simulate from a geological prior and train a neural posterior estimator on (summary statistics of gravity and magnetic data) → (quantities of interest).
- **Expected gain:**
  - Direct, calibrated (checkable by simulation-based calibration) posteriors on the decision variables that Alt Carbon cares about.
  - Instant inference for new surveys.
- **Risk:** prior misspecification, and summary-statistic design.
- **Effort:** 10-15 days.

### A4. Data assimilation: ensemble Kalman methods
- **Key papers:**
  - Evensen (1994), *JGR* 99(C5):10143. DOI 10.1029/94JC00572. **UNVERIFIED**: the DOI came from EarthRef only.
  - Iglesias, Law and Stuart (2013), Ensemble Kalman methods for inverse problems, *Inverse Problems* 29:045001. DOI 10.1088/0266-5611/29/4/045001; arXiv 1209.2736.
  - Gravity application: *Sci. Rep.* 2025. DOI 10.1038/s41598-025-30141-y.
- **What it solves:** derivative-free ensemble inversion with approximate uncertainty, using only black-box forward runs.
- **How to transfer:**
  - The ensemble is drawn from the current probabilistic prior, for example GMM samples conditioned on geology.
  - Iterate EKI with localisation and inflation.
  - **The key advantage is for Task 4.** A DC/IP or MT forward code can be used without adjoints, and all three physics are updated together through the ensemble cross-covariance. That covariance automatically carries the coupling present in the prior ensemble.
- **Expected gain:** fast multi-physics inversion with approximate uncertainty, and very easy parallelisation.
- **Risk:**
  - The results are Gaussian-ish, and the ensemble collapses.
  - Uncertainty is underestimated with small ensembles; localisation is needed with fewer than about 200 members.
  - Non-Gaussian lithology mixtures are handled poorly. **Mitigation:** apply EKI in a latent space, or use a Gaussian-anamorphosis transform.
- **Effort:** 6-10 days.

### A5. Ranking: the top 5 for Alt Carbon

The scores below weigh expected gain, risk, effort and fit to the existing probabilistic gravity-magnetic code.

| Rank | Idea | Why | Gain | Risk | Effort (days) |
|---|---|---|---|---|---|
| 1 | **HMC/Gibbs sampling of the existing probabilistic posterior** (A2.5), with a Gauss-Newton mass matrix | Linear operators make it cheap. It turns the current MAP or approximate coupling into true posterior samples and volume-at-depth probabilities. It is also the engine needed for Task 4 (a). | High | Medium | 8-15 |
| 2 | **Joint TV, directional TV or parallel-level-set structural coupling** (A1.2) | Convex, and a well-tested upgrade to the cross-gradient. It scales to 3 or more properties. It is the backbone of formulation (b). | Medium-High | Low | 4-8 |
| 3 | **Latent generative prior with posterior sampling** (A3.1/A3.2; reproduce arXiv 2603.06829) | The only route to geologically realistic multimodal posteriors. A gravity-magnetic precedent already exists. Extends to a third property (formulation (c)). | Very high (if the prior matches) | High | 25-40 |
| 4 | **Ensemble Kalman inversion** (A4) | No adjoints needed, so it is the fastest path to adding a PDE-based third physics with uncertainty. It is also a good benchmark against HMC. | Medium | Medium | 6-10 |
| 5 | **Multi-output neural field** (A3.3) | Removes the mesh mismatch, and gives implicit cross-property coupling. 2025-2026 gravity papers show feasibility. | Medium | Medium | 8-12 (+5) |

**Honourable mentions:**
- SBI for decision quantities (A3.6): high value for business questions.
- EHT-style imaging-parameter survey (A2.3): cheap and honest.
- SARA or TV sparsity (A1.1/A2.2).
- FNO surrogate (A3.5): only once the third physics is PDE-based.

**Not recommended now:** PINNs, and learned primal-dual. Learned primal-dual needs large training sets and is mesh-specific.

---

## Part B: Adding a third physical property

### B1. Literature on joint inversion with more than two physics

| Work | Physics | Coupling | Notes / ID |
|---|---|---|---|
| Astic, Heagy and Oldenburg (2021), *GJI* 224(1):40-68 | Gravity + magnetics (framework is general to N physics) | Dynamic Gaussian mixture model (GMM) petrophysical and geological prior (PGI) | DOI 10.1093/gji/ggaa378; arXiv 2002.09515. Implemented in SimPEG (joint PGI tutorial on octree). This is the closest deterministic analogue of formulation (a). |
| Heagy, Astic, Capriotti, Weis and Oldenburg, carbon-mineralisation inversion | Gravity + magnetics for CO2-reactive ultramafics | Sparse and compact norms, plus volume-with-depth estimates | arXiv 2203.13894. **Directly relevant to Alt Carbon's use case.** |
| Giraud et al. (2021), Tomofast-x v1.0, *GMD* 14:6681 | Gravity + magnetics | Cross-gradient, petrophysical clustering, geological uncertainty | DOI 10.5194/gmd-14-6681-2021 |
| Ogarko et al. (2024), Tomofast-x 2.0, *GMD* 17:2325 | Gravity + magnetics (MVI) | Wavelet compression, non-uniform grids | DOI 10.5194/gmd-17-2325-2024 |
| Giraud et al. (2019), *Solid Earth* 10:193 | Gravity | Geological-uncertainty-weighted local gradient regularisation | DOI 10.5194/se-10-193-2019 |
| Giraud et al. (2019), *GJI* 218(1):666 | Gravity + magnetics | Sensitivity to petrophysical and geological input uncertainty | DOI not shown. **UNVERIFIED** |
| Moorkamp, Heincke, Jegen, Roberts and Hobbs (2011), jif3D, *GJI* 184:477 | MT + gravity + seismic refraction | Direct parameter coupling vs cross-gradient | DOI 10.1111/j.1365-246x.2010.04856.x. Finds cross-gradient preferable where petrophysics is unclear. |
| Heincke, Jegen, Moorkamp, Hobbs and Chen (2017), *J. Appl. Geophys.* 136:279 | Seismic + gravity + MT | Adaptive petrophysical coupling | DOI 10.1016/j.jappgeo.2016.10.028 |
| Sun and Li (2016), *Geophysics* 81(3):ID37 | Seismic traveltime + gravity (general to N) | Guided fuzzy c-means clustering | DOI 10.1190/geo2015-0457.1 |
| Lelièvre, Farquharson and Hurich (2012), *Geophysics* 77(1):K1 | Seismic traveltime + gravity, unstructured tetrahedral meshes | Several coupling measures | DOI 10.1190/1.3513182. DOI from a secondary index; check before citing. |
| Gallardo and Meju (2003), *GRL* 30 | DC resistivity + seismic | Cross-gradient | DOI 10.1029/2003GL017370 |
| Haber and Oldenburg (1997), *Inverse Problems* 13:63 | Generic | Structural (curvature) coupling | DOI 10.1088/0266-5611/13/1/006 |
| Haber and Holtzman Gazit (2013), *Surv. Geophys.* 34:675 | Generic | Joint TV (convex) | DOI 10.1007/s10712-013-9232-4 |
| Bosch (1999), lithologic tomography, *JGR* | Seismic + gravity | Lithology-class petrophysics, MCMC | DOI 10.1029/1998JB900014 (from EarthRef). Bosch and McGaughey (2001), *TLE*, covers gravity + magnetics with lithologic constraints; no DOI shown, **UNVERIFIED**. |
| Lithological tomography with the correlated pseudo-marginal method | Lithology-based Bayesian inversion | Pseudo-marginal MCMC over latent petrophysics | arXiv 2110.05210 |
| Ghalenoei, Dettmer, Ali and Kim (2022), *GJI* 230(1):363 | Gravity + magnetics | Trans-dimensional Voronoi, hierarchical noise | DOI 10.1093/gji/ggac083. Companion paper: *Inverse Problems* 38:075011 (alpha shapes, autoregressive noise). |
| Gunning and Sams (2018), *Geophys. Prosp.* | Seismic | MRF lithotype prior with facies-conditional Gaussians | DOI 10.1111/1365-2478.12625 (from a Zendy record). Shows the analytic posterior structure for joint categorical and continuous inference. |
| Gravity + DC resistivity on a deformed hexahedral mesh, *Minerals* 14(10):1012 (2024/25) | Gravity + DC | Structural | **UNVERIFIED** (no DOI shown) |
| IP + self-potential cross-gradient joint inversion, *GJI* 243(3) | IP + SP | Cross-gradient | **UNVERIFIED** (no DOI shown) |
| SimPEG: Cockett et al. (2015), *Comput. Geosci.*; Heagy et al. (2017) EM framework | Gravity, magnetics, DC/IP, EM, MT on tensor and octree meshes | Software | DOI 10.1016/j.cageo.2015.09.015; DOI 10.1016/j.cageo.2017.06.018 |

**Gap:** no paper found does fully Bayesian, sampled joint inversion of gravity + magnetics + (DC/IP, EM or seismic) with a latent lithology field in 3D.
- Astic et al. give the MAP version.
- Bosch and Ghalenoei give sampled versions with two physics.
- This gap is a publishable contribution for the PhD.

**Choosing the third property.** This choice is for Alt Carbon's targets: reactive basalts and ultramafics, and serpentinisation or carbonation fronts.
- **DC resistivity / IP chargeability is the most natural choice.**
  - Serpentinisation produces magnetite, which raises susceptibility and chargeability and lowers density. Carbonation destroys magnetite.
  - Clay or weathering and pore fluids control resistivity.
  - DC/IP is cheap to acquire on the ground and is supported by SimPEG on octree meshes.
- **EM (TEM or airborne) or MT** is better for depth to conductive basement.
- **Seismic** works for velocity-density petrophysics in sedimentary cover, but is expensive in India for small sites.

### B2. Common setup and symbols

| Symbol | Meaning |
|---|---|
| $i=1,\dots,M$ | Cells of the shared **model mesh** $\mathcal M$ (an octree) |
| $\mathbf m=(\mathbf m^{(1)},\mathbf m^{(2)},\mathbf m^{(3)})$ | Properties on $\mathcal M$. $\mathbf m^{(1)}=\Delta\rho$ (g/cc), $\mathbf m^{(2)}=\kappa$ (SI) or $\log\kappa$, $\mathbf m^{(3)}=\log\sigma$ (S/m) and/or $\operatorname{logit}\eta$ (chargeability). $\mathbf m_i\in\mathbb R^{P}$ is the property vector in cell $i$, with $P=3$ or $4$. |
| $\mathbf d_j,\ j\in\{g,m,e\}$ | Gravity, magnetic and third (electrical/EM/seismic) data vectors, with $N_j$ data each |
| $\mathcal F_j$ | Forward operator: $\mathcal F_g(\mathbf m)=\mathbf G_g\mathbf m^{(1)}$, $\mathcal F_m(\mathbf m)=\mathbf G_m\mathbf m^{(2)}$, and $\mathcal F_e$ is nonlinear (PDE). For IP, $\mathcal F_e$ is often linearised: $\mathbf d_{IP}\approx \mathbf J_\sigma \boldsymbol\eta$. |
| $\mathbf P_j$ | Projection or interpolation from model mesh $\mathcal M$ to the forward mesh of physics $j$ (see B4) |
| $\mathbf C_j(\boldsymbol\tau_j)$ | Data noise covariance with hyperparameters $\boldsymbol\tau_j$, for example a scale $s_j$ or AR parameters |
| $z_i\in\{1,\dots,K\}$ | Latent lithology or rock-unit label in cell $i$ |
| $\boldsymbol\mu_k,\boldsymbol\Sigma_k,\pi_k$ | Petrophysical mean vector, covariance and proportion of unit $k$ |
| $\beta$ | Potts interaction strength (spatial continuity of lithology) |
| $\mathcal N(i)$ | Neighbours of cell $i$ on the octree, with face-adjacency weights $w_{ik}$ |
| $\mathbf W_j$ | Depth or sensitivity weighting for potential fields |

Likelihood shared by all three formulations:
$$
p(\mathbf d\mid\mathbf m,\boldsymbol\tau)=\prod_{j\in\{g,m,e\}}\mathcal N\!\big(\mathbf d_j\,\big|\,\mathcal F_j(\mathbf P_j\mathbf m),\ \mathbf C_j(\boldsymbol\tau_j)\big)^{\gamma_j},
$$
- $\gamma_j\in(0,1]$ is an optional likelihood-tempering weight (see B5). Set $\gamma_j=1$ for a strictly Bayesian treatment.

### B3. Three candidate formulations

#### (a) Latent lithology field with a petrophysical Gaussian mixture (hierarchical Bayesian, Potts/MRF)

$$
\boxed{
p(\mathbf m,\mathbf z,\boldsymbol\Theta,\boldsymbol\tau\mid\mathbf d)\ \propto\
\underbrace{\prod_{j}\mathcal N\!\big(\mathbf d_j\mid\mathcal F_j(\mathbf P_j\mathbf m),\mathbf C_j(\boldsymbol\tau_j)\big)}_{\text{3 likelihoods}}
\ \underbrace{\prod_{i=1}^{M}\mathcal N\!\big(\mathbf m_i\mid\boldsymbol\mu_{z_i},\boldsymbol\Sigma_{z_i}\big)}_{\text{petrophysics}}
\ \underbrace{\frac{1}{Z(\beta)}\exp\!\Big(\beta\sum_{i}\sum_{k\in\mathcal N(i)}w_{ik}\,\delta_{z_i z_k}+\sum_i\log\pi_{z_i}(\mathbf x_i)\Big)}_{\text{Potts geology prior}}
\ p(\boldsymbol\Theta)\,p(\boldsymbol\tau)
}
$$

**What each part means:**
- $\boldsymbol\Theta=\{\boldsymbol\mu_k,\boldsymbol\Sigma_k\}_{k=1}^K$.
- $p(\boldsymbol\Theta)=\prod_k \mathrm{NIW}(\boldsymbol\mu_k,\boldsymbol\Sigma_k\mid \boldsymbol\mu_{0k},\lambda_{0k},\boldsymbol\Psi_k,\nu_k)$. The hyperparameters come from core and hand-sample petrophysics, as in Astic's "dynamic" GMM.
- $\pi_{z}(\mathbf x_i)$ is an optional spatially varying geological-map prior, such as a mapped basalt unit, or probabilities from a geological model as in Giraud.
- $p(\boldsymbol\tau)$ is an inverse-gamma prior on each dataset's noise scale $s_j^2$, where $\mathbf C_j=s_j^2\mathbf C_{j,0}$. This gives hierarchical data weighting.
- The full covariance $\boldsymbol\Sigma_k$ (P×P) encodes **within-unit cross-property correlation**. An example is the density-susceptibility trend from the degree of serpentinisation, $\Delta\rho\downarrow$ and $\kappa\uparrow$, within a "serpentinite" unit.
  - This captures what direct parameter coupling captures, but per lithology.
  - Linear trends can be extended to curved ones with "rotated/nonlinear" GMMs, using one component per segment.
- If chargeability or conductivity is unknown in some units, use broad marginals for that unit: a large $\Sigma_k^{(3,3)}$.

**Inference: blocked Gibbs.**
1. $\mathbf z\mid\mathbf m,\boldsymbol\Theta$: each cell's conditional is $p(z_i=k\mid\cdot)\propto \pi_k\,\mathcal N(\mathbf m_i\mid\boldsymbol\mu_k,\boldsymbol\Sigma_k)\exp(\beta\sum_{k'\in\mathcal N(i)}w_{ik'}\delta_{k z_{k'}})$.
   - Use chequerboard or graph-colouring parallel updates, or Swendsen-Wang for large $\beta$.
2. $\boldsymbol\Theta\mid\mathbf m,\mathbf z$: conjugate NIW update.
3. $\boldsymbol\tau\mid\mathbf m,\mathbf d$: conjugate inverse-gamma update, $s_j^2\mid\cdot\sim\mathrm{IG}\big(a_j+N_j/2,\ b_j+\tfrac12\|\mathbf d_j-\mathcal F_j\|^2_{\mathbf C_{j,0}^{-1}}\big)$.
4. $\mathbf m\mid\mathbf z,\boldsymbol\Theta,\boldsymbol\tau,\mathbf d$:
   - For gravity and magnetics alone this is **exactly Gaussian**. The update is $\mathbf m\mid\cdot\sim\mathcal N(\hat{\mathbf m},\mathbf H^{-1})$ with $\mathbf H=\sum_j\mathbf P_j^\top\mathbf G_j^\top\mathbf C_j^{-1}\mathbf G_j\mathbf P_j+\mathrm{blkdiag}(\boldsymbol\Sigma_{z_i}^{-1})$. Sample with perturbation-optimisation: solve with CG on a randomly perturbed right-hand side.
   - With $\mathcal F_e$ nonlinear, use HMC or MALA (A2.5) with a Gauss-Newton mass matrix.
   - Alternatively, use a Laplace / randomise-then-optimise proposal inside Metropolis-Hastings.
   - A **cheap first step:** linearised IP ($\mathbf d_{IP}=\mathbf J\boldsymbol\eta$, $\mathbf J$ from a fixed DC conductivity model). This keeps everything Gaussian-conditional.

**Fixing $\beta$:**
- Fix it from geological correlation lengths. $Z(\beta)$ is intractable.
- Or estimate it with the exchange algorithm or a pseudo-likelihood.

**Link to PGI:** if $\beta=0$ and $\mathbf z$ is marginalised, the prior reduces to the GMM prior of Astic et al. (2021). The MAP of this formulation is PGI with a spatial lithology term, so the existing SimPEG PGI code is a validated MAP baseline.

**Pros:**
- Interpretable; it outputs $P(z_i=\text{reactive rock}\mid\mathbf d)$ directly.
- Handles partially decoupled properties.
- Hierarchical noise means no hand-tuned data weights.

**Cons:**
- Multimodality: label switching, and $\mathbf z$ mixing at large $\beta$.
- Needs petrophysical data for $K$ units.

#### (b) Structural Gibbs prior: cross-gradient or joint total variation

$$
\boxed{
p(\mathbf m,\boldsymbol\tau\mid\mathbf d,\boldsymbol\alpha,\lambda)\ \propto\
\prod_{j}\mathcal N\!\big(\mathbf d_j\mid\mathcal F_j(\mathbf P_j\mathbf m),\mathbf C_j(\boldsymbol\tau_j)\big)
\ \exp\!\Big(-\sum_{p=1}^{P}\alpha_p\,\big\|\mathbf W_p(\mathbf m^{(p)}-\mathbf m^{(p)}_{\mathrm{ref}})\big\|_2^2\ -\ \lambda\,\Phi_{\mathrm{s}}(\mathbf m)\Big)\ p(\boldsymbol\tau)
}
$$

The structural term $\Phi_{\mathrm{s}}$ can take one of three forms:

- **Cross-gradient, summed over pairs:**
$$
\Phi_{\mathrm{xg}}(\mathbf m)=\sum_{p<q}\sum_{i}V_i\,\big\|\nabla_h\tilde{\mathbf m}^{(p)}_i\times\nabla_h\tilde{\mathbf m}^{(q)}_i\big\|_2^2 ,\qquad \tilde{\mathbf m}^{(p)}=\mathbf m^{(p)}/s_p
$$
- **Joint total variation (convex):**
$$
\Phi_{\mathrm{JTV}}(\mathbf m)=\sum_{i}V_i\sqrt{\textstyle\sum_{p=1}^{P}\big\|\nabla_h\tilde{\mathbf m}^{(p)}_i\big\|_2^2+\varepsilon^2}
$$
- **Parallel level sets, per pair:**
$$
\Phi_{\mathrm{PLS}}(\mathbf m)=\sum_{p<q}\sum_i V_i\Big(\|\nabla_h\tilde{\mathbf m}^{(p)}_i\|_\varepsilon\|\nabla_h\tilde{\mathbf m}^{(q)}_i\|_\varepsilon-\big|\langle\nabla_h\tilde{\mathbf m}^{(p)}_i,\nabla_h\tilde{\mathbf m}^{(q)}_i\rangle\big|_\varepsilon\Big)
$$

**Symbols:**
- $\nabla_h$ is the octree cell-gradient operator (face-to-cell averaged).
- $V_i$ is the cell volume.
- $s_p$ is a property scale that makes properties comparable, for example a robust standard deviation.
- $\alpha_p$ are smallness and smoothness weights.
- $\lambda$ is the coupling strength.

**The posterior is a Gibbs or MRF density.** Its normaliser $Z(\boldsymbol\alpha,\lambda)$ is intractable. Two consequences follow:
1. Treat $(\boldsymbol\alpha,\lambda)$ as fixed. Choose them with the discrepancy principle (each $\chi^2_j\approx N_j$), cross-validation, or an EHT-style survey (A2.3).
2. To sample, use MALA or HMC on the smoothed ($\varepsilon>0$) potential. JTV is log-concave when the forward model is linear, so samplers mix well for gravity + magnetics. The cross-gradient term is non-convex, so expect poor mixing there.

**Pros:**
- No petrophysical relationships needed. This is right when chargeability-density links are unknown (Moorkamp et al. 2011 conclusion).
- Easy to add to existing code.

**Cons:**
- Assumes co-located boundaries.
- Gives no lithology probabilities directly, so it needs post-hoc clustering.
- Uncertainty depends on $\lambda$, which is not inferred.

#### (c) Learned joint prior: normalising flow or diffusion

Let $\mathcal G_\phi$ be a generative model trained on multi-property 3D realisations. To build the training set:
- Sample structural geology from a geological simulator, such as Noddy/Noddyverse (as in arXiv 2603.06829) or GemPy with Indian-analogue stratigraphy.
- Sample $(\Delta\rho,\kappa,\log\sigma,\eta)$ per unit from the petrophysical GMM of (a).
- Add alteration fields, for example serpentinisation degree.

**(c1) Flow or VAE latent prior.** Let $\mathbf m=\mathcal G_\phi(\mathbf w)$ with $\mathbf w\in\mathbb R^{L}$, $L\ll PM$:
$$
\boxed{
p(\mathbf w,\boldsymbol\tau\mid\mathbf d)\ \propto\ \prod_j\mathcal N\!\big(\mathbf d_j\mid\mathcal F_j(\mathbf P_j\mathcal G_\phi(\mathbf w)),\mathbf C_j(\boldsymbol\tau_j)\big)\ \mathcal N(\mathbf w\mid\mathbf 0,\mathbf I_L)\ p(\boldsymbol\tau)
}
$$
- Sample with HMC in $\mathbf w$-space. Gradients come by autodiff through $\mathcal G_\phi$ and $\mathbf G_j^\top$, and by adjoint for $\mathcal F_e$.
- Add an optional slack term $\mathbf m=\mathcal G_\phi(\mathbf w)+\boldsymbol\delta$, with $\boldsymbol\delta\sim\mathcal N(\mathbf 0,\sigma_\delta^2\mathbf I)$, to absorb prior misspecification.

**(c2) Diffusion or score prior with DPS guidance.** A score network $\mathbf s_\phi(\mathbf m_t,t)\approx\nabla\log p_t(\mathbf m_t)$ is trained on the same data, in a VAE latent space for 3D. The reverse SDE uses the approximate posterior score:
$$
\nabla_{\mathbf m_t}\log p_t(\mathbf m_t\mid\mathbf d)\ \approx\ \mathbf s_\phi(\mathbf m_t,t)\ -\ \sum_j \frac{\zeta_j}{2}\nabla_{\mathbf m_t}\Big\|\mathbf d_j-\mathcal F_j\big(\mathbf P_j\hat{\mathbf m}_0(\mathbf m_t)\big)\Big\|^2_{\mathbf C_j^{-1}},
\qquad
\hat{\mathbf m}_0(\mathbf m_t)=\frac{\mathbf m_t+\sigma_t^2\,\mathbf s_\phi(\mathbf m_t,t)}{\alpha_t}
$$
- This is Chung et al. (arXiv 2209.14687) with the Tweedie estimate.
- $\zeta_j$ are guidance step sizes. They act as effective data weights and must be calibrated.

**Pros:**
- Realistic, multimodal, non-Gaussian joint priors. The cross-property coupling is learned, including nonlinear and alteration-driven relationships.
- Gravity-magnetic precedent exists (arXiv 2603.06829).

**Cons:**
- Expensive to train.
- DPS posteriors are not exactly calibrated.
- Results fail silently if the field geology is out of distribution.
- **Must** be validated with posterior predictive checks and held-out synthetic models.

### B4. Mesh mismatch: octree potential-field mesh vs PDE meshes

The problem:
- Gravity and magnetics are computed by integral equations on any cell set. A **topography-conforming octree with no padding** is ideal.
- DC/IP, EM and MT need a **PDE mesh** with large padding (5-10× survey size), boundary conditions, fine cells near electrodes, and often different refinement.
- Seismic traveltimes need a regular or tetrahedral eikonal grid.

**Recommended design:**
1. **One model mesh $\mathcal M$.** Use a shared octree refined under all surveys, at the finest resolution any physics can resolve. All unknowns and all priors (GMM, Potts, JTV) live on $\mathcal M$.
2. **Per-physics forward meshes $\mathcal M_j$, with linear maps $\mathbf P_j:\mathcal M\to\mathcal M_j$.**
   - Use conservative volume-weighted averaging for a fine-to-coarse map; for log-conductivity, do this in log space.
   - Use injection for the coarse-to-fine map.
   - Padding cells in $\mathcal M_j$ are filled from a fixed background value or a single extra "halfspace" parameter.
   - Gradients pass through $\mathbf P_j^\top$.
   - In SimPEG this is a chain of `maps`: `InjectActiveCells` ∘ `Mesh2Mesh` ∘ `ExpMap` ∘ `Projection`, with one wire per property.
3. **Simplest route for DC/IP:** run DC/IP forward **on the same octree**, extended with padding cells. SimPEG solves DC/IP with finite volumes on a `TreeMesh`. Then $\mathbf P_e$ is just active-cell injection, and the mismatch largely disappears.
4. **Mesh-free alternative:** a neural field $f_\theta(\mathbf x)$ (A3.3) is queried at each $\mathcal M_j$'s cell centres. Use quadrature, averaging several points per coarse cell, to avoid aliasing.
5. **Pitfalls:**
   - Resolution mismatch biases structural coupling. Gradients of a property defined on a coarse forward mesh are piecewise-constant on fine model cells, so compute $\nabla_h$ on $\mathcal M$ only.
   - Topography must be identical across meshes.
   - Depth weighting $\mathbf W$ applies to potential-field properties only.

### B5. Data weighting between datasets

Datasets differ in N (10³-10⁵ stations for airborne magnetics vs about 10³ for DC/IP), noise characterisation, and correlated errors. If nothing is done, the largest or least-noisy dataset dominates.

**Options, from most principled to least:**
1. **Hierarchical noise (default for (a) and (c)).** $\mathbf C_j=s_j^2\mathbf C_{j,0}$ with $s_j^2\sim\mathrm{IG}(a_j,b_j)$, inferred jointly as in Ghalenoei et al. (2022), which also infers autoregressive noise.
   - Marginalising $s_j$ gives a Student-t likelihood, which is robust to outliers.
   - Use AR(1) or Matérn along-line correlation for airborne magnetics. Otherwise $N_j$ is effectively overstated.
2. **Per-dataset χ² targets.** Require $\phi_j=\|\mathbf d_j-\mathcal F_j\|^2_{\mathbf C_j^{-1}}\approx N_j$ for every $j$, not just in sum. This is what Astic et al.'s multi-target misfit does for PGI; use it for MAP and (b).
3. **Effective-sample-size tempering.** Use $\gamma_j=N_{\mathrm{eff,j}}/N_j$ to correct for correlated noise or model error that has not been modelled. This makes the likelihoods "power posteriors". Report sensitivity to $\gamma$.
4. **Thinning or decimation** of dense magnetic data, for example adaptive quadtree, before inversion. This is cheaper and reduces correlation.

**Always do the following:**
- Report the per-dataset normalised misfit $\phi_j/N_j$.
- Run posterior predictive checks per dataset.
- Run a single-physics vs joint comparison to confirm that the third dataset changes the posterior where its sensitivity is high, near-surface for DC/IP.

### B6. Recommended first prototype

**Formulation (a), with DC resistivity + IP chargeability as the third property, built in SimPEG on one octree.**

1. **Week 1-2: synthetic benchmark.**
   - Build a model of a serpentinised ultramafic or basalt body with a carbonated cap.
   - Use 4 units: host, fresh ultramafic, serpentinite and carbonated zone.
   - Properties: $(\Delta\rho,\log\kappa,\log\sigma,\eta)$.
   - Simulate gravity, magnetics, and pole-dipole DC/IP lines.
2. **Week 2-3: MAP baseline.**
   - Joint PGI with 3 physics, following Astic et al. (2021).
   - SimPEG's PGI handles multiple physics through combined `DataMisfit` and `PGI` regularisation.
   - Compare against (b) with JTV or cross-gradient as the structural baseline.
3. **Week 3-5: sampled posterior.**
   - Blocked Gibbs for (a): Potts $\mathbf z$ updates; Gaussian $\mathbf m$ update using **linearised IP** and DC fixed at the MAP conductivity. This keeps the conditional Gaussian.
   - Then switch to HMC on $\log\sigma$ with adjoint gradients, or EKI (A4) if adjoints are a bottleneck.
   - Hierarchical noise scales for the data weighting.
4. **Outputs:**
   - $P(z_i=\text{serpentinite}\mid\mathbf d)$.
   - Posterior volume of reactive rock above depth $z^*$.
   - The improvement over gravity + magnetics alone, as reduced posterior variance near surface.
5. **Later:** replace the Potts/GMM prior with a learned latent prior (c1) trained on the same synthetic family, and compare calibration.

**Why (a) first:**
- It extends the current probabilistic coupling naturally.
- Its MAP has an existing validated implementation (PGI).
- It outputs lithology probabilities directly, which are the quantity Alt Carbon needs.
- Each Gibbs block is simple.
- (b) serves as a cheap baseline and (c) as the research extension.

---

## References with IDs found in this search

- Adler and Öktem 2018. DOI 10.1109/TMI.2018.2799231; arXiv 1707.06474.
- Astic, Heagy and Oldenburg 2021. DOI 10.1093/gji/ggaa378; arXiv 2002.09515.
- Bosch 1999. DOI 10.1029/1998JB900014 (from EarthRef).
- Carrillo, McEwen and Wiaux 2012. DOI 10.1111/j.1365-2966.2012.21605.x; arXiv 1205.3123.
- Chung et al. (DPS). arXiv 2209.14687.
- Cockett et al. 2015 (SimPEG). DOI 10.1016/j.cageo.2015.09.015.
- Heagy et al. 2017 (SimPEG EM). DOI 10.1016/j.cageo.2017.06.018.
- Cranmer, Brehmer and Louppe 2020. DOI 10.1073/pnas.1912789117; arXiv 1911.01429.
- EHT Paper IV. DOI 10.3847/2041-8213/ab0e85; arXiv 1906.11241.
- Ehrhardt et al. 2015. DOI 10.1088/0266-5611/31/1/015001.
- Ehrhardt et al. 2016, PLS in TMI. DOI 10.1109/TMI.2016.2549601.
- Ehrhardt and Betcke 2016. DOI 10.1137/15M1047325; arXiv 1511.06631.
- Evensen 1994. DOI 10.1029/94JC00572 (UNVERIFIED, from EarthRef only).
- Fichtner, Zunino and Gebraad 2019. DOI 10.1093/gji/ggy496.
- Gallardo and Meju 2003. DOI 10.1029/2003GL017370.
- Ghalenoei et al. 2022. DOI 10.1093/gji/ggac083.
- Giraud et al. 2021. DOI 10.5194/gmd-14-6681-2021.
- Ogarko et al. 2024. DOI 10.5194/gmd-17-2325-2024.
- Giraud et al. 2019. DOI 10.5194/se-10-193-2019.
- Haber and Oldenburg 1997. DOI 10.1088/0266-5611/13/1/006.
- Haber and Holtzman Gazit 2013. DOI 10.1007/s10712-013-9232-4.
- Hamilton and Hauptmann (Deep D-bar). arXiv 1711.03180.
- Heagy et al., carbon mineralisation. arXiv 2203.13894.
- Heincke et al. 2017. DOI 10.1016/j.jappgeo.2016.10.028.
- Iglesias, Law and Stuart 2013. DOI 10.1088/0266-5611/29/4/045001; arXiv 1209.2736.
- Kaiser and Squires 1993. DOI 10.1086/172297.
- Kawar et al. (DDRM). arXiv 2201.11793.
- Laloy et al. 2018 (SGAN prior; Water Resources Research 54:381). arXiv 1708.04975.
- Lelièvre, Farquharson and Hurich 2012. DOI 10.1190/1.3513182 (secondary index).
- Li et al. (FNO). arXiv 2010.08895.
- Lustig, Donoho and Pauly 2007. DOI 10.1002/mrm.21391.
- Moorkamp et al. 2011. DOI 10.1111/j.1365-246x.2010.04856.x.
- Raissi, Perdikaris and Karniadakis 2019. DOI 10.1016/j.jcp.2018.10.045.
- Remy et al. arXiv 2201.05561.
- Sun and Bouman (DPI). arXiv 2010.14462; α-DPI arXiv 2201.08506.
- Sun and Li 2016. DOI 10.1190/geo2015-0457.1.
- Zhao, Curtis and Zhang 2022. DOI 10.1093/gji/ggab298.
- Gravity and magnetic ML papers: arXiv 2603.06829, arXiv 2510.17876, arXiv 2608.08959, arXiv 2601.02890, arXiv 2507.15809 (UNVERIFIED title), and *Sci. Rep.* DOI 10.1038/s41598-025-30141-y.
- Lithological tomography (correlated pseudo-marginal). arXiv 2110.05210.
- Gunning and Sams 2018. DOI 10.1111/1365-2478.12625.
- EFKAN. arXiv 2502.02195.
- FWI diffusion. arXiv 2512.12797.
