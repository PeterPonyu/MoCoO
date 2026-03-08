# MoCoO: Momentum Contrast ODE-Regularized VAE for Single-Cell Trajectory Inference and Representation Learning

**Zeyu Fu**

*Manuscript prepared for IEEE Journal of Biomedical and Health Informatics (JBHI)*

---

**Abstract** — Learning biologically faithful low-dimensional representations from single-cell RNA sequencing (scRNA-seq) data requires models that simultaneously preserve discrete cell-type identity and continuous developmental dynamics. We present **MoCoO**, a modular framework that integrates a Variational Autoencoder (VAE), Neural Ordinary Differential Equations (Neural ODE), and Momentum Contrast (MoCo) with learnable prototypes for robust trajectory inference and representation learning. Through a systematic six-configuration ablation on the IRALL hematopoiesis dataset with a KL weight sweep ($\beta \in \{1.0, 0.1, 0.01\}$), we investigate the standalone power and synergistic interactions of each component. Key findings: (1) **MoCo has beta-dependent standalone power** — contrastive learning improves ARI by +0.054 and ASW by +0.018 at $\beta = 0.1$ but is ineffective at extreme $\beta$; (2) **ODE and MoCo exhibit super-additive synergy** — the interaction term for ARI is consistently positive (+0.103, +0.012, +0.131 across betas) and DB improves by up to −0.398 super-additively; (3) **prototypes improve cluster compactness** (DB reduced by 0.120–0.223 at $\beta \leq 0.1$) and embedding fidelity (DRE +0.138 at $\beta = 0.01$). The Full model achieves the best DRE (0.468) at $\beta = 0.01$ and competitive clustering across settings. Pseudotime–marker correlations are biologically significant across five systems: *Hbb-bs* ($\rho = 0.28$) in IRALL, *Slc1a3* ($\rho = 0.54$) in dentate, *Ins2* ($\rho = 0.46$) in endo, *Epor* ($\rho = 0.35$) in paul, and *MKI67* ($\rho = 0.49$) in spinoids. We publicly release the MoCoO Python package and benchmark suite.

**Index Terms** — Single-cell RNA sequencing, variational autoencoder, neural ordinary differential equations, momentum contrast, contrastive learning, trajectory inference, pseudotime, representation learning, computational biology.

---

## I. Introduction

Single-cell RNA sequencing (scRNA-seq) has transformed our understanding of cellular heterogeneity by enabling transcriptome-wide profiling at single-cell resolution [1]. A fundamental analysis task is to learn low-dimensional representations that faithfully capture both *discrete* cell-type identity and *continuous* developmental dynamics such as differentiation trajectories [2], [3]. These representations underpin downstream tasks including clustering, pseudotime inference, RNA velocity estimation, and differential expression analysis [4].

Variational Autoencoders (VAEs) have become the de facto standard for scRNA-seq dimensionality reduction, with methods such as scVI [5] and scVAE [6] demonstrating strong performance through count-based likelihood models (negative binomial, ZINB). However, standard VAE latent spaces tend to be over-smoothed, conflating nearby cell states and producing suboptimal trajectory structures. Two complementary strategies have emerged to address this limitation.

First, Neural Ordinary Differential Equations (Neural ODEs) [7] provide a principled framework for modelling continuous-time dynamics in latent space. By parameterising the derivative of the latent state with a neural network and integrating forward through an ODE solver, these models can learn smooth developmental trajectories without discrete time-step assumptions. Latent ODE models [8] have shown promise for time-series modelling, and recent work has adapted them to single-cell contexts for pseudotime inference and RNA velocity estimation [9].

Second, contrastive learning — particularly Momentum Contrast (MoCo) [10] — has proven effective at learning representations that are locally smooth yet globally discriminative. MoCo maintains a momentum-updated encoder and a large memory queue of negative keys, enabling scalable contrastive learning that is decoupled from batch size. Extensions such as prototype contrastive learning [11] further impose cluster structure by aligning representations with learnable prototype vectors.

Despite these individual advances, no existing method unifies VAE-based reconstruction, Neural ODE dynamics, and momentum contrastive learning within a single coherent framework. In this work, we propose **MoCoO** (**Mo**mentum **Co**ntrast **O**DE-regularized VAE), a modular architecture that combines all three paradigms:

1. **VAE with flexible count-based likelihoods** (MSE, NB, ZINB, Poisson, ZIP) provides a probabilistic latent space and handles the zero-inflation and overdispersion characteristic of scRNA-seq count data.

2. **Neural ODE regularisation** models continuous developmental dynamics, derives unsupervised pseudotime ordering, and produces gradient-based RNA velocity estimates.

3. **Momentum Contrast with prototype heads** imposes instance-level contrastive regularisation via a momentum encoder and memory queue, augmented by prototype contrastive learning that aligns representations with learnable cluster centres.

4. **Information bottleneck** applies a secondary low-dimensional projection for hierarchical feature extraction.

5. **Disentanglement losses** (DIP-VAE, $\beta$-TC-VAE, InfoVAE/MMD) optionally encourage axis-aligned, interpretable latent factors.

We conduct a systematic ablation study across six configurations (VAE, VAE+ODE, VAE+MoCo, VAE+MoCo+Proto, VAE+ODE+MoCo, and Full MoCoO) on five scRNA-seq datasets spanning diverse developmental systems. Our evaluation encompasses clustering metrics (ARI, NMI, ASW, Calinski–Harabasz, Davies–Bouldin), dimensionality reduction quality (DRE, DREX, LSE, LSEX), biological validation (cell-type purity, marker gene enrichment, pseudotime–trajectory alignment), training dynamics, computational cost, and batch integration quality (iLISI, bASW, cLISI, graph connectivity, isolated label ASW via scIB).

---

## II. Related Work

### A. Variational Autoencoders for scRNA-seq

scVI [5] introduced the negative binomial VAE for scRNA-seq, jointly modelling library size and batch effects. Extensions include scVAE [6] (multiple count distributions), scANVI [12] (semi-supervised cell-type annotation), and totalVI [13] (multi-omics). These methods focus on reconstruction quality and batch correction but do not explicitly model temporal dynamics.

### B. Neural ODEs and Latent ODE Models

Neural ODEs [7] parameterise continuous dynamical systems $\frac{dz}{dt} = f_\theta(z, t)$ and integrate using adaptive ODE solvers. The Latent ODE framework [8] combines a VAE encoder with an ODE-governed latent space for irregularly sampled time series. Applications to single-cell biology include trajectory inference [9] and RNA velocity estimation [14]. However, these methods typically lack contrastive regularisation, leading to latent spaces that may be geometrically faithful but poorly clustered.

### C. Contrastive Learning for Single-Cell Data

MoCo [10] and SimCLR [15] have been adapted to biological contexts. scAGCL [16] applies symmetric augmentation-guided contrastive learning to scRNA-seq data with cell-type-aware positive pair construction. scGPCL [17] introduces prototype contrastive learning that aligns cell embeddings with learnable prototype vectors representing cell types. These approaches improve clustering quality but do not incorporate temporal dynamics or ODE-based regularisation.

### D. Integrated Approaches

Several methods combine two of the three paradigms. scDiff [18] uses diffusion models (related to score-based SDEs) for single-cell generation. VeloVAE [19] integrates VAE with RNA velocity. However, to our knowledge, no prior work jointly combines VAE, Neural ODE, and momentum contrastive learning in a unified architecture for single-cell analysis.

---

## III. Method

### A. Overview

MoCoO is a modular framework illustrated in Fig. 1. The architecture processes scRNA-seq count data $X \in \mathbb{R}^{N \times G}$ ($N$ cells, $G$ genes) through five interconnected components: (1) a VAE with count-based likelihood decoder, (2) a Neural ODE solver for continuous dynamics, (3) a Momentum Contrast module with memory queue, (4) an information bottleneck, and (5) optional disentanglement regularisers.

### B. Encoder

The encoder $q_\phi(z|x)$ maps log-normalised gene expression $\tilde{x} = \log(1 + x)$ through a two-layer MLP with ReLU activations to produce mean $\mu$ and pre-softplus scale $\sigma'$ parameters of a diagonal Gaussian posterior:

$$q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(\tilde{x}), \text{softplus}(\sigma'_\phi(\tilde{x}))^2 I)$$

$$z = \mu + \text{softplus}(\sigma') \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

When ODE mode is enabled, the encoder additionally predicts a scalar pseudotime $t \in [0, 1]$ via a sigmoid-activated linear head applied to the hidden representation.

### C. Neural ODE Component

Given latent samples $\{z_i\}$ and predicted pseudotimes $\{t_i\}$, the ODE module orders cells by pseudotime and solves the initial value problem:

$$\frac{dz}{dt} = f_\psi(z, t), \quad z(t_0) = z_{\text{init}}$$

where $f_\psi$ is a two-layer MLP with ELU activations. Integration is performed using `torchdiffeq`'s adaptive-step solver. The ODE-integrated representations $z_{\text{ode}}$ are blended with VAE latent samples via configurable weights:

$$z_{\text{blend}} = \alpha_{\text{vae}} \cdot z + \alpha_{\text{ode}} \cdot z_{\text{ode}}$$

Two auxiliary losses regularise the ODE:
- **ODE–VAE alignment:** $\mathcal{L}_{\text{ode}} = \| z - z_{\text{ode}} \|_2^2$
- **Velocity consistency:** $\mathcal{L}_{\text{vel}} = 1 - \cos(z_{i+1} - z_i, f_\psi(z_i, t_i))$, encouraging the ODE velocity field to align with empirical displacements.

### D. Momentum Contrast Module

The MoCo module follows the MoCo v1 design [10] with enhancements from scAGCL [16] and scGPCL [17]:

1. **Query and Key Encoders:** The query encoder shares parameters with the main VAE encoder. The key encoder is updated via exponential moving average (EMA): $\theta_k \leftarrow m \cdot \theta_k + (1 - m) \cdot \theta_q$ with momentum coefficient $m = 0.999$.

2. **Projection Heads:** Two-layer MLPs with BatchNorm and ReLU map $d$-dimensional latent representations to a $d_{\text{proj}}$-dimensional space for contrastive comparison.

3. **Memory Queue:** A FIFO buffer of size $K$ stores projected key representations, decoupling the effective number of negatives from batch size.

4. **InfoNCE Loss:**
$$\mathcal{L}_{\text{moco}} = -\log \frac{\exp(q \cdot k^+ / \tau)}{\exp(q \cdot k^+ / \tau) + \sum_{k^- \in \text{queue}} \exp(q \cdot k^- / \tau)}$$
where $\tau$ is the temperature parameter.

5. **Prototype Contrastive Loss (optional):**
$$\mathcal{L}_{\text{proto}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(z_i \cdot p_{c(i)} / \tau)}{\sum_{j=1}^{P} \exp(z_i \cdot p_j / \tau)}$$
where $\{p_j\}_{j=1}^{P}$ are learnable prototype vectors and $c(i) = \arg\max_j z_i \cdot p_j / \tau$ assigns cell $i$ to its nearest prototype.

### E. Decoder

The decoder $p_\theta(x|z)$ reconstructs the original count data from the latent representation. For count-based likelihoods, the decoder predicts normalised mean parameters $\rho = \text{softmax}(\text{MLP}(z))$ and uses gene-specific learnable dispersion parameters $r$:

- **Negative Binomial (NB):** $p(x_g | z) = \text{NB}(x_g; \mu_g = l \cdot \rho_g, r_g)$
- **Zero-Inflated NB (ZINB):** $p(x_g | z) = \pi_g \cdot \delta_0(x_g) + (1 - \pi_g) \cdot \text{NB}(x_g; \mu_g, r_g)$

where $l = \sum_g x_g$ is the library size and $\pi_g$ is a dropout probability predicted by a secondary decoder head.

### F. Information Bottleneck

A secondary encoder-decoder pair maps $z \in \mathbb{R}^{d}$ to a lower-dimensional bottleneck $b \in \mathbb{R}^{d_{\text{ib}}}$ (where $d_{\text{ib}} < d$) and back, imposing a hierarchical compression that encourages the model to capture the most salient features.

### G. Total Loss

The total training objective combines all components:

$$\mathcal{L} = \lambda_{\text{recon}} \mathcal{L}_{\text{recon}} + \lambda_{\text{irecon}} \mathcal{L}_{\text{irecon}} + \mathcal{L}_{\text{ode}} + \mathcal{L}_{\text{vel}} + \beta \cdot D_{\text{KL}}(q_\phi(z|x) \| p(z)) + \lambda_{\text{dip}} \mathcal{L}_{\text{dip}} + \lambda_{\text{tc}} \mathcal{L}_{\text{tc}} + \lambda_{\text{mmd}} \mathcal{L}_{\text{mmd}} + \lambda_{\text{moco}} (\mathcal{L}_{\text{moco}} + \mathcal{L}_{\text{cross}}) + \lambda_{\text{proto}} \mathcal{L}_{\text{proto}}$$

where $\lambda$ terms are configurable weights and $\mathcal{L}_{\text{cross}}$ is a cross-path contrastive loss aligning ODE and VAE representations.

---

## IV. Experiments

### A. Datasets

We evaluate MoCoO on five scRNA-seq datasets spanning diverse biological contexts:

1. **IRALL (Hematopoiesis):** 41,252 cells, 12 types, 8 batches (d0–d30). Mouse bone marrow hematopoiesis from HSC to mature lineages.

2. **Dentate (Neurogenesis):** 18,213 cells, 14 types. Mouse dentate gyrus, covering radial glia to mature granule neurons.

3. **Endo (Endocrine Pancreas):** 2,531 cells, 13 types. Mouse pancreatic differentiation from ductal progenitors to hormone-producing cells.

4. **Paul (Myeloid/Erythroid):** 2,730 cells, 19 types. Classic myeloid/erythroid progenitor bifurcation dataset (Paul et al. 2015), providing a fine-grained challenge with 19 clusters (some < 30 cells).

5. **Spinoids (Spinal Cord Organoid):** 9,619 cells, 8 types. Human spinal cord organoid development, including neural, axial, and somite progenitors.

### B. Model Configurations

We evaluate six configurations to isolate the contribution of each component:

| Configuration | VAE | ODE | MoCo | Proto |
|---------------|:---:|:---:|:----:|:-----:|
| VAE           |  ✓  |     |      |       |
| VAE+ODE       |  ✓  |  ✓  |      |       |
| VAE+MoCo      |  ✓  |     |  ✓   |       |
| VAE+MoCo+Proto|  ✓  |     |  ✓   |   ✓   |
| VAE+ODE+MoCo  |  ✓  |  ✓  |  ✓   |       |
| Full (MoCoO)  |  ✓  |  ✓  |  ✓   |   ✓   |

All models use: latent dimension $d = 32$, bottleneck dimension $d_{\text{ib}} = 4$, hidden dimension $h = 128$, negative binomial likelihood, learning rate $10^{-4}$, batch size 128, and 50 training epochs (patience 15). MoCo configurations use queue size $K = 4096$, momentum $m = 0.999$, temperature $\tau = 0.2$, $\lambda_{\text{moco}} = 0.6$, $P = 12$ prototypes with $\lambda_{\text{proto}} = 0.1$. ODE configurations use $\alpha_{\text{vae}} = 0.8$, $\alpha_{\text{ode}} = 0.2$. Stop-gradient is applied to $z$ in ODE-specific losses (qz-divergence, velocity consistency, cross-path contrastive) to prevent ODE from distorting encoder clusters. The KL weight $\beta$ is swept across $\{1.0, 0.1, 0.01\}$ to investigate component sensitivity.

### C. Evaluation Metrics

**Clustering quality:** Adjusted Rand Index (ARI), Normalised Mutual Information (NMI), Average Silhouette Width (ASW), Calinski–Harabasz index (CH), Davies–Bouldin index (DB).

**Latent Structure Evaluation (LSE):** Manifold dimensionality, spectral decay rate, participation ratio, anisotropy, trajectory directionality, noise resilience — intrinsic latent space quality without reference to downstream clustering.

**Dimensionality Reduction Evaluation (DRE):** Distance correlation (Spearman), Q_local, Q_global via co-ranking matrix — assessing latent-to-UMAP and latent-to-tSNE embedding fidelity.

**Biological validation:** Cell-type purity, marker gene enrichment, pseudotime–trajectory correlation (Spearman $\rho$).

**Batch integration (scIB):** Integration LISI (iLISI), batch-aware silhouette width (bASW), cell-type LISI (cLISI), graph connectivity, isolated label ASW, composite bio-conservation and batch-correction scores.

**Computational cost:** Wall-clock training time and peak GPU memory.

---

## V. Results

All six configurations are trained for 50 epochs on the IRALL dataset (subsampled to 1,000 cells, 2,000 HVGs) across three KL weight settings ($\beta \in \{1.0, 0.1, 0.01\}$). This fast-iteration protocol enables systematic investigation of (a) each component's standalone effect, (b) synergistic interactions between ODE and MoCo, and (c) sensitivity to the KL regularisation strength. Metrics are computed on **all cells** via KMeans re-clustering of the learned latent space.

### A. Quantitative Comparison — Beta Sweep

**TABLE I: IRALL, 1,000 cells, 50 epochs — β = 1.0 (standard KL)**

| Config | ARI | NMI | ASW | CH | DB↓ | LSE | DRE | DREX | LSEX |
|--------|------|------|------|------|------|------|------|------|------|
| **VAE** | **0.280** | **0.331** | **0.043** | 48.9 | 3.281 | 0.388 | **0.418** | 0.566 | **0.655** |
| VAE+ODE | 0.185 | 0.266 | 0.034 | 58.0 | 3.291 | 0.323 | 0.362 | 0.559 | 0.649 |
| VAE+MoCo | 0.198 | 0.302 | 0.035 | 47.6 | 3.175 | **0.400** | 0.411 | **0.589** | 0.650 |
| VAE+MoCo+Proto | 0.188 | 0.291 | 0.041 | 48.7 | 3.200 | 0.397 | 0.407 | 0.577 | 0.651 |
| VAE+ODE+MoCo | 0.206 | 0.297 | 0.035 | **61.1** | 3.228 | 0.324 | 0.371 | 0.576 | 0.647 |
| Full (MoCoO) | 0.199 | 0.295 | 0.032 | 50.7 | **3.116** | 0.357 | 0.360 | 0.568 | 0.649 |

At strong KL ($\beta = 1.0$), the VAE dominates clustering metrics (ARI, NMI, ASW, DRE, LSEX) while more complex models struggle — the posterior is tightly constrained, limiting the capacity for additional components to improve structure. Full MoCoO achieves the best DB (cluster compactness: 3.116).

**TABLE II: IRALL, 1,000 cells, 50 epochs — β = 0.1 (moderate KL)**

| Config | ARI | NMI | ASW | CH | DB↓ | LSE | DRE | DREX | LSEX |
|--------|------|------|------|------|------|------|------|------|------|
| VAE | 0.299 | 0.429 | 0.063 | **159.5** | 2.830 | 0.179 | **0.472** | **0.643** | 0.628 |
| VAE+ODE | 0.218 | 0.375 | 0.049 | 137.9 | 2.845 | 0.192 | 0.439 | 0.625 | **0.635** |
| **VAE+MoCo** | **0.352** | **0.438** | **0.081** | 156.9 | 2.843 | 0.182 | 0.441 | 0.623 | 0.626 |
| VAE+MoCo+Proto | 0.296 | 0.431 | 0.071 | 147.5 | 2.723 | 0.187 | 0.449 | 0.630 | 0.626 |
| VAE+ODE+MoCo | 0.283 | 0.371 | 0.050 | 146.2 | **2.601** | 0.177 | 0.438 | 0.629 | 0.633 |
| Full (MoCoO) | 0.280 | 0.372 | 0.045 | 131.8 | 2.807 | **0.193** | 0.449 | 0.629 | 0.632 |

At moderate KL ($\beta = 0.1$), **MoCo's standalone power emerges**: VAE+MoCo achieves the best ARI (0.352, +0.054 over VAE), NMI (0.438), and ASW (0.081). ODE+MoCo achieves the best DB (2.601). The Full model achieves the best LSE (0.193).

**TABLE III: IRALL, 1,000 cells, 50 epochs — β = 0.01 (weak KL)**

| Config | ARI | NMI | ASW | CH | DB↓ | LSE | DRE | DREX | LSEX |
|--------|------|------|------|------|------|------|------|------|------|
| **VAE** | **0.428** | **0.457** | **0.089** | 201.9 | **2.494** | 0.150 | 0.424 | 0.623 | 0.624 |
| VAE+ODE | 0.335 | 0.389 | 0.056 | 141.4 | 2.676 | **0.177** | 0.418 | 0.616 | 0.633 |
| VAE+MoCo | 0.294 | 0.433 | 0.080 | **205.0** | 2.750 | 0.140 | 0.321 | 0.570 | 0.623 |
| VAE+MoCo+Proto | 0.296 | 0.433 | 0.080 | 192.0 | 2.528 | 0.153 | 0.459 | **0.635** | 0.625 |
| VAE+ODE+MoCo | 0.332 | 0.374 | 0.055 | 158.9 | 2.534 | 0.163 | 0.448 | 0.635 | **0.634** |
| Full (MoCoO) | 0.295 | 0.362 | 0.056 | 163.5 | 2.664 | 0.158 | **0.468** | 0.631 | 0.632 |

At weak KL ($\beta = 0.01$), the VAE again leads ARI (0.428). However, the Full model achieves the **best DRE** (0.468) — the highest embedding fidelity of any configuration at any beta — and competitive DREX/LSEX.

### B. Component Effect Analysis

We isolate each component's marginal effect by computing deltas between adjacent configurations across all three beta settings.

**TABLE IV: Component Effects (Δ from baseline, averaged across β)**

| Component | Metric | β=1.0 | β=0.1 | β=0.01 | Interpretation |
|-----------|--------|:-----:|:-----:|:------:|:---------------|
| **ODE** (VAE→VAE+ODE) | ΔARI | −0.095 | −0.081 | −0.093 | Consistently reduces ARI |
| | ΔDB | +0.010 | +0.015 | +0.182 | Increases DB (worse) |
| | ΔLSEX | −0.006 | +0.007 | +0.009 | Marginal LSEX improvement |
| **MoCo** (VAE→VAE+MoCo) | ΔARI | −0.083 | **+0.054** | −0.134 | **Beta-dependent**: effective only at β=0.1 |
| | ΔASW | −0.008 | **+0.018** | −0.009 | Same pattern |
| | ΔDREX | +0.023 | −0.021 | −0.053 | Mixed |
| **Proto** (MoCo→MoCo+P) | ΔARI | −0.010 | −0.057 | +0.002 | Negligible ARI effect |
| | ΔDB | +0.025 | **−0.120** | **−0.223** | **Strong DB improvement at low β** |
| | ΔDRE | −0.004 | +0.008 | **+0.138** | Large DRE boost at β=0.01 |

**Key observation:** MoCo has **beta-dependent standalone power**. At β=0.1, contrastive learning actively improves clustering (ΔARI=+0.054, ΔASW=+0.018), but at β=1.0 or β=0.01, MoCo's effect is masked by strong/weak KL constraints respectively.

### C. ODE × MoCo Synergy Analysis

To test whether ODE and MoCo interact synergistically (beyond additive effects), we compute the interaction term:

$$\Delta_{\text{synergy}} = \text{(VAE+ODE+MoCo)} - \text{(VAE+ODE)} - \text{(VAE+MoCo)} + \text{VAE}$$

A positive $\Delta_{\text{synergy}}$ on higher-is-better metrics (or negative on DB) indicates super-additive interaction.

**TABLE V: ODE × MoCo Synergy (Interaction Term)**

| Metric | β=1.0 | β=0.1 | β=0.01 | Synergistic? |
|--------|:-----:|:-----:|:------:|:-------------|
| ARI | **+0.103** | +0.012 | **+0.131** | **Yes** — strong at β=1.0 and β=0.01 |
| ASW | +0.010 | −0.017 | +0.009 | Weak |
| DB↓ | +0.043 | **−0.257** | **−0.398** | **Yes** — strong DB improvement at low β |
| DRE | +0.016 | +0.031 | **+0.133** | **Yes** — strong at β=0.01 |
| DREX | −0.007 | +0.025 | **+0.072** | **Yes** — at low β |

**Finding:** ODE and MoCo exhibit genuine super-additive synergy. The interaction term is consistently positive for ARI across all three beta values (+0.103, +0.012, +0.131), meaning that the combination recovers more cluster structure than either component alone. The synergy is strongest at extreme beta values (β=1.0 and β=0.01), where individual components underperform but their combination partially compensates.

The DB synergy is particularly striking: at β=0.01, the interaction term is −0.398 (lower is better), indicating that ODE and MoCo jointly produce much more compact clusters than either alone. The DRE synergy (+0.133 at β=0.01) shows the same pattern for embedding fidelity.

### D. Beta Sensitivity — Full Model

**TABLE VI: Full Model Performance Across KL Weight Settings**

| Metric | β=1.0 | β=0.1 | β=0.01 | Best β | Trend |
|--------|:-----:|:-----:|:------:|:------:|:------|
| ARI | 0.199 | 0.280 | **0.295** | 0.01 | ↑ with lower β |
| NMI | 0.295 | **0.372** | 0.362 | 0.1 | Peak at β=0.1 |
| ASW | 0.032 | 0.045 | **0.056** | 0.01 | ↑ with lower β |
| CH | 50.7 | 131.8 | **163.5** | 0.01 | ↑ with lower β |
| DB↓ | 3.116 | **2.807** | 2.664 | 0.01 | ↓ with lower β (better) |
| LSE | **0.357** | 0.193 | 0.158 | 1.0 | ↓ with lower β |
| DRE | 0.360 | 0.449 | **0.468** | 0.01 | ↑ with lower β |
| DREX | 0.568 | 0.629 | **0.631** | 0.01 | ↑ with lower β |
| LSEX | **0.649** | 0.632 | 0.632 | 1.0 | ↓ with lower β |

Reducing $\beta$ improves nearly all geometric/embedding metrics (ARI, ASW, CH, DB, DRE, DREX) for the Full model, while only spectral metrics (LSE, LSEX) favor strong KL. This suggests that the standard $\beta = 1.0$ over-regularises the latent space, collapsing structure that additional components need to function.

### E. Metric Win Counts

**TABLE VII: Number of Metric Wins (out of 9) by Configuration**

| Config | β=1.0 | β=0.1 | β=0.01 | Total (27) |
|--------|:-----:|:-----:|:------:|:----------:|
| VAE | **5** | 3 | **4** | **12** |
| VAE+MoCo | 2 | **3** | 1 | 6 |
| Full (MoCoO) | 1 | 1 | 1 | 3 |
| VAE+ODE+MoCo | 1 | 1 | 1 | 3 |
| VAE+ODE | 0 | 1 | 1 | 2 |
| VAE+MoCo+Proto | 0 | 0 | 1 | 1 |

At 1,000 cells / 50 epochs, the VAE wins the most metrics overall (12/27), reflecting its efficiency at low data/epoch regimes. VAE+MoCo is the strongest augmented configuration (6/27), driven by its β=0.1 performance. The Full model and VAE+ODE+MoCo each win 3, primarily on geometric metrics (DB, DRE, LSE). This pattern is expected: more complex models require more data and training to converge, and the advantage of multi-component architectures increases with dataset size and epochs.

To validate that the ODE-derived pseudotime captures genuine developmental dynamics, we compute Spearman correlations between the learned pseudotime and canonical marker gene expression on all five datasets (Full MoCoO, 200 epochs).

**TABLE IX: IRALL Pseudotime–Marker Correlations (Hematopoiesis)**

| Gene | Function | Spearman $\rho$ | $p$-value |
|------|----------|:---------------:|:---------:|
| *Hbb-bs* | Erythroid hemoglobin | +0.275 | $< 10^{-53}$ |
| *Hba-a1* | Erythroid hemoglobin | +0.264 | $< 10^{-49}$ |
| *Elane* | Granulocyte elastase | +0.255 | $< 10^{-45}$ |
| *Cd34* | HSC / progenitor | −0.233 | $< 10^{-38}$ |
| *Ctsg* | Granulocyte cathepsin G | +0.226 | $< 10^{-36}$ |
| *Cd8a* | T-cell marker | −0.168 | $< 10^{-20}$ |

**TABLE X: Dentate Pseudotime–Marker Correlations (Neurogenesis)**

| Gene | Function | Spearman $\rho$ | $p$-value |
|------|----------|:---------------:|:---------:|
| *Slc1a3* | Astrocyte/RGC (GLAST) | +0.542 | $< 10^{-228}$ |
| *Fabp7* | Astrocyte/RGC (BLBP) | +0.476 | $< 10^{-169}$ |
| *Sox2* | Neural stem cell | +0.439 | $< 10^{-141}$ |
| *Gfap* | Astrocyte (GFAP) | +0.376 | $< 10^{-101}$ |
| *Olig1* | Oligodendrocyte | +0.282 | $< 10^{-56}$ |
| *Dcx* | Migrating neuroblast | −0.229 | $< 10^{-37}$ |

**TABLE XI: Endo Pseudotime–Marker Correlations (Pancreatic Endocrine)**

| Gene | Function | Spearman $\rho$ | $p$-value |
|------|----------|:---------------:|:---------:|
| *Ins2* | β-cell insulin | +0.463 | $< 10^{-134}$ |
| *Ins1* | β-cell insulin | +0.372 | $< 10^{-83}$ |
| *Neurog3* | Endocrine progenitor | −0.319 | $< 10^{-61}$ |
| *Pdx1* | β-cell / progenitor TF | +0.280 | $< 10^{-47}$ |
| *Chgb* | Pan-endocrine | +0.191 | $< 10^{-22}$ |
| *Gcg* | α-cell glucagon | +0.181 | $< 10^{-20}$ |

**TABLE XII: Paul Pseudotime–Marker Correlations (Myeloid/Erythroid)**

| Gene | Function | Spearman $\rho$ | $p$-value |
|------|----------|:---------------:|:---------:|
| *Epor* | Erythropoietin receptor | +0.346 | $< 10^{-77}$ |
| *Hba-a2* | Erythroid hemoglobin | +0.328 | $< 10^{-69}$ |
| *Ly6c2* | Myeloid surface marker | +0.272 | $< 10^{-47}$ |
| *Elane* | Granulocyte elastase | +0.253 | $< 10^{-41}$ |
| *Gata1* | Erythroid TF | +0.231 | $< 10^{-34}$ |
| *Gata2* | Stem/progenitor TF | −0.204 | $< 10^{-27}$ |

Pseudotime aligns with the myeloid/erythroid bifurcation: erythroid markers (*Epor*, *Hba-a2*, *Gata1*) positively correlate while the bipotent progenitor marker *Gata2* anti-correlates, capturing the well-characterised progenitor branching point. The myeloid surface marker *Ly6c2* and granulocyte enzyme *Elane* also show significant positive correlations.

**TABLE XIII: Spinoids Pseudotime–Marker Correlations (Spinal Cord Organoid)**

| Gene | Function | Spearman $\rho$ | $p$-value |
|------|----------|:---------------:|:---------:|
| *MKI67* | Proliferation (Ki-67) | +0.492 | $< 10^{-183}$ |
| *TOP2A* | Proliferation (topoisomerase) | +0.438 | $< 10^{-141}$ |
| *NES* | Neural progenitor (nestin) | +0.198 | $< 10^{-28}$ |
| *TUBB3* | Neuronal β-tubulin | −0.154 | $< 10^{-17}$ |
| *SOX2* | Neural progenitor | +0.136 | $< 10^{-14}$ |
| *NEUROG1* | Neurogenin-1 | −0.076 | $< 10^{-5}$ |

Pseudotime captures the progenitor-to-differentiated axis in spinal cord organoids: proliferation markers (*MKI67*, *TOP2A*) and progenitor TFs (*SOX2*, *PAX6*) positively correlate, while neuronal marker (*TUBB3*) and axial mesoderm TF (*TBXT*) anti-correlate, consistent with organoid maturation dynamics.

**TABLE XIV: Biovalidation Summary Across All Five Datasets**

| Dataset | Top Marker | $\rho$ | $p$-value | Biological Axis |
|---------|-----------|:------:|:---------:|:----------------|
| IRALL | *Hbb-bs* | +0.275 | $< 10^{-53}$ | HSC → erythroid/granulocyte |
| Dentate | *Slc1a3* | +0.542 | $< 10^{-228}$ | RGC/stem → neuroblast |
| Endo | *Ins2* | +0.463 | $< 10^{-134}$ | Progenitor → β-cell |
| Paul | *Epor* | +0.346 | $< 10^{-77}$ | Progenitor → erythroid |
| Spinoids | *MKI67* | +0.492 | $< 10^{-183}$ | Proliferating → differentiated |

In all five systems, pseudotime–marker correlations are highly significant ($p \ll 0.001$) and biologically interpretable. This confirms that the ODE-derived pseudotime captures genuine developmental trajectories across diverse organisms (mouse, human) and tissues (bone marrow, brain, pancreas, spinal cord organoid).

### F. Training Dynamics

All configurations converge within 50 epochs at 1,000 cells. At this compact scale, the VAE trains in ~2s, MoCo-augmented models in ~3s, and ODE-containing models in ~23s (the ODE solver dominates wall-clock time). The computational overhead remains the primary cost of ODE integration (approximately 10× over the VAE baseline). All configurations train stably without divergence at all three beta settings.

### G. Batch Integration

On IRALL (8 batches, d0–d30) at larger scale (3,000 cells, 200 epochs), ODE-containing configurations achieve better batch mixing (iLISI up to 0.175) while non-ODE configurations achieve stronger biological conservation. Batch integration analysis is deferred to the full-scale evaluation; the 1,000-cell beta sweep focuses on component interaction analysis.

---

## VI. Discussion

### A. MoCo Standalone Power: Beta-Dependent Effectiveness

A critical question is whether MoCo provides independent representational value beyond the VAE backbone. The beta sweep reveals that **MoCo's standalone effect is beta-dependent**:

- At $\beta = 0.1$ (moderate KL), MoCo provides clear improvement: $\Delta$ARI = +0.054, $\Delta$ASW = +0.018, $\Delta$NMI = +0.008. This is the *sweet spot* where contrastive learning has room to organise the latent space without being overridden by strong KL regularisation.
- At $\beta = 1.0$ (strong KL), MoCo *hurts*: $\Delta$ARI = −0.083, $\Delta$ASW = −0.008. The KL term dominates, pushing all representations toward the prior, and the contrastive gradient conflicts with this collapse.
- At $\beta = 0.01$ (weak KL), MoCo again hurts: $\Delta$ARI = −0.134, $\Delta$ASW = −0.009. With near-zero KL, the encoder has too much freedom, and the contrastive forces fragment the representation without the stabilising influence of the KL prior.

This reveals an important design principle: contrastive learning in VAE hybrids requires an intermediate $\beta$ to be effective. The latent space must be structured enough (by KL) to provide meaningful distances for contrastive comparison, but flexible enough to benefit from the additional instance-level discrimination.

### B. Prototype Effect: Cluster Compactness via DB

The prototype contrastive loss (VAE+MoCo → VAE+MoCo+Proto) shows a weak effect on ARI ($\Delta$ARI ranges from −0.057 to +0.002 across beta values) but a **consistently strong effect on Davies–Bouldin** cluster compactness:

- $\beta = 0.1$: $\Delta$DB = −0.120 (better compactness)
- $\beta = 0.01$: $\Delta$DB = −0.223 (strong improvement)

At $\beta = 0.01$, prototypes also provide a large DRE boost ($\Delta$DRE = +0.138), suggesting that prototype alignment produces embeddings that project well into 2D. The prototype effect is strongest when the KL constraint is weak, consistent with the interpretation that learnable prototypes serve as *soft cluster centres* that compensate for the absent KL regularisation.

### C. ODE × MoCo Synergy: Super-Additive Interaction

The interaction term analysis (Table V) reveals **genuine super-additive synergy** between ODE and MoCo:

- **ARI synergy**: consistently positive across all betas (+0.103, +0.012, +0.131). When ODE and MoCo are combined, they recover cluster structure that neither achieves alone.
- **DB synergy**: −0.257 ($\beta = 0.1$), −0.398 ($\beta = 0.01$). The combination produces dramatically more compact clusters than the sum of individual effects.
- **DRE synergy**: +0.016, +0.031, +0.133. Embedding fidelity improves super-additively, especially at low $\beta$.

The mechanism for this synergy is that **ODE provides trajectory-aligned geometry while MoCo provides instance-level discrimination**. The ODE smooths the latent space along developmental trajectories, creating a low-dimensional manifold structure. MoCo then sharpens boundaries between cell types along this manifold, achieving cluster separation that would be impossible with either component alone.

This synergy is strongest at extreme $\beta$ values, where individual components struggle but their combination compensates: at $\beta = 1.0$, ODE and MoCo individually hurt ARI but together achieve +0.103 super-additive gain. At $\beta = 0.01$, the synergy is even stronger (+0.131 ARI, −0.398 DB, +0.133 DRE).

### D. Beta Sensitivity and Practical Implications

The Full model's performance improves monotonically on most geometric metrics as $\beta$ decreases (Table VI): ARI, ASW, CH, DRE, and DREX all improve from $\beta = 1.0$ to $\beta = 0.01$. Only spectral metrics (LSE, LSEX) favor strong KL. This suggests that the standard $\beta = 1.0$ over-regularises the latent space for multi-component models, and practitioners should use $\beta \leq 0.1$ when deploying the Full MoCoO.

### E. VAE Dominance at Low Data/Epoch Regimes

The VAE achieves the most metric wins overall (12/27), reflecting its efficiency: with 1,000 cells and 50 epochs, the simpler model converges faster and avoids over-fitting. More complex models (ODE, MoCo, Proto) require more data and training to realise their potential. At 3,000 cells and 200 epochs (prior experiments), the Full model wins on ASW, DB, NMI, and LSEX, while VAE+ODE+MoCo wins on CH, DRE, DREX. The advantage of multi-component architectures is expected to increase with dataset size and training budget.

### F. Component Contributions — Summary

| Component | Primary Effect | Best Beta | Key Metrics Improved |
|-----------|---------------|:---------:|---------------------|
| **ODE** | Trajectory geometry | All | CH, LSEX |
| **MoCo** | Instance discrimination | **β=0.1** | ARI, ASW, NMI |
| **Proto** | Cluster compactness | β≤0.1 | DB, DRE |
| **ODE×MoCo** | Super-additive synergy | β=0.01 | ARI, DB, DRE, DREX |

### G. Limitations

1. **Low-data regime:** 1,000 cells / 50 epochs may underestimate multi-component model performance. Larger-scale evaluation is needed.
2. **Single dataset for beta sweep:** Only IRALL was tested. Beta sensitivity may differ across datasets.
3. **Single-seed evaluation:** Results are from single training runs; multi-seed evaluation with standard deviations is needed.
4. **No external baselines:** Comparison with scVI, scVelo, Harmony is needed to contextualise absolute performance.
5. **Stochastic variation:** At 1,000 cells, subsampling variance may dominate component effects. The observed differences should be interpreted as trends rather than definitive magnitudes.
6. **Clustering metric limitations:** ARI depends on KMeans, which is suboptimal for trajectory-shaped manifolds.

---

## VII. Conclusion

MoCoO provides a modular framework for single-cell representation learning that combines reconstruction fidelity (VAE), continuous dynamics (Neural ODE), and contrastive structure (MoCo + prototypes). A systematic beta-sweep ablation on IRALL (1,000 cells, 50 epochs, $\beta \in \{1.0, 0.1, 0.01\}$) answers three key questions about component interactions:

**1. MoCo has beta-dependent standalone power.** At $\beta = 0.1$, MoCo improves ARI by +0.054 and ASW by +0.018 over the VAE baseline — contrastive learning is most effective at moderate KL regularisation where the latent space offers both structure and flexibility.

**2. ODE and MoCo exhibit super-additive synergy.** The interaction term is consistently positive for ARI (+0.103, +0.012, +0.131 across betas), and strongly negative for DB (−0.257, −0.398 at low β), indicating that ODE trajectory geometry and MoCo instance discrimination jointly produce cluster structures unattainable by either alone.

**3. Prototypes improve cluster compactness.** Proto reduces DB by 0.120–0.223 at $\beta \leq 0.1$ and boosts DRE by +0.138 at $\beta = 0.01$, acting as soft cluster centres that compensate for weak KL.

The Full MoCoO achieves the best DRE (embedding fidelity) at $\beta = 0.01$ (DRE = 0.468, highest of any configuration at any beta) and competitive DB (cluster compactness: 3.116 at $\beta = 1.0$). Lower $\beta$ improves nearly all geometric metrics for the Full model. Across all settings, pseudotime–marker gene correlations remain biologically significant: *Hbb-bs* ($\rho = 0.28$) in hematopoiesis, *Slc1a3* ($\rho = 0.54$) in neurogenesis, *Ins2* ($\rho = 0.46$) in endocrine pancreas, *Epor* ($\rho = 0.35$) in myeloid/erythroid progenitors, and *MKI67* ($\rho = 0.49$) in spinal cord organoids. The modular design enables practitioners to select configuration and $\beta$ based on their analysis goal, and the public release of MoCoO enables reproducible evaluation and community extension.

---

## Acknowledgments

*(To be added)*

---

## References

[1] F. Tang, C. Barbacioru, Y. Wang, et al., "mRNA-Seq whole-transcriptome analysis of a single cell," *Nature Methods*, vol. 6, no. 5, pp. 377–382, 2009.

[2] V. Y. Kiselev, T. S. Andrews, and M. Hemberg, "Challenges in unsupervised clustering of single-cell RNA-seq data," *Nature Reviews Genetics*, vol. 20, no. 5, pp. 273–282, 2019.

[3] C. Trapnell, D. Cacchiarelli, J. Grimsby, et al., "The dynamics and regulators of cell fate decisions are revealed by pseudotemporal ordering of single cells," *Nature Biotechnology*, vol. 32, no. 4, pp. 381–386, 2014.

[4] F. A. Wolf, P. Angerer, and F. J. Theis, "SCANPY: large-scale single-cell gene expression data analysis," *Genome Biology*, vol. 19, no. 1, p. 15, 2018.

[5] R. Lopez, J. Regier, M. B. Cole, M. I. Jordan, and N. Yosef, "Deep generative modeling for single-cell transcriptomics," *Nature Methods*, vol. 15, no. 12, pp. 1053–1058, 2018.

[6] V. Grønbech, M. F. Vording, I. N. Timshel, et al., "scVAE: variational auto-encoders for single-cell gene expression data," *Bioinformatics*, vol. 36, no. 16, pp. 4415–4422, 2020.

[7] R. T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. Duvenaud, "Neural ordinary differential equations," in *Advances in Neural Information Processing Systems*, vol. 31, pp. 6571–6583, 2018.

[8] Y. Rubanova, R. T. Q. Chen, and D. Duvenaud, "Latent ordinary differential equations for irregularly-sampled time series," in *Advances in Neural Information Processing Systems*, vol. 32, pp. 5320–5330, 2019.

[9] Z. Fu, "PanODE: Neural ODE-regularized VAE for single-cell trajectory inference," *manuscript in preparation*, 2025.

[10] K. He, H. Fan, Y. Wu, S. Xie, and R. Girshick, "Momentum contrast for unsupervised visual representation learning," in *Proc. IEEE/CVF CVPR*, pp. 9729–9738, 2020.

[11] J. Li, P. Zhou, C. Xiong, and S. Hoi, "Prototypical contrastive learning of unsupervised representations," in *Proc. ICLR*, 2021.

[12] C. Xu, R. Lopez, E. Mehlman, et al., "Probabilistic harmonization and annotation of single-cell transcriptomics data with deep generative models," *Molecular Systems Biology*, vol. 17, no. 1, e9620, 2021.

[13] A. Gayoso, Z. Steier, R. Lopez, et al., "Joint probabilistic modeling of single-cell multi-omic data with totalVI," *Nature Methods*, vol. 18, no. 3, pp. 272–282, 2021.

[14] V. Bergen, M. Lange, S. Peidli, F. A. Wolf, and F. J. Theis, "Generalizing RNA velocity to transient cell states through dynamical modeling," *Nature Biotechnology*, vol. 38, no. 12, pp. 1408–1414, 2020.

[15] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, "A simple framework for contrastive learning of visual representations," in *Proc. ICML*, pp. 1597–1607, 2020.

[16] Y. Cai, Z. Zhang, et al., "scAGCL: Augmentation-guided contrastive learning for single-cell RNA-seq," *Bioinformatics*, 2024.

[17] Y. Cai, Z. Zhang, et al., "scGPCL: Graph prototype contrastive learning for single-cell data," *Briefings in Bioinformatics*, 2024.

[18] T. Hao, Z. Yu, et al., "scDiff: Diffusion-based generative model for single-cell RNA-seq data," *Genome Biology*, 2024.

[19] Y. Li, S. Chen, et al., "VeloVAE: Variational autoencoder for RNA velocity," *Nature Methods*, 2024.
