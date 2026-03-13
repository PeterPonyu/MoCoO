"""Metric display metadata for plotting and table generation.

Each entry is a tuple of (metric_key, display_label, higher_is_better).
"""

# Core metrics panel
CORE_METRICS = [
    ("NMI", "NMI \u2191", True),
    ("ARI", "ARI \u2191", True),
    ("ASW", "ASW \u2191", True),
    ("DAV", "DAV \u2193", False),
    ("DRE_umap_overall_quality", "DRE UMAP \u2191", True),
    ("LSE_overall_quality", "LSE Overall \u2191", True),
]

# Extended metrics panels
EXT_METRICS_CLUSTERING = [
    ("COR", "Corr \u2191", True),
    ("CAL", "Cal-H \u2191", True),
]

EXT_METRICS_DRE = [
    ("DRE_umap_distance_correlation", "DRE UMAP DistCorr \u2191", True),
    ("DRE_umap_Q_local", "DRE UMAP Qloc \u2191", True),
    ("DRE_umap_Q_global", "DRE UMAP Qglob \u2191", True),
    ("DRE_tsne_distance_correlation", "DRE tSNE DistCorr \u2191", True),
    ("DRE_tsne_Q_local", "DRE tSNE Qloc \u2191", True),
    ("DRE_tsne_Q_global", "DRE tSNE Qglob \u2191", True),
    ("DRE_tsne_overall_quality", "DRE tSNE Overall \u2191", True),
]

EXT_METRICS_LSE = [
    ("LSE_manifold_dimensionality", "LSE ManDim \u2191", True),
    ("LSE_spectral_decay_rate", "LSE SpDecay \u2191", True),
    ("LSE_participation_ratio", "LSE PartRat \u2191", True),
    ("LSE_anisotropy_score", "LSE Aniso \u2193", False),
    ("LSE_noise_resilience", "LSE NoiseR \u2191", True),
    ("LSE_core_quality", "LSE Core \u2191", True),
]

EXT_METRICS_DREX = [
    ("DREX_trustworthiness", "DREX Trust \u2191", True),
    ("DREX_continuity", "DREX Cont \u2191", True),
    ("DREX_distance_spearman", "DREX Spear \u2191", True),
    ("DREX_distance_pearson", "DREX Pearson \u2191", True),
    ("DREX_local_scale_quality", "DREX LocScale \u2191", True),
    ("DREX_neighborhood_symmetry", "DREX NbrSym \u2191", True),
    ("DREX_overall_quality", "DREX Overall \u2191", True),
]

EXT_METRICS_LSEX = [
    ("LSEX_two_hop_connectivity", "LSEX 2Hop \u2191", True),
    ("LSEX_radial_concentration", "LSEX RadConc \u2191", True),
    ("LSEX_local_curvature", "LSEX LocCurv \u2191", True),
    ("LSEX_entropy_stability", "LSEX EntStab \u2191", True),
    ("LSEX_overall_quality", "LSEX Overall \u2191", True),
]

ALL_METRIC_GROUPS = [
    ("Core Clustering", CORE_METRICS),
    ("Extended Clustering", EXT_METRICS_CLUSTERING),
    ("DR Quality (DRE)", EXT_METRICS_DRE),
    ("Latent Structure (LSE)", EXT_METRICS_LSE),
    ("Extended DR (DREX)", EXT_METRICS_DREX),
    ("Extended Latent (LSEX)", EXT_METRICS_LSEX),
]
