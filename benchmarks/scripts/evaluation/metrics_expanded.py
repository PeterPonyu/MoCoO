"""Expanded metrics computation for MoCoO benchmark.

Computes the full metric battery from PanODE-LAB using only standard
scientific Python libraries (sklearn, scipy, numpy, scanpy).

Metric categories
-----------------
1. **Clustering**: NMI, ARI, ASW, DAV, CAL, COR
2. **DRE** (Dimensionality Reduction Eval): distance_correlation, Q_local,
   Q_global, overall_quality — for UMAP and tSNE projections
3. **LSE** (Latent Space Eval): manifold_dimensionality, spectral_decay_rate,
   participation_ratio, anisotropy_score, noise_resilience, core_quality,
   overall_quality
4. **DREX** (Extended DR): trustworthiness, continuity, distance_spearman,
   distance_pearson, neighborhood_symmetry, overall_quality
5. **LSEX** (Extended Latent): two_hop_connectivity, radial_concentration,
   local_curvature, entropy_stability, overall_quality
6. **Latent Diagnostics**: norm, std, variance, pairwise distance stats
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
from sklearn.cluster import KMeans
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# Clustering metrics
# ═══════════════════════════════════════════════════════════════════════════════

def _clustering_metrics(latent, labels):
    """NMI, ARI, ASW, DAV, CAL, COR."""
    latent = np.asarray(latent, dtype=float)
    n_clusters = len(np.unique(labels))
    pred = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(latent)

    m = {
        'NMI': normalized_mutual_info_score(labels, pred),
        'ARI': adjusted_rand_score(labels, pred),
    }

    try:
        m['ASW'] = silhouette_score(latent, pred) if len(np.unique(pred)) > 1 else np.nan
    except Exception:
        m['ASW'] = np.nan
    try:
        m['DAV'] = davies_bouldin_score(latent, pred)
    except Exception:
        m['DAV'] = np.nan
    try:
        m['CAL'] = calinski_harabasz_score(latent, pred)
    except Exception:
        m['CAL'] = np.nan
    try:
        acorr = np.abs(np.corrcoef(latent.T))
        m['COR'] = float(acorr.sum(axis=1).mean() - 1)
    except Exception:
        m['COR'] = np.nan

    return m


# ═══════════════════════════════════════════════════════════════════════════════
# DRE — Dimensionality Reduction Evaluation (latent → 2D projection)
# ═══════════════════════════════════════════════════════════════════════════════

def _knn_indices(X, k):
    """Return k-NN index arrays for rows of X."""
    X = np.asarray(X, dtype=float)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
    nn.fit(X)
    return nn.kneighbors(X, return_distance=False)[:, 1:]  # exclude self


def _q_local(knn_source, knn_target, k):
    """Fraction of k-NN preserved between source and target spaces."""
    n = knn_source.shape[0]
    overlap = 0.0
    for i in range(n):
        s = set(knn_source[i, :k])
        t = set(knn_target[i, :k])
        overlap += len(s & t) / k
    return overlap / n


def _distance_correlation(X_source, X_target, max_samples=2000):
    """Correlation between pairwise distances in two spaces."""
    X_source = np.asarray(X_source, dtype=float)
    X_target = np.asarray(X_target, dtype=float)
    n = X_source.shape[0]
    if n > max_samples:
        idx = np.random.RandomState(42).choice(n, max_samples, replace=False)
        X_source = X_source[idx]
        X_target = X_target[idx]
    d_s = pdist(X_source)
    d_t = pdist(X_target)
    if d_s.std() < 1e-10 or d_t.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(d_s, d_t)[0, 1])


def _dre_metrics(latent, projection_2d, k=15, prefix="DRE_umap"):
    """Compute DRE metrics: distance_correlation, Q_local, Q_global, overall."""
    latent = np.asarray(latent, dtype=float)
    projection_2d = np.asarray(projection_2d, dtype=float)
    m = {}
    try:
        knn_src = _knn_indices(latent, max(k, 50))
        knn_tgt = _knn_indices(projection_2d, max(k, 50))

        m[f'{prefix}_distance_correlation'] = _distance_correlation(latent, projection_2d)
        m[f'{prefix}_Q_local'] = _q_local(knn_src, knn_tgt, k)
        m[f'{prefix}_Q_global'] = _q_local(knn_src, knn_tgt, min(50, knn_src.shape[1]))
        m[f'{prefix}_overall_quality'] = np.mean([
            m[f'{prefix}_distance_correlation'],
            m[f'{prefix}_Q_local'],
            m[f'{prefix}_Q_global'],
        ])
    except Exception:
        for key in ('distance_correlation', 'Q_local', 'Q_global', 'overall_quality'):
            m[f'{prefix}_{key}'] = np.nan
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# LSE — Latent Space Evaluation (intrinsic quality)
# ═══════════════════════════════════════════════════════════════════════════════

def _lse_metrics(latent):
    """Compute LSE metrics from singular value decomposition of latent space."""
    latent = np.asarray(latent, dtype=float)
    m = {}
    try:
        z = latent - latent.mean(axis=0)
        _, s, _ = np.linalg.svd(z, full_matrices=False)
        s = np.maximum(s, 0)

        # Manifold dimensionality: participation ratio
        p = s ** 2 / (s ** 2).sum()
        participation_ratio = 1.0 / np.sum(p ** 2) if np.sum(p ** 2) > 0 else 0
        m['LSE_manifold_dimensionality'] = participation_ratio / latent.shape[1]

        # Spectral decay rate
        log_s = np.log(s[s > 1e-10] + 1e-10)
        if len(log_s) > 1:
            x = np.arange(len(log_s))
            slope = np.polyfit(x, log_s, 1)[0]
            m['LSE_spectral_decay_rate'] = max(0, -slope)
        else:
            m['LSE_spectral_decay_rate'] = 0.0

        m['LSE_participation_ratio'] = participation_ratio

        # Anisotropy: ratio of largest to average SV (lower is more isotropic)
        m['LSE_anisotropy_score'] = float(s[0] / (s.mean() + 1e-10))

        # Noise resilience: fraction of variance in top-80% SVs
        cumvar = np.cumsum(s ** 2) / (s ** 2).sum()
        n_sig = np.searchsorted(cumvar, 0.8) + 1
        m['LSE_noise_resilience'] = n_sig / len(s)

        # Core quality: geometric mean of normalized participation ratio and noise resilience
        norm_pr = min(1.0, participation_ratio / latent.shape[1])
        m['LSE_core_quality'] = np.sqrt(norm_pr * m['LSE_noise_resilience'])

        m['LSE_overall_quality'] = np.mean([
            m['LSE_manifold_dimensionality'],
            min(1.0, m['LSE_spectral_decay_rate']),
            m['LSE_noise_resilience'],
            m['LSE_core_quality'],
        ])
    except Exception:
        for key in ('manifold_dimensionality', 'spectral_decay_rate',
                    'participation_ratio', 'anisotropy_score',
                    'noise_resilience', 'core_quality', 'overall_quality'):
            m[f'LSE_{key}'] = np.nan
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# DREX — Extended Dimensionality Reduction metrics
# ═══════════════════════════════════════════════════════════════════════════════

def _trustworthiness(X_high, X_low, k=15):
    """How well the low-D embedding preserves neighborhood structure."""
    X_high = np.asarray(X_high, dtype=float)
    X_low = np.asarray(X_low, dtype=float)
    from sklearn.manifold import trustworthiness as _tw
    return _tw(X_high, X_low, n_neighbors=k)


def _continuity(X_high, X_low, k=15):
    """How well the high-D neighborhoods are continued in low-D."""
    X_high = np.asarray(X_high, dtype=float)
    X_low = np.asarray(X_low, dtype=float)
    # Continuity = trustworthiness with roles reversed
    n = X_high.shape[0]
    nn_high = _knn_indices(X_high, k)
    nn_low = _knn_indices(X_low, k)

    cont = 0.0
    for i in range(n):
        low_set = set(nn_low[i])
        for j_idx, j in enumerate(nn_high[i]):
            if j not in low_set:
                # rank of j in low-D
                cont += 1
    max_cont = n * k * (2 * n - 3 * k - 1)
    if max_cont == 0:
        return 1.0
    return 1.0 - (2.0 / max_cont) * cont


def _drex_metrics(latent, projection_2d, k=15):
    """DREX: trustworthiness, continuity, distance correlations, neighborhood symmetry."""
    latent = np.asarray(latent, dtype=float)
    projection_2d = np.asarray(projection_2d, dtype=float)
    m = {}
    try:
        m['DREX_trustworthiness'] = _trustworthiness(latent, projection_2d, k)
        m['DREX_continuity'] = _continuity(latent, projection_2d, k)

        # Distance correlations (Spearman and Pearson)
        n = min(latent.shape[0], 2000)
        if latent.shape[0] > n:
            idx = np.random.RandomState(42).choice(latent.shape[0], n, replace=False)
            d_h = pdist(latent[idx])
            d_l = pdist(projection_2d[idx])
        else:
            d_h = pdist(latent)
            d_l = pdist(projection_2d)

        m['DREX_distance_spearman'] = float(spearmanr(d_h, d_l).correlation)
        m['DREX_distance_pearson'] = float(pearsonr(d_h, d_l)[0])

        # Local scale quality: correlation of k-NN distances
        nn_h = NearestNeighbors(n_neighbors=k + 1).fit(latent)
        dists_h = nn_h.kneighbors(latent, return_distance=True)[0][:, 1:]
        nn_l = NearestNeighbors(n_neighbors=k + 1).fit(projection_2d)
        dists_l = nn_l.kneighbors(projection_2d, return_distance=True)[0][:, 1:]
        local_h = dists_h.mean(axis=1)
        local_l = dists_l.mean(axis=1)
        m['DREX_local_scale_quality'] = float(spearmanr(local_h, local_l).correlation)

        # Neighborhood symmetry: overlap of kNN in both directions
        knn_h = _knn_indices(latent, k)
        knn_l = _knn_indices(projection_2d, k)
        sym = 0.0
        for i in range(latent.shape[0]):
            s_h = set(knn_h[i])
            s_l = set(knn_l[i])
            sym += len(s_h & s_l) / k
        m['DREX_neighborhood_symmetry'] = sym / latent.shape[0]

        m['DREX_overall_quality'] = np.mean([
            m['DREX_trustworthiness'],
            m['DREX_continuity'],
            max(0, m['DREX_distance_spearman']),
            max(0, m['DREX_distance_pearson']),
            max(0, m['DREX_local_scale_quality']),
            m['DREX_neighborhood_symmetry'],
        ])
    except Exception as e:
        for key in ('trustworthiness', 'continuity', 'distance_spearman',
                    'distance_pearson', 'local_scale_quality',
                    'neighborhood_symmetry', 'overall_quality'):
            m[f'DREX_{key}'] = np.nan
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# LSEX — Extended Latent Space metrics
# ═══════════════════════════════════════════════════════════════════════════════

def _lsex_metrics(latent, k=15):
    """LSEX: two-hop connectivity, radial concentration, local curvature, entropy stability."""
    latent = np.asarray(latent, dtype=float)
    m = {}
    try:
        n = latent.shape[0]
        knn = _knn_indices(latent, k)

        # Two-hop connectivity: fraction of 2-hop neighbors that are unique
        two_hop_unique = 0.0
        for i in range(n):
            one_hop = set(knn[i])
            two_hop = set()
            for j in knn[i]:
                two_hop.update(knn[j])
            two_hop -= one_hop
            two_hop.discard(i)
            two_hop_unique += len(two_hop) / max(1, k * k)
        m['LSEX_two_hop_connectivity'] = two_hop_unique / n

        # Radial concentration: how concentrated neighbors are vs uniform
        dists = NearestNeighbors(n_neighbors=k + 1).fit(latent).kneighbors(
            latent, return_distance=True)[0][:, 1:]
        # Coefficient of variation of neighbor distances
        cv = dists.std(axis=1) / (dists.mean(axis=1) + 1e-10)
        m['LSEX_radial_concentration'] = 1.0 - float(cv.mean())

        # Local curvature: linearity of kNN neighborhoods
        curvature = 0.0
        for i in range(min(n, 2000)):  # subsample for speed
            nbrs = latent[knn[i]]
            center = nbrs.mean(axis=0)
            residuals = nbrs - center
            _, s, _ = np.linalg.svd(residuals, full_matrices=False)
            # High first SV ratio → linear neighborhood → low curvature
            curvature += s[0] / (s.sum() + 1e-10)
        m['LSEX_local_curvature'] = curvature / min(n, 2000)

        # Entropy stability: consistency of neighborhood structure across scales
        q_k = _q_local(knn, knn, k)  # self-consistency baseline
        knn_half = _knn_indices(latent, max(k // 2, 3))
        knn_double = _knn_indices(latent, min(k * 2, n - 1))
        q_half = _q_local(knn_half, _knn_indices(latent, max(k // 2, 3)), max(k // 2, 3))
        m['LSEX_entropy_stability'] = float(np.mean([q_k, q_half]))

        m['LSEX_overall_quality'] = np.mean([
            m['LSEX_two_hop_connectivity'],
            max(0, m['LSEX_radial_concentration']),
            m['LSEX_local_curvature'],
            m['LSEX_entropy_stability'],
        ])
    except Exception:
        for key in ('two_hop_connectivity', 'radial_concentration',
                    'local_curvature', 'entropy_stability', 'overall_quality'):
            m[f'LSEX_{key}'] = np.nan
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# Latent Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_latent_diagnostics(latent, max_samples=2000):
    """Latent collapse and redundancy stats."""
    z = np.asarray(latent, dtype=float)
    std = z.std(axis=0)
    var = z.var(axis=0)

    n = z.shape[0]
    if n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        z_sub = z[idx]
    else:
        z_sub = z

    try:
        dists = pdist(z_sub)
        dist_mean = float(np.mean(dists))
        dist_std = float(np.std(dists))
    except Exception:
        dist_mean = dist_std = np.nan

    return {
        'diag_mean_norm': float(np.linalg.norm(z.mean(axis=0))),
        'diag_std_mean': float(std.mean()),
        'diag_std_min': float(std.min()),
        'diag_std_max': float(std.max()),
        'diag_var_mean': float(var.mean()),
        'diag_near_zero_dims': int((std < 1e-3).sum()),
        'diag_pairwise_dist_mean': dist_mean,
        'diag_pairwise_dist_std': dist_std,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2D projections helper
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_2d_projections(latent):
    """Compute UMAP and tSNE embeddings for DRE/DREX evaluation."""
    latent = np.asarray(latent, dtype=float)
    import scanpy as sc

    umap_2d = None
    tsne_2d = None
    try:
        adata = sc.AnnData(latent.astype(np.float32))
        sc.pp.neighbors(adata, use_rep='X', n_neighbors=15)
        sc.tl.umap(adata)
        umap_2d = adata.obsm['X_umap']
        sc.tl.tsne(adata, use_rep='X')
        tsne_2d = adata.obsm['X_tsne']
    except Exception as e:
        print(f"  Warning: 2D projection failed: {e}")
    return umap_2d, tsne_2d


# ═══════════════════════════════════════════════════════════════════════════════
# Primary API
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(latent, labels, dre_k=15):
    """Compute the full metric battery.

    Parameters
    ----------
    latent : np.ndarray (n_cells, latent_dim)
    labels : array-like (n_cells,)
    dre_k : int
        Number of neighbors for DRE/DREX evaluations.

    Returns
    -------
    dict
        All metric values (NaN for any that fail).
    """
    latent = np.asarray(latent, dtype=float)
    labels = np.asarray(labels, dtype=int)
    metrics = {}

    # 1. Clustering
    metrics.update(_clustering_metrics(latent, labels))

    # 2. 2D projections
    umap_2d, tsne_2d = _compute_2d_projections(latent)

    # 3. DRE (UMAP)
    if umap_2d is not None:
        metrics.update(_dre_metrics(latent, umap_2d, dre_k, "DRE_umap"))
    else:
        for k in ('distance_correlation', 'Q_local', 'Q_global', 'overall_quality'):
            metrics[f'DRE_umap_{k}'] = np.nan

    # 4. DRE (tSNE)
    if tsne_2d is not None:
        metrics.update(_dre_metrics(latent, tsne_2d, dre_k, "DRE_tsne"))
    else:
        for k in ('distance_correlation', 'Q_local', 'Q_global', 'overall_quality'):
            metrics[f'DRE_tsne_{k}'] = np.nan

    # 5. LSE
    metrics.update(_lse_metrics(latent))

    # 6. DREX (using UMAP)
    if umap_2d is not None:
        metrics.update(_drex_metrics(latent, umap_2d, dre_k))
    else:
        for k in ('trustworthiness', 'continuity', 'distance_spearman',
                   'distance_pearson', 'local_scale_quality',
                   'neighborhood_symmetry', 'overall_quality'):
            metrics[f'DREX_{k}'] = np.nan

    # 7. LSEX
    metrics.update(_lsex_metrics(latent, dre_k))

    # 8. Latent diagnostics
    metrics.update(compute_latent_diagnostics(latent))

    # 9. Store projections for visualization
    metrics['_umap_2d'] = umap_2d
    metrics['_tsne_2d'] = tsne_2d

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Metric display metadata (for plotting)
# ═══════════════════════════════════════════════════════════════════════════════

# Core metrics panel
CORE_METRICS = [
    ("NMI",                        "NMI ↑",            True),
    ("ARI",                        "ARI ↑",            True),
    ("ASW",                        "ASW ↑",            True),
    ("DAV",                        "DAV ↓",            False),
    ("DRE_umap_overall_quality",   "DRE UMAP ↑",      True),
    ("LSE_overall_quality",        "LSE Overall ↑",    True),
]

# Extended metrics panels
EXT_METRICS_CLUSTERING = [
    ("COR",  "Corr ↑",     True),
    ("CAL",  "Cal-H ↑",    True),
]

EXT_METRICS_DRE = [
    ("DRE_umap_distance_correlation", "DRE UMAP DistCorr ↑", True),
    ("DRE_umap_Q_local",             "DRE UMAP Qloc ↑",     True),
    ("DRE_umap_Q_global",            "DRE UMAP Qglob ↑",    True),
    ("DRE_tsne_distance_correlation", "DRE tSNE DistCorr ↑", True),
    ("DRE_tsne_Q_local",             "DRE tSNE Qloc ↑",     True),
    ("DRE_tsne_Q_global",            "DRE tSNE Qglob ↑",    True),
    ("DRE_tsne_overall_quality",      "DRE tSNE Overall ↑",  True),
]

EXT_METRICS_LSE = [
    ("LSE_manifold_dimensionality", "LSE ManDim ↑",   True),
    ("LSE_spectral_decay_rate",     "LSE SpDecay ↑",  True),
    ("LSE_participation_ratio",     "LSE PartRat ↑",  True),
    ("LSE_anisotropy_score",        "LSE Aniso ↓",    False),
    ("LSE_noise_resilience",        "LSE NoiseR ↑",   True),
    ("LSE_core_quality",            "LSE Core ↑",     True),
]

EXT_METRICS_DREX = [
    ("DREX_trustworthiness",      "DREX Trust ↑",    True),
    ("DREX_continuity",           "DREX Cont ↑",     True),
    ("DREX_distance_spearman",    "DREX Spear ↑",    True),
    ("DREX_distance_pearson",     "DREX Pearson ↑",  True),
    ("DREX_local_scale_quality",  "DREX LocScale ↑", True),
    ("DREX_neighborhood_symmetry","DREX NbrSym ↑",   True),
    ("DREX_overall_quality",      "DREX Overall ↑",  True),
]

EXT_METRICS_LSEX = [
    ("LSEX_two_hop_connectivity",  "LSEX 2Hop ↑",     True),
    ("LSEX_radial_concentration",  "LSEX RadConc ↑",   True),
    ("LSEX_local_curvature",       "LSEX LocCurv ↑",   True),
    ("LSEX_entropy_stability",     "LSEX Entropy ↑",   True),
    ("LSEX_overall_quality",       "LSEX Overall ↑",   True),
]

ALL_METRIC_GROUPS = [
    ("Core Clustering", CORE_METRICS),
    ("Extended Clustering", EXT_METRICS_CLUSTERING),
    ("DR Quality (DRE)", EXT_METRICS_DRE),
    ("Latent Structure (LSE)", EXT_METRICS_LSE),
    ("Extended DR (DREX)", EXT_METRICS_DREX),
    ("Extended Latent (LSEX)", EXT_METRICS_LSEX),
]
