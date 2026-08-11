"""MoCoO agent - top-level API."""

import random
from .environment import Env
from .mixin import VectorFieldMixin
import tqdm
import time
import torch
import numpy as np
from anndata import AnnData
from typing import Optional, Dict, List
from scipy.stats import spearmanr, pearsonr


class MoCoO(Env, VectorFieldMixin):
    """
    MoCoO: Momentum Contrast ODE-Regularized VAE
    
    Unified framework combining VAE, Neural ODE, and MoCo for single-cell analysis.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix
    layer : str, default='counts'
        Layer containing raw counts
    recon, beta, dip, tc, info : float
        Loss weights
    hidden_dim : int, default=128
        Hidden layer size
    latent_dim : int, default=10
        Latent space dimension
    i_dim : int, default=2
        Bottleneck dimension
    use_ode : bool, default=False
        Enable Neural ODE
    use_moco : bool, default=False
        Enable MoCo
    loss_mode : str, default='nb'
        'mse', 'nb', 'zinb', 'poisson', 'zip'
    lr : float, default=1e-4
        Learning rate
    vae_reg, ode_reg : float
        ODE path weights (must sum to 1.0)
    moco_weight : float, default=1.0
        MoCo loss weight
    moco_T : float, default=0.2
        MoCo temperature
    moco_K : int, default=4096
        MoCo queue size
    use_prototype : bool, default=False
        Enable optional PCL-/prototype-inspired contrastive
    n_prototypes : int, default=10
        Number of learnable prototypes
    aug_prob, mask_prob, noise_prob : float
        Augmentation parameters
    use_qm : bool, default=False
        Use mean instead of sampled latent
    grad_clip : float, default=1.0
        Gradient clipping
    train_size, val_size, test_size : float
        Split proportions (must sum to 1.0)
    batch_size : int, default=128
        Mini-batch size
    random_seed : int, default=42
        Random seed
    device : torch.device, optional
        Computation device
    """
    
    def __init__(
        self,
        adata: AnnData,
        layer: str = 'counts',
        recon: float = 1.0,
        beta: float = 1.0,
        dip: float = 0.0,
        tc: float = 0.0,
        info: float = 0.0,
        hidden_dim: int = 128,
        latent_dim: int = 10,
        i_dim: int = 2,
        use_ode: bool = False,
        use_moco: bool = False,
        loss_mode: str = 'nb',
        lr: float = 1e-4,
        vae_reg: float = 0.5,
        ode_reg: float = 0.5,
        moco_weight: float = 1.0,
        moco_T: float = 0.2,
        moco_K: int = 4096,
        use_prototype: bool = False,
        n_prototypes: int = 12,
        proto_weight: float = 1.0,
        aug_prob: float = 0.5,
        mask_prob: float = 0.1,
        noise_prob: float = 0.1,
        use_qm: bool = False,
        grad_clip: float = 1.0,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        batch_size: int = 128,
        random_seed: int = 42,
        device: Optional[torch.device] = None,
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not (0.99 <= train_size + val_size + test_size <= 1.01):
            raise ValueError(f"Splits must sum to 1.0, got {train_size + val_size + test_size}")
        
        if use_ode and not (0.99 <= vae_reg + ode_reg <= 1.01):
            raise ValueError(f"ODE weights must sum to 1.0, got {vae_reg + ode_reg}")
        
        if i_dim >= latent_dim:
            raise ValueError(f"i_dim ({i_dim}) must be < latent_dim ({latent_dim})")
        
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        
        super().__init__(
            adata=adata,
            layer=layer,
            recon=recon,
            beta=beta,
            dip=dip,
            tc=tc,
            info=info,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            i_dim=i_dim,
            use_ode=use_ode,
            use_moco=use_moco,
            loss_mode=loss_mode,
            lr=lr,
            vae_reg=vae_reg,
            ode_reg=ode_reg,
            moco_weight=moco_weight,
            moco_T=moco_T,
            moco_K=moco_K,
            use_prototype=use_prototype,
            n_prototypes=n_prototypes,
            proto_weight=proto_weight,
            aug_prob=aug_prob,
            mask_prob=mask_prob,
            noise_prob=noise_prob,
            use_qm=use_qm,
            device=device,
            grad_clip=grad_clip,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            batch_size=batch_size,
            random_seed=random_seed,
        )
        
        self.train_time = 0.0
        self.peak_memory_gb = 0.0
        self.actual_epochs = 0
        
        print(f"\n{'='*70}")
        print(f"MoCoO initialized on {device}")
        print(f"  ODE: {use_ode} | MoCo: {use_moco} | Loss: {loss_mode}")
        print(f"  Architecture: {self.n_var} → {hidden_dim} → {latent_dim} → {i_dim}")
        print(f"  Batch size: {batch_size} | MoCo queue: {moco_K if use_moco else 'N/A'}")
        print(f"{'='*70}\n")
    
    def fit(
        self,
        epochs: int = 400,
        patience: int = 25,
        val_every: int = 5,
        track_metrics: bool = True,
    ) -> 'MoCoO':
        """
        Train with early stopping.
        
        Parameters
        ----------
        epochs : int
            Maximum epochs
        patience : int
            Early stopping patience
        val_every : int
            Validation frequency
        track_metrics : bool, default=True
            If True, compute intermediate clustering metrics (ARI, NMI, etc.)
            during validation. Set to False for faster training when only
            the loss-based early stopping is needed.
        """
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        # Build config tag for the progress bar description
        tag_parts = ['VAE']
        if self.use_ode:
            tag_parts.append('ODE')
        if self.use_moco:
            tag_parts.append('MoCo')
        if self.proto_weight > 0:
            tag_parts.append('Proto')
        config_tag = '+'.join(tag_parts)
        
        bar_fmt = (
            '{l_bar}{bar:30}{r_bar}'
        )
        
        with tqdm.tqdm(
            total=epochs,
            desc=f"  {config_tag}",
            bar_format=bar_fmt,
            dynamic_ncols=True,
        ) as pbar:
            for epoch in range(epochs):
                train_loss = self.train_epoch()
                
                if (epoch + 1) % val_every == 0 or epoch == 0:
                    val_loss, val_score = self.validate(
                        track_metrics=track_metrics
                    )
                    
                    should_stop, improved = self.check_early_stopping(val_loss, patience)
                    
                    # Build concise postfix
                    postfix = {
                        'trn': f'{train_loss:.1f}',
                        'val': f'{val_loss:.1f}',
                    }
                    if val_score is not None:
                        postfix.update({
                            'ARI': f'{val_score[0]:.3f}',
                            'NMI': f'{val_score[1]:.3f}',
                            'ASW': f'{val_score[2]:.3f}',
                        })
                    postfix['best'] = f'{self.best_val_loss:.1f}'
                    postfix['pat'] = f'{self.patience_counter}/{patience}'
                    if improved:
                        postfix['↑'] = '✓'
                    
                    pbar.set_postfix(postfix)
                    
                    if should_stop:
                        self.actual_epochs = epoch + 1
                        pbar.write(
                            f"  ↳ Early stop @ epoch {epoch + 1}, "
                            f"best val={self.best_val_loss:.2f}"
                        )
                        self.load_best_model()
                        break
                
                pbar.update(1)
            else:
                self.actual_epochs = epochs
        
        self.train_time = time.time() - start_time
        self.peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0.0
        
        return self

    # ------------------------------------------------------------------ #
    #  Model persistence                                                  #
    # ------------------------------------------------------------------ #

    def save_model(self, path: str) -> None:
        """Save model weights and config to disk.

        Parameters
        ----------
        path : str
            File path for the checkpoint (e.g. ``"model.pt"``).
        """
        checkpoint = {
            'state_dict': self.nn.state_dict(),
            'config': {
                'use_ode': self.use_ode,
                'use_moco': self.use_moco,
                'loss_mode': self.loss_mode,
                'vae_reg': self.vae_reg,
                'ode_reg': self.ode_reg,
            },
        }
        if hasattr(self, 'fm_net'):
            checkpoint['fm_state_dict'] = self.fm_net.state_dict()
            checkpoint['config']['fm_hidden_dim'] = self.fm_net.net[0].out_features
        torch.save(checkpoint, path)

    def load_model(self, path: str) -> None:
        """Load model weights from a checkpoint.

        Parameters
        ----------
        path : str
            File path to the checkpoint saved by :meth:`save_model`.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.nn.load_state_dict(checkpoint['state_dict'])
        self.best_model_state = {
            k: v.cpu().clone() for k, v in self.nn.state_dict().items()
        }
        if 'fm_state_dict' in checkpoint:
            if not hasattr(self, 'fm_net'):
                hd = checkpoint.get('config', {}).get('fm_hidden_dim', 128)
                self.fm_init(hidden_dim=hd)
                # Unfreeze VAE so user can resume Phase-1 if desired
                for p in self.nn.parameters():
                    p.requires_grad_(True)
            self.fm_net.load_state_dict(checkpoint['fm_state_dict'])

    def get_latent(self) -> np.ndarray:
        """Extract latent representations."""
        return self.take_latent(self.X)

    def get_latent_qm(self) -> np.ndarray:
        """Extract latent representations using deterministic posterior mean."""
        return self.take_latent(self.X, use_qm=True)

    def get_bottleneck(self) -> np.ndarray:
        """Extract bottleneck representations."""
        return self.take_bottleneck(self.X)
    
    def get_test_latent(self) -> np.ndarray:
        """Extract test latent."""
        return self.take_latent(self.X_test)
    
    def get_time(self) -> np.ndarray:
        """Extract pseudotime (ODE only)."""
        if not self.use_ode:
            raise RuntimeError("get_time() requires use_ode=True")
        return self.take_time(self.X)
    
    def get_pseudotime(self, adata: Optional[AnnData] = None) -> np.ndarray:
        """Extract ODE-derived pseudotime, optionally storing in adata.obs.

        Parameters
        ----------
        adata : AnnData, optional
            If provided, stores pseudotime in ``adata.obs['mocoo_pseudotime']``.

        Returns
        -------
        pseudotime : np.ndarray, shape (n_cells,)
            Pseudotime values in [0, 1] for all cells.
        """
        if not self.use_ode:
            raise RuntimeError("get_pseudotime() requires use_ode=True")
        pt = self.take_time(self.X)
        if adata is not None:
            adata.obs['mocoo_pseudotime'] = pt
        return pt

    def get_latent_smoothness(self) -> Dict[str, float]:
        """Compute latent space smoothness metrics.

        Returns
        -------
        metrics : dict
            - knn_entropy: mean k-NN graph entropy (higher = smoother)
            - pairwise_dist_mean / std: pairwise Euclidean distance statistics
            - effective_dim: PCA participation ratio (effective dimensionality)
        """
        from sklearn.neighbors import NearestNeighbors

        latent = self.get_latent()
        n = latent.shape[0]

        # k-NN entropy (k=10)
        k = min(10, n - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(latent)
        dists, indices = nn.kneighbors(latent)
        # Compute entropy of neighbourhood label distribution
        entropies = []
        for i in range(n):
            nbr_labels = self.labels[indices[i, 1:]]
            counts = np.bincount(nbr_labels, minlength=len(np.unique(self.labels)))
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            entropies.append(-np.sum(probs * np.log(probs + 1e-12)))
        knn_entropy = float(np.mean(entropies))

        # Pairwise distances (subsample for speed)
        from sklearn.metrics import pairwise_distances
        sub = min(500, n)
        idx = np.random.choice(n, sub, replace=False)
        pdist = pairwise_distances(latent[idx])
        upper = pdist[np.triu_indices(sub, k=1)]

        # PCA participation ratio
        centered = latent - latent.mean(axis=0)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        eigenvalues = (s ** 2) / (n - 1)
        eigenvalues = eigenvalues / eigenvalues.sum()
        participation_ratio = float(1.0 / np.sum(eigenvalues ** 2))

        return {
            'knn_entropy': knn_entropy,
            'pairwise_dist_mean': float(upper.mean()),
            'pairwise_dist_std': float(upper.std()),
            'effective_dim': participation_ratio,
        }

    def pseudotime_marker_correlation(
        self,
        adata: AnnData,
        marker_genes: Optional[List[str]] = None,
        top_n: int = 20,
        layer: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compute correlation between ODE pseudotime and gene expression.

        Parameters
        ----------
        adata : AnnData
            Must contain the genes of interest.
        marker_genes : list of str, optional
            Specific genes to test. If None, returns top_n most correlated.
        top_n : int
            Number of top correlated genes to return when marker_genes is None.
        layer : str, optional
            AnnData layer to use for expression. None = adata.X.

        Returns
        -------
        correlations : dict
            ``{gene: {'spearman_r': float, 'spearman_p': float,
                       'pearson_r': float, 'pearson_p': float}}``
        """
        if not self.use_ode:
            raise RuntimeError("pseudotime_marker_correlation() requires use_ode=True")

        from scipy.sparse import issparse as _issparse

        pt = self.take_time(self.X)

        # Get expression matrix
        if layer and layer in adata.layers:
            X = adata.layers[layer]
        else:
            X = adata.X
        if _issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        gene_names = list(adata.var_names)

        if marker_genes is not None:
            # Filter to genes present in adata
            genes_to_test = [g for g in marker_genes if g in gene_names]
            if not genes_to_test:
                raise ValueError(f"None of the marker genes found in adata.var_names")
        else:
            # Find top_n most correlated genes
            all_corrs = []
            for j in range(X.shape[1]):
                expr = X[:, j]
                if expr.std() < 1e-8:
                    all_corrs.append(0.0)
                else:
                    r, _ = spearmanr(pt, expr)
                    all_corrs.append(abs(r) if not np.isnan(r) else 0.0)
            top_idx = np.argsort(all_corrs)[-top_n:][::-1]
            genes_to_test = [gene_names[i] for i in top_idx]

        results = {}
        for gene in genes_to_test:
            j = gene_names.index(gene)
            expr = X[:, j]
            if expr.std() < 1e-8:
                results[gene] = {
                    'spearman_r': 0.0, 'spearman_p': 1.0,
                    'pearson_r': 0.0, 'pearson_p': 1.0,
                }
                continue
            sr, sp = spearmanr(pt, expr)
            pr, pp = pearsonr(pt, expr)
            results[gene] = {
                'spearman_r': float(sr) if not np.isnan(sr) else 0.0,
                'spearman_p': float(sp) if not np.isnan(sp) else 1.0,
                'pearson_r': float(pr) if not np.isnan(pr) else 0.0,
                'pearson_p': float(pp) if not np.isnan(pp) else 1.0,
            }

        return results
    
    def get_velocity(self) -> np.ndarray:
        """Extract velocity vectors (ODE only)."""
        if not self.use_ode:
            raise RuntimeError("get_velocity() requires use_ode=True")
        return self.take_grad(self.X)
    
    def get_transition(self, top_k: int = 30) -> np.ndarray:
        """Extract transition matrix (ODE only)."""
        if not self.use_ode:
            raise RuntimeError("get_transition() requires use_ode=True")
        return self.take_transition(self.X, top_k=top_k)
    
    def get_resource_metrics(self) -> Dict[str, float]:
        """Get training resource metrics."""
        return {
            'train_time': self.train_time,
            'peak_memory_gb': self.peak_memory_gb,
            'actual_epochs': self.actual_epochs,
        }
    
    def get_loss_history(self) -> Dict[str, np.ndarray]:
        """Get loss history."""
        if len(self.loss) == 0:
            return {}
        
        loss_array = np.array(self.loss)
        return {
            'total': loss_array[:, 0],
            'train': np.array(self.train_losses),
            'val': np.array(self.val_losses),
        }
    
    def get_metrics_history(self) -> Dict[str, np.ndarray]:
        """Get validation metrics history."""
        if len(self.val_scores) == 0:
            return {}
        
        score_array = np.array(self.val_scores)
        return {
            'ARI': score_array[:, 0],
            'NMI': score_array[:, 1],
            'ASW': score_array[:, 2],
            'CH': score_array[:, 3],
            'DB': score_array[:, 4],
            'Corr': score_array[:, 5],
        }

    # ------------------------------------------------------------------ #
    #  Phase-2: Latent Flow Matching                                      #
    # ------------------------------------------------------------------ #

    def train_fm(
        self,
        epochs: int = 200,
        lr: float = 1e-3,
        hidden_dim: int = 128,
        batch_size: Optional[int] = None,
    ) -> 'MoCoO':
        """Phase-2: train a conditional flow-matching velocity network on
        the frozen VAE encoder's posterior means.

        Parameters
        ----------
        epochs : int
            FM training epochs.
        lr : float
            Learning rate for the velocity network.
        hidden_dim : int
            Hidden width of the velocity MLP.
        batch_size : int, optional
            Mini-batch size (defaults to Phase-1 batch_size).
        """
        if batch_size is None:
            batch_size = self.batch_size

        # Initialise velocity net (freezes VAE automatically)
        self.fm_init(lr=lr, hidden_dim=hidden_dim)

        # Pre-compute all posterior means once (frozen encoder)
        z_all = self._encode_targets(self.X_train_norm)
        n = z_all.shape[0]

        bar_fmt = '{l_bar}{bar:30}{r_bar}'
        with tqdm.tqdm(total=epochs, desc='  FM', bar_format=bar_fmt,
                       dynamic_ncols=True) as pbar:
            for epoch in range(epochs):
                perm = torch.randperm(n, device=self.device)
                epoch_loss = 0.0
                n_batches = 0
                for start in range(0, n, batch_size):
                    idx = perm[start:start + batch_size]
                    loss = self.fm_update(z_all[idx])
                    epoch_loss += loss
                    n_batches += 1
                avg = epoch_loss / max(n_batches, 1)
                pbar.set_postfix(loss=f'{avg:.4f}')
                pbar.update(1)

        # Re-enable VAE gradients (in case user resumes Phase-1 later)
        for p in self.nn.parameters():
            p.requires_grad_(True)

        return self

    def get_fm_latent(self, t_start: float = 0.9, steps: int = 100) -> np.ndarray:
        """Get FM-refined latent for all cells.

        Parameters
        ----------
        t_start : float
            Flow starting time (0.9 = light smoothing, 0.5 = heavy denoising).
        steps : int
            Euler integration steps.
        """
        return self.take_fm_refined(self.X, t_start=t_start, steps=steps)

    def get_fm_sample(self, n: int = 100, steps: int = 100) -> np.ndarray:
        """Generate *n* new cell latents from pure noise via FM."""
        return self.take_fm_sample(n, steps=steps)

    def get_fm_loss_history(self) -> np.ndarray:
        """Return per-step FM training loss."""
        if not hasattr(self, 'fm_loss_history'):
            return np.array([])
        return np.array(self.fm_loss_history)

    # ------------------------------------------------------------------ #
    #  Downstream analysis API                                            #
    # ------------------------------------------------------------------ #

    def get_gene_jacobian(self, batch_size: int = 128) -> np.ndarray:
        """Decoder Jacobian dmu/dz per cell.

        Returns
        -------
        np.ndarray, shape (N, G, D)
            J[i, g, d] = partial mu_g / partial z_d at cell i.
        """
        return self.take_gene_jacobian(self.X, batch_size=batch_size)

    def get_gene_velocity(self, batch_size: int = 128) -> np.ndarray:
        """Gene-space velocity: v_gene = (dmu/dz) * (dz/dt).

        Returns
        -------
        np.ndarray, shape (N, G)
        """
        if not self.use_ode:
            raise RuntimeError("get_gene_velocity() requires use_ode=True")
        return self.take_gene_velocity(self.X, batch_size=batch_size)

    def get_decoded(self, z: Optional[np.ndarray] = None) -> np.ndarray:
        """Decode latent z to gene-space proportions.

        Parameters
        ----------
        z : np.ndarray (N, D), optional
            If None, decodes the standard VAE latent of all cells.

        Returns
        -------
        np.ndarray, shape (N, G)
        """
        if z is None:
            z = self.get_latent_qm()
        return self.take_decoded(z)

    def get_divergence(self, batch_size: int = 128) -> np.ndarray:
        """ODE velocity divergence trace(df/dz) per cell.

        Returns
        -------
        np.ndarray, shape (N,)
        """
        if not self.use_ode:
            raise RuntimeError("get_divergence() requires use_ode=True")
        return self.take_divergence(self.X, batch_size=batch_size)

    def generate_cells(self, n: int = 100, steps: int = 100,
                       decode: bool = True) -> np.ndarray:
        """Generate *n* new cells via FM, optionally decoded to gene space.

        Parameters
        ----------
        n : int
        steps : int
            FM Euler integration steps.
        decode : bool
            If True, return gene-space proportions (N, G).
            If False, return latent (N, D).

        Returns
        -------
        np.ndarray
        """
        if not hasattr(self, 'fm_net'):
            raise RuntimeError("Call train_fm() before generate_cells()")
        z = self.take_fm_sample(n, steps)
        if not decode:
            return z
        return self.take_decoded(z)

    def get_uncertainty(self, n_samples: int = 50) -> Dict[str, np.ndarray]:
        """Posterior sampling uncertainty per cell.

        Returns
        -------
        dict
            - ``latent_std``: (N, D) per-dim std across samples.
            - ``uncertainty``: (N,) mean std.
        """
        from mocoo.evaluation.uncertainty import posterior_uncertainty
        return posterior_uncertainty(
            self.nn.encoder, self.X,
            n_samples=n_samples, device=self.device,
        )

    def annotate_cells(
        self,
        query_data: Optional[np.ndarray] = None,
        reference_data: Optional[np.ndarray] = None,
        reference_labels: Optional[np.ndarray] = None,
        method: str = 'prototype',
        k: int = 15,
    ) -> Dict:
        """Annotate cells via prototype assignment or kNN transfer.

        Parameters
        ----------
        query_data : np.ndarray (N_q, G), optional
            Log1p-normalised query expression. If None, uses self.X.
        reference_data : np.ndarray (N_r, G), optional
            Expression for kNN reference cells. If None, uses self.X.
        reference_labels : np.ndarray, optional
            Labels for reference cells (needed for kNN).
        method : {'prototype', 'knn'}
        k : int
            Number of neighbors for kNN.

        Returns
        -------
        dict with 'labels', 'confidence', and method-specific keys.
        """
        from mocoo.evaluation.annotation_transfer import (
            annotate_by_prototype, annotate_by_knn,
        )

        if method == 'prototype':
            if not (self.use_moco and hasattr(self.nn, 'moco')
                    and self.nn.moco.use_prototype):
                raise RuntimeError(
                    "Prototype annotation requires use_moco=True and "
                    "use_prototype=True"
                )
            latent = self.get_latent_qm() if query_data is None else self.take_latent(query_data, use_qm=True)
            # Project latent into prototype space (proj_head_q + L2-norm)
            # to match how prototype_contrastive_loss() uses them.
            with torch.no_grad():
                z_t = torch.tensor(latent, dtype=torch.float).to(self.device)
                h = self.nn.moco.proj_head_q(z_t)
                h = torch.nn.functional.normalize(h, dim=1)
                projected = h.cpu().numpy()
                protos = torch.nn.functional.normalize(
                    self.nn.moco.prototypes, dim=1,
                ).cpu().numpy()
            return annotate_by_prototype(projected, protos, reference_labels)

        elif method == 'knn':
            if reference_labels is None:
                raise ValueError("kNN annotation requires reference_labels")
            if reference_data is not None:
                ref_latent = self.take_latent(reference_data, use_qm=True)
            else:
                ref_latent = self.get_latent_qm()
            if query_data is not None:
                q_latent = self.take_latent(query_data, use_qm=True)
            else:
                q_latent = ref_latent
            return annotate_by_knn(q_latent, ref_latent, reference_labels, k=k)

        else:
            raise ValueError(f"Unknown method: {method!r}. Use 'prototype' or 'knn'.")