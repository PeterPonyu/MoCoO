"""
MoCoO Mixin Classes - Loss functions, metrics, ODE solver, vector field analysis.
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from scipy.sparse import issparse, csr_matrix
from typing import Optional, Tuple
from anndata import AnnData


class scviMixin:
    """Count-based likelihood functions for single-cell RNA-seq data."""
    
    def _normal_kl(self, mu1, lv1, mu2, lv2):
        """KL divergence between two diagonal Gaussians."""
        v1 = torch.exp(lv1)
        v2 = torch.exp(lv2)
        lstd1 = lv1 / 2.0
        lstd2 = lv2 / 2.0
        return lstd2 - lstd1 + (v1 + (mu1 - mu2) ** 2) / (2.0 * v2) - 0.5
    
    def _log_nb(self, x, mu, theta, eps=1e-8):
        """Negative Binomial log-likelihood."""
        log_theta_mu_eps = torch.log(theta + mu + eps)
        return (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )
    
    def _log_zinb(self, x, mu, theta, pi, eps=1e-8):
        """Zero-Inflated Negative Binomial log-likelihood."""
        pi = torch.sigmoid(pi)
        log_nb = self._log_nb(x, mu, theta, eps)
        case_zero = torch.log(pi + (1 - pi) * torch.exp(log_nb) + eps)
        case_nonzero = torch.log(1 - pi + eps) + log_nb
        return torch.where(x < eps, case_zero, case_nonzero)
    
    def _log_poisson(self, x, mu, eps=1e-8):
        """Poisson log-likelihood."""
        return x * torch.log(mu + eps) - mu - torch.lgamma(x + 1)
    
    def _log_zip(self, x, mu, pi, eps=1e-8):
        """Zero-Inflated Poisson log-likelihood."""
        pi = torch.sigmoid(pi)
        log_pois = self._log_poisson(x, mu, eps)
        case_zero = torch.log(pi + (1 - pi) * torch.exp(log_pois) + eps)
        case_nonzero = torch.log(1 - pi + eps) + log_pois
        return torch.where(x < eps, case_zero, case_nonzero)


class betatcMixin:
    """β-TC-VAE total correlation loss for disentanglement."""
    
    def _betatc_compute_gaussian_log_density(self, samples, mean, log_var):
        normalization = torch.log(torch.tensor(2 * np.pi, device=samples.device))
        inv_sigma = torch.exp(-log_var)
        tmp = samples - mean
        return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)
    
    def _betatc_compute_total_correlation(self, z_sampled, z_mean, z_logvar):
        log_qz_prob = self._betatc_compute_gaussian_log_density(
            z_sampled.unsqueeze(1),
            z_mean.unsqueeze(0),
            z_logvar.unsqueeze(0)
        )
        log_qz_product = log_qz_prob.exp().sum(dim=1).log().sum(dim=1)
        log_qz = log_qz_prob.sum(dim=2).exp().sum(dim=1).log()
        return (log_qz - log_qz_product).mean()


class infoMixin:
    """InfoVAE maximum mean discrepancy loss."""
    
    def _compute_mmd(self, z_posterior, z_prior):
        mean_pz_pz = self._compute_kernel_mean(
            self._compute_kernel(z_prior, z_prior), unbiased=True
        )
        mean_pz_qz = self._compute_kernel_mean(
            self._compute_kernel(z_prior, z_posterior), unbiased=False
        )
        mean_qz_qz = self._compute_kernel_mean(
            self._compute_kernel(z_posterior, z_posterior), unbiased=True
        )
        return mean_pz_pz - 2 * mean_pz_qz + mean_qz_qz
    
    def _compute_kernel_mean(self, kernel, unbiased):
        N = kernel.shape[0]
        if unbiased:
            sum_kernel = kernel.sum() - torch.diagonal(kernel).sum()
            return sum_kernel / (N * (N - 1))
        return kernel.mean()
    
    def _compute_kernel(self, z0, z1):
        batch_size, z_size = z0.shape
        z0 = z0.unsqueeze(1).expand(batch_size, batch_size, z_size)
        z1 = z1.unsqueeze(0).expand(batch_size, batch_size, z_size)
        sigma = 2 * z_size
        return torch.exp(-((z0 - z1).pow(2).sum(dim=-1) / sigma))


class dipMixin:
    """Disentangled Inferred Prior (DIP-VAE) loss."""
    
    def _dip_loss(self, q_m, q_s):
        cov_matrix = self._dip_cov_matrix(q_m, q_s)
        cov_diag = torch.diagonal(cov_matrix)
        cov_off_diag = cov_matrix - torch.diag(cov_diag)
        dip_loss_d = torch.sum((cov_diag - 1) ** 2)
        dip_loss_od = torch.sum(cov_off_diag ** 2)
        return 10 * dip_loss_d + 5 * dip_loss_od
    
    def _dip_cov_matrix(self, q_m, q_s):
        cov_q_mean = torch.cov(q_m.T)
        E_var = torch.mean(torch.exp(q_s), dim=0)
        return cov_q_mean + torch.diag(E_var)


class envMixin:
    """Environment mixin for clustering and evaluation metrics."""
    
    def _calc_score_with_labels(self, latent, labels):
        n_clusters = len(np.unique(labels))
        pred_labels = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=42
        ).fit_predict(latent)
        
        ari = adjusted_rand_score(labels, pred_labels)
        nmi = normalized_mutual_info_score(labels, pred_labels)
        asw = silhouette_score(latent, pred_labels)
        cal = calinski_harabasz_score(latent, pred_labels)
        dav = davies_bouldin_score(latent, pred_labels)
        cor = self._calc_corr(latent)
        
        return (ari, nmi, asw, cal, dav, cor)
    
    def _calc_corr(self, latent):
        acorr = np.abs(np.corrcoef(latent.T))
        return acorr.sum(axis=1).mean().item() - 1


class NODEMixin:
    """Neural ODE solver mixin."""
    
    @staticmethod
    def get_step_size(step_size: Optional[float], t0: float, t1: float, n_points: int) -> dict:
        if step_size is None:
            return {}
        elif step_size == "auto":
            return {"step_size": (t1 - t0) / (n_points - 1)}
        else:
            return {"step_size": float(step_size)}
    
    def solve_ode(
        self,
        ode_func: nn.Module,
        z0: torch.Tensor,
        t: torch.Tensor,
        method: str = "rk4",
        step_size: Optional[float] = None,
    ) -> torch.Tensor:
        device = z0.device
        cpu_z0 = z0.cpu()
        cpu_t = t.cpu()
        
        options = self.get_step_size(
            step_size,
            cpu_t[0].item(),
            cpu_t[-1].item(),
            len(cpu_t)
        )
        
        try:
            pred_z = odeint(
                ode_func.cpu(),
                cpu_z0,
                cpu_t,
                method=method,
                options=options
            )
        except Exception as e:
            import warnings
            warnings.warn(f"ODE solving failed: {e}, returning constant trajectory")
            pred_z = cpu_z0.unsqueeze(0).expand(len(cpu_t), -1)
        
        return pred_z.to(device)


class VectorFieldMixin:
    """Vector field analysis for ODE models."""
    
    def get_vfres(
        self,
        adata: AnnData,
        zs_key: str,
        E_key: str,
        vf_key: str = "X_vf",
        T_key: str = "cosine_similarity",
        dv_key: str = "X_dv",
        reverse: bool = False,
        scale: int = 10,
        self_transition: bool = False,
        smooth: float = 0.5,
        density: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.use_ode:
            raise ValueError("Vector field analysis requires use_ode=True")
        
        grads = self.take_grad(self.X)
        adata.obsm[vf_key] = grads
        
        adata.obsp[T_key] = self.get_similarity(adata, zs_key=zs_key, vf_key=vf_key, reverse=reverse)
        adata.obsm[dv_key] = self.get_vf(adata, T_key=T_key, E_key=E_key, scale=scale, self_transition=self_transition)
        
        E = np.asarray(adata.obsm[E_key])
        V = np.asarray(adata.obsm[dv_key])
        return self.get_vfgrid(E=E, V=V, smooth=smooth, density=density)
    
    def get_similarity(
        self,
        adata: AnnData,
        zs_key: str,
        vf_key: str = "X_vf",
        reverse: bool = False,
    ) -> csr_matrix:
        V = np.array(adata.obsm[vf_key])
        if reverse:
            V = -V
        
        ncells = adata.n_obs
        norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        sim = (V @ V.T) / (norms @ norms.T)
        np.fill_diagonal(sim, 0.0)
        
        rows, cols = np.nonzero(sim)
        vals = sim[rows, cols]
        return csr_matrix((vals, (rows, cols)), shape=(ncells, ncells))
    
    def get_vf(
        self,
        adata: AnnData,
        T_key: str,
        E_key: str,
        scale: int = 10,
        self_transition: bool = False,
    ) -> np.ndarray:
        T = adata.obsp[T_key]
        E = adata.obsm[E_key]
        
        if not self_transition:
            T = T - np.diag(T.diagonal()) if hasattr(T, 'diagonal') else T
        
        if issparse(T):
            T = T.toarray()
        
        row_sums = T.sum(axis=1)
        row_sums[row_sums == 0] = 1
        T = T / row_sums[:, None]
        
        V = T @ E - E
        return V * scale
    
    def get_vfgrid(
        self,
        E: np.ndarray,
        V: np.ndarray,
        smooth: float = 0.5,
        density: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        from scipy.interpolate import griddata
        
        n_grid = int(50 * density)
        grs = []
        for dim_i in range(2):
            m, M = E[:, dim_i].min(), E[:, dim_i].max()
            m -= 0.01 * np.abs(M - m)
            M += 0.01 * np.abs(M - m)
            grs.append(np.linspace(m, M, n_grid))
        
        meshes = np.meshgrid(*grs)
        E_grid = np.vstack([i.flat for i in meshes]).T
        
        V_grid = np.zeros_like(E_grid)
        for i in range(2):
            V_grid[:, i] = griddata(E, V[:, i], E_grid, method='linear', fill_value=0)
        
        if smooth > 0:
            from scipy.ndimage import gaussian_filter
            for i in range(2):
                V_grid_reshaped = V_grid[:, i].reshape(n_grid, n_grid)
                V_grid[:, i] = gaussian_filter(V_grid_reshaped, sigma=smooth).flatten()
        
        return E_grid, V_grid


def quiver_autoscale(E: np.ndarray, V: np.ndarray) -> float:
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots()
    scale_factor = np.abs(E).max()
    if scale_factor == 0:
        scale_factor = 1.0
    
    Q = ax.quiver(
        E[:, 0] / scale_factor,
        E[:, 1] / scale_factor,
        V[:, 0],
        V[:, 1],
        angles="xy",
        scale=None,
        scale_units="xy",
    )
    
    try:
        fig.canvas.draw()
        quiver_scale = Q.scale if Q.scale is not None else 1.0
    except Exception:
        quiver_scale = 1.0
    finally:
        plt.close(fig)
    
    return quiver_scale / scale_factor


def l2_norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    if issparse(x):
        return np.sqrt(x.multiply(x).sum(axis=axis).A1)
    else:
        return np.sqrt(np.sum(x * x, axis=axis))