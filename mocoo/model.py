"""MoCoO model - training and loss computation."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from .mixin import scviMixin, dipMixin, betatcMixin, infoMixin
from .module import VAE


class MoCoOModel(scviMixin, dipMixin, betatcMixin, infoMixin):
    """MoCoO model with VAE, ODE, and MoCo."""
    
    def __init__(
        self,
        recon, irecon, beta, dip, tc, info,
        state_dim, hidden_dim, latent_dim, i_dim,
        use_ode, use_moco, loss_mode, lr,
        vae_reg, ode_reg, moco_weight, use_qm, moco_T, moco_K,
        use_prototype, n_prototypes,
        proto_weight,
        device, grad_clip=1.0,
        *args, **kwargs,
    ):
        self.use_ode = use_ode
        self.use_moco = use_moco
        self.use_qm = use_qm
        self.loss_mode = loss_mode
        self.recon = recon
        self.irecon = irecon
        self.beta = beta
        self.dip = dip
        self.tc = tc
        self.info = info
        self.moco_weight = moco_weight
        self.proto_weight = proto_weight
        self.grad_clip = grad_clip
        self.vae_reg = vae_reg
        self.ode_reg = ode_reg
        self.device = device
        self.loss = []
        
        self.nn = VAE(
            state_dim, hidden_dim, latent_dim, i_dim,
            use_ode, use_moco, loss_mode, moco_T, moco_K,
            use_prototype, n_prototypes, device,
        )
        
        self.nn_optimizer = optim.Adam(self.nn.parameters(), lr=lr)
        self.moco_criterion = nn.CrossEntropyLoss()
    
    def _compute_recon_loss(self, x_raw, pred_x, dropout_logits=None):
        if self.loss_mode == "mse":
            return F.mse_loss(x_raw, pred_x, reduction="none").sum(-1).mean()
        
        # Clamp lib_size for numerical stability (following HSDE)
        l = torch.clamp(x_raw.sum(-1, keepdim=True), min=1.0)
        pred_x_scaled = pred_x * l + 1e-8
        disp = torch.exp(self.nn.decoder.disp)
        
        if self.loss_mode == "nb":
            return -self._log_nb(x_raw, pred_x_scaled, disp).sum(-1).mean()
        elif self.loss_mode == "zinb":
            return -self._log_zinb(x_raw, pred_x_scaled, disp, dropout_logits).sum(-1).mean()
        elif self.loss_mode == "poisson":
            return -self._log_poisson(x_raw, pred_x_scaled).sum(-1).mean()
        elif self.loss_mode == "zip":
            return -self._log_zip(x_raw, pred_x_scaled, dropout_logits).sum(-1).mean()
        else:
            raise ValueError(f"Unknown loss_mode: {self.loss_mode}")

    @staticmethod
    def _velocity_consistency_loss(q_z, velocity):
        if q_z.shape[0] < 3:
            return torch.tensor(0.0, device=q_z.device)
        
        displacement = q_z[1:] - q_z[:-1]
        velocity_mid = velocity[:-1]
        
        disp_norm = F.normalize(displacement, dim=-1, eps=1e-8)
        vel_norm = F.normalize(velocity_mid, dim=-1, eps=1e-8)
        
        cosine_sim = (disp_norm * vel_norm).sum(dim=-1)
        return (1.0 - cosine_sim).mean()
    
    @torch.no_grad()
    def take_latent(self, state, use_qm=None):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        effective_qm = use_qm if use_qm is not None else self.use_qm

        if self.use_ode:
            q_z, q_m, q_s, t = self.nn.encoder(state)

            # Sort by time and add jitter for uniqueness (match training)
            sort_idx = torch.argsort(t)
            t_sorted = t[sort_idx]
            z_base = (q_m if effective_qm else q_z)[sort_idx]
            
            # Ensure strictly increasing times
            eps = 1e-6
            t_unique = t_sorted.clone()
            for i in range(1, len(t_unique)):
                if t_unique[i] <= t_unique[i-1] + eps:
                    t_unique[i] = t_unique[i-1] + eps
            
            z0 = z_base[0]
            q_z_ode = self.nn.solve_ode(self.nn.ode_solver, z0, t_unique)
            
            # Blend and unsort back to original order
            blended = self.vae_reg * z_base + self.ode_reg * q_z_ode
            unsort_idx = torch.argsort(sort_idx)
            return blended[unsort_idx].cpu().numpy()
        else:
            q_z, q_m, q_s = self.nn.encoder(state)
            return (q_m if effective_qm else q_z).cpu().numpy()
    
    @torch.no_grad()
    def take_bottleneck(self, state):
        states = torch.tensor(state, dtype=torch.float).to(self.device)
        
        if self.use_ode:
            q_z, q_m, q_s, t = self.nn.encoder(states)
        else:
            q_z, q_m, q_s = self.nn.encoder(states)
        
        le = self.nn.latent_encoder(q_z)
        return le.cpu().numpy()
    
    @torch.no_grad()
    def take_time(self, state):
        states = torch.tensor(state, dtype=torch.float).to(self.device)
        _, _, _, t = self.nn.encoder(states)
        return t.cpu().numpy()
    
    @torch.no_grad()
    def take_grad(self, state):
        states = torch.tensor(state, dtype=torch.float).to(self.device)
        q_z, q_m, q_s, t = self.nn.encoder(states)
        grads = self.nn.ode_solver(t, q_z).cpu().numpy()
        return grads
    
    @torch.no_grad()
    def take_transition(self, state, top_k: int = 30):
        states = torch.tensor(state, dtype=torch.float).to(self.device)
        q_z, q_m, q_s, t = self.nn.encoder(states)
        
        grads = self.nn.ode_solver(t, q_z).cpu().numpy()
        z_latent = q_z.cpu().numpy()
        z_future = z_latent + 1e-2 * grads
        
        distances = pairwise_distances(z_latent, z_future)
        sigma = np.median(distances) + 1e-8
        similarity = np.exp(-(distances**2) / (2 * sigma**2))
        transition_matrix = similarity / (similarity.sum(axis=1, keepdims=True) + 1e-8)
        
        n_cells = transition_matrix.shape[0]
        sparse_trans = np.zeros_like(transition_matrix)
        for i in range(n_cells):
            top_indices = np.argsort(transition_matrix[i])[::-1][:top_k]
            sparse_trans[i, top_indices] = transition_matrix[i, top_indices]
            sparse_trans[i] /= sparse_trans[i].sum() + 1e-8
        
        return sparse_trans
    
    def update(self, x_norm, x_raw=None, x_q_norm=None, x_k_norm=None):
        if x_raw is None:
            x_raw = x_norm
        
        x_norm_t = torch.tensor(x_norm, dtype=torch.float).to(self.device)
        x_raw_t = torch.tensor(x_raw, dtype=torch.float).to(self.device)
        
        if self.use_moco and x_q_norm is not None and x_k_norm is not None:
            x_q = torch.tensor(x_q_norm, dtype=torch.float).to(self.device)
            x_k = torch.tensor(x_k_norm, dtype=torch.float).to(self.device)
            out = self.nn(x_norm_t, x_q, x_k)
        else:
            out = self.nn(x_norm_t)
        
        q_z, q_m, q_s = out['q_z'], out['q_m'], out['q_s']
        
        if self.use_ode:
            # Reuse sort indices from forward pass (no re-encoding needed)
            sort_idx = out['sort_idx']
            n_sorted = len(out['x_sorted'])
            x_raw_sorted = x_raw_t[sort_idx][:n_sorted]
            
            # Dual-path reconstruction:
            # VAE path gets full weight, ODE path scaled by ode_reg
            recon_loss_ec = self._compute_recon_loss(
                x_raw_sorted, out['pred_x'], out.get('dropout_x'))
            recon_loss_ode = self._compute_recon_loss(
                x_raw_sorted, out['pred_x_ode'], out.get('dropout_x_ode'))
            recon_loss = recon_loss_ec + self.ode_reg * recon_loss_ode
            
            # Unidirectional z_div: ODE learns to match encoder, not vice versa
            # stop-gradient on q_z prevents ODE trajectory from distorting encoder clusters
            qz_div = self.ode_reg * F.mse_loss(q_z.detach(), out['q_z_ode'], reduction="none").sum(-1).mean()
            vel_loss = self._velocity_consistency_loss(q_z.detach(), out['velocity'])
        else:
            recon_loss = self._compute_recon_loss(x_raw_t, out['pred_x'], out.get('dropout_x'))
            qz_div = torch.tensor(0.0, device=self.device)
            vel_loss = torch.tensor(0.0, device=self.device)
        
        if self.irecon:
            x_target = x_raw_sorted if self.use_ode else x_raw_t
            irecon_loss = self.irecon * self._compute_recon_loss(
                x_target, out['pred_xl'], out.get('dropout_xl')
            )
        else:
            irecon_loss = torch.tensor(0.0, device=self.device)
        
        moco_loss = torch.tensor(0.0, device=self.device)
        cross_loss = torch.tensor(0.0, device=self.device)
        proto_loss = torch.tensor(0.0, device=self.device)
        
        if 'moco_logits' in out:
            moco_loss = self.moco_criterion(out['moco_logits'], out['moco_labels'])
        
        if 'cross_path_loss' in out:
            cross_loss = out['cross_path_loss']
        
        if (self.use_moco and self.proto_weight > 0
                and hasattr(self.nn, 'moco') and self.nn.moco.use_prototype):
            # Always use pure VAE latent for proto loss (never blend with ODE)
            # This keeps proto consistent with VAE+MoCo+Proto behavior
            proto_loss = self.nn.moco.prototype_contrastive_loss(out['q_z'])
        
        p_m = torch.zeros_like(q_m)
        p_s = torch.zeros_like(q_s)
        kl_div = self.beta * self._normal_kl(q_m, q_s, p_m, p_s).sum(-1).mean()
        
        dip_loss = self.dip * self._dip_loss(q_m, q_s) if self.dip else torch.tensor(0.0, device=self.device)
        tc_loss = self.tc * self._betatc_compute_total_correlation(q_z, q_m, q_s) if self.tc else torch.tensor(0.0, device=self.device)
        mmd_loss = self.info * self._compute_mmd(q_z, torch.randn_like(q_z)) if self.info else torch.tensor(0.0, device=self.device)
        
        total_loss = (
            self.recon * recon_loss +
            irecon_loss +
            qz_div +
            self.ode_reg * 0.1 * vel_loss +
            kl_div +
            dip_loss +
            tc_loss +
            mmd_loss +
            self.moco_weight * (moco_loss + cross_loss) +
            self.proto_weight * proto_loss
        )
        
        self.nn_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.nn.parameters(), self.grad_clip)
        self.nn_optimizer.step()
        
        self.loss.append((
            total_loss.item(),
            recon_loss.item(),
            irecon_loss.item() if isinstance(irecon_loss, torch.Tensor) else irecon_loss,
            kl_div.item(),
            dip_loss.item() if isinstance(dip_loss, torch.Tensor) else dip_loss,
            tc_loss.item() if isinstance(tc_loss, torch.Tensor) else tc_loss,
            mmd_loss.item() if isinstance(mmd_loss, torch.Tensor) else mmd_loss,
            moco_loss.item() if isinstance(moco_loss, torch.Tensor) else moco_loss,
            cross_loss.item() if isinstance(cross_loss, torch.Tensor) else cross_loss,
            vel_loss.item() if isinstance(vel_loss, torch.Tensor) else vel_loss,
            proto_loss.item() if isinstance(proto_loss, torch.Tensor) else proto_loss,
        ))
