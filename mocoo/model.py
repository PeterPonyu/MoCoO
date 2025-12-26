"""
MoCoO Model Class - Training and Loss Computation.
"""

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
        self.grad_clip = grad_clip
        self.vae_reg = vae_reg
        self.ode_reg = ode_reg
        self.device = device
        self.loss = []
        
        self.nn = VAE(
            state_dim, hidden_dim, latent_dim, i_dim,
            use_ode, use_moco, loss_mode, moco_T, moco_K, device,
        )
        
        self.nn_optimizer = optim.Adam(self.nn.parameters(), lr=lr)
        self.moco_criterion = nn.CrossEntropyLoss()
    
    def _compute_recon_loss(self, x_raw, pred_x, dropout_logits=None):
        """Compute reconstruction loss using raw counts."""
        if self.loss_mode == "mse":
            return F.mse_loss(x_raw, pred_x, reduction="none").sum(-1).mean()
        
        l = x_raw.sum(-1, keepdim=True)
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
    
    @torch.no_grad()
    def take_latent(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        
        if self.use_ode:
            q_z, q_m, q_s, t = self.nn.encoder(state)
            
            t_cpu = t.cpu()
            t_sorted, sort_idx, sort_idxr = np.unique(t_cpu, return_index=True, return_inverse=True)
            t_sorted = torch.tensor(t_sorted)
            
            z_base = q_m if self.use_qm else q_z
            z_sorted = z_base[sort_idx]
            z0 = z_sorted[0]
            q_z_ode = self.nn.solve_ode(self.nn.ode_solver, z0, t_sorted)
            q_z_ode = q_z_ode[sort_idxr]
            
            return (self.vae_reg * z_base + self.ode_reg * q_z_ode).cpu().numpy()
        else:
            q_z, q_m, q_s = self.nn.encoder(state)
            return (q_m if self.use_qm else q_z).cpu().numpy()
    
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
        grads = self.nn.ode_solver(t, q_z.cpu()).numpy()
        return grads
    
    @torch.no_grad()
    def take_transition(self, state, top_k: int = 30):
        states = torch.tensor(state, dtype=torch.float).to(self.device)
        q_z, q_m, q_s, t = self.nn.encoder(states)
        
        grads = self.nn.ode_solver(t, q_z.cpu()).numpy()
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
        """
        Training update.
        x_norm: normalized input for forward pass
        x_raw: raw counts for loss computation (if None, uses x_norm)
        """
        if x_raw is None:
            x_raw = x_norm
        
        x_norm_t = torch.tensor(x_norm, dtype=torch.float).to(self.device)
        x_raw_t = torch.tensor(x_raw, dtype=torch.float).to(self.device)
        
        moco_loss = torch.zeros(1, device=self.device)
        
        if self.use_moco and x_q_norm is not None and x_k_norm is not None:
            x_q = torch.tensor(x_q_norm, dtype=torch.float).to(self.device)
            x_k = torch.tensor(x_k_norm, dtype=torch.float).to(self.device)
            outputs = self.nn(x_norm_t, x_q, x_k)
        else:
            outputs = self.nn(x_norm_t)
        
        has_dropout = self.loss_mode in ["zinb", "zip"]
        
        if self.use_ode:
            if has_dropout:
                if self.use_moco:
                    q_z, q_m, q_s, x, pred_x, dropout_logits, le, pred_xl, dropout_logitsl, q_z_ode, pred_x_ode, dropout_logits_ode, logits, labels = outputs
                    moco_loss = self.moco_criterion(logits, labels)
                else:
                    q_z, q_m, q_s, x, pred_x, dropout_logits, le, pred_xl, dropout_logitsl, q_z_ode, pred_x_ode, dropout_logits_ode = outputs
                
                x_raw_sorted = x_raw_t[torch.argsort(self.nn.encoder(x_norm_t)[3])][:len(x)]
                
                recon_loss = self._compute_recon_loss(x_raw_sorted, pred_x, dropout_logits)
                recon_loss += self._compute_recon_loss(x_raw_sorted, pred_x_ode, dropout_logits_ode)
                
                if self.irecon:
                    irecon_loss = self.irecon * self._compute_recon_loss(x_raw_sorted, pred_xl, dropout_logitsl)
                else:
                    irecon_loss = torch.zeros(1, device=self.device)
            else:
                if self.use_moco:
                    q_z, q_m, q_s, x, pred_x, le, pred_xl, q_z_ode, pred_x_ode, logits, labels = outputs
                    moco_loss = self.moco_criterion(logits, labels)
                else:
                    q_z, q_m, q_s, x, pred_x, le, pred_xl, q_z_ode, pred_x_ode = outputs
                
                x_raw_sorted = x_raw_t[torch.argsort(self.nn.encoder(x_norm_t)[3])][:len(x)]
                
                recon_loss = self._compute_recon_loss(x_raw_sorted, pred_x)
                recon_loss += self._compute_recon_loss(x_raw_sorted, pred_x_ode)
                
                if self.irecon:
                    irecon_loss = self.irecon * self._compute_recon_loss(x_raw_sorted, pred_xl)
                else:
                    irecon_loss = torch.zeros(1, device=self.device)
            
            qz_div = F.mse_loss(q_z, q_z_ode, reduction="none").sum(-1).mean()
        
        else:
            if has_dropout:
                if self.use_moco:
                    q_z, q_m, q_s, pred_x, dropout_logits, le, pred_xl, dropout_logitsl, logits, labels = outputs
                    moco_loss = self.moco_criterion(logits, labels)
                else:
                    q_z, q_m, q_s, pred_x, dropout_logits, le, pred_xl, dropout_logitsl = outputs
                
                recon_loss = self._compute_recon_loss(x_raw_t, pred_x, dropout_logits)
                
                if self.irecon:
                    irecon_loss = self.irecon * self._compute_recon_loss(x_raw_t, pred_xl, dropout_logitsl)
                else:
                    irecon_loss = torch.zeros(1, device=self.device)
            else:
                if self.use_moco:
                    q_z, q_m, q_s, pred_x, le, pred_xl, logits, labels = outputs
                    moco_loss = self.moco_criterion(logits, labels)
                else:
                    q_z, q_m, q_s, pred_x, le, pred_xl = outputs
                
                recon_loss = self._compute_recon_loss(x_raw_t, pred_x)
                
                if self.irecon:
                    irecon_loss = self.irecon * self._compute_recon_loss(x_raw_t, pred_xl)
                else:
                    irecon_loss = torch.zeros(1, device=self.device)
            
            qz_div = torch.zeros(1, device=self.device)
        
        # KL divergence
        p_m = torch.zeros_like(q_m)
        p_s = torch.zeros_like(q_s)
        kl_div = self.beta * self._normal_kl(q_m, q_s, p_m, p_s).sum(-1).mean()
        
        # Disentanglement losses
        dip_loss = self.dip * self._dip_loss(q_m, q_s) if self.dip else torch.zeros(1, device=self.device)
        tc_loss = self.tc * self._betatc_compute_total_correlation(q_z, q_m, q_s) if self.tc else torch.zeros(1, device=self.device)
        mmd_loss = self.info * self._compute_mmd(q_z, torch.randn_like(q_z)) if self.info else torch.zeros(1, device=self.device)
        
        total_loss = (
            self.recon * recon_loss +
            irecon_loss +
            qz_div +
            kl_div +
            dip_loss +
            tc_loss +
            mmd_loss +
            self.moco_weight * moco_loss
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
        ))