"""MoCoO model - training and loss computation."""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from .mixin import scviMixin, dipMixin, betatcMixin, infoMixin
from .module import VAE, FlowMatchingVelocity


class MoCoOModel(scviMixin, dipMixin, betatcMixin, infoMixin):
    """MoCoO model with VAE, ODE, and MoCo."""
    
    def __init__(
        self,
        recon, beta, dip, tc, info,
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

    @torch.no_grad()
    def take_latent(self, state, use_qm=None):
        """Extract latent representation (scTour-style ODE blending)."""
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        effective_qm = use_qm if use_qm is not None else self.use_qm

        if self.use_ode:
            q_z, q_m, q_s, t = self.nn.encoder(state)
            t = t.cpu()
            
            # Sort and deduplicate using np.unique (following scTour _get_latentsp)
            t_sorted, sort_idx, sort_ridx = np.unique(
                t.numpy(), return_index=True, return_inverse=True)
            t_sorted = torch.tensor(t_sorted)
            
            z = q_z if not effective_qm else q_m
            z_sorted = z[sort_idx]
            z0 = z_sorted[0]
            
            q_z_ode = self.nn.solve_ode(self.nn.ode_solver, z0, t_sorted)
            q_z_ode = q_z_ode[sort_ridx]  # unsort back to original order
            
            blended = self.vae_reg * z + self.ode_reg * q_z_ode
            return blended.cpu().numpy()
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
        ode_device = next(self.nn.ode_solver.parameters()).device
        grads = self.nn.ode_solver(t.to(ode_device), q_z.to(ode_device)).cpu().numpy()
        return grads
    
    @torch.no_grad()
    def take_transition(self, state, top_k: int = 30):
        states = torch.tensor(state, dtype=torch.float).to(self.device)
        q_z, q_m, q_s, t = self.nn.encoder(states)
        
        ode_device = next(self.nn.ode_solver.parameters()).device
        grads = self.nn.ode_solver(t.to(ode_device), q_z.to(ode_device)).cpu().numpy()
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
            
            # Symmetric dual-path reconstruction (following scTour)
            recon_loss_ec = self._compute_recon_loss(
                x_raw_sorted, out['pred_x'], out.get('dropout_x'))
            recon_loss_ode = self._compute_recon_loss(
                x_raw_sorted, out['pred_x_ode'], out.get('dropout_x_ode'))
            recon_loss = self.vae_reg * recon_loss_ec + self.ode_reg * recon_loss_ode
            
            # Bidirectional z_div (following scTour): both encoder and ODE
            # co-adapt toward each other
            qz_div = F.mse_loss(q_z, out['q_z_ode'], reduction="none").sum(-1).mean()
        else:
            recon_loss = self._compute_recon_loss(x_raw_t, out['pred_x'], out.get('dropout_x'))
            qz_div = torch.tensor(0.0, device=self.device)
        
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
            qz_div +
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
            kl_div.item(),
            dip_loss.item() if isinstance(dip_loss, torch.Tensor) else dip_loss,
            tc_loss.item() if isinstance(tc_loss, torch.Tensor) else tc_loss,
            mmd_loss.item() if isinstance(mmd_loss, torch.Tensor) else mmd_loss,
            moco_loss.item() if isinstance(moco_loss, torch.Tensor) else moco_loss,
            cross_loss.item() if isinstance(cross_loss, torch.Tensor) else cross_loss,
            qz_div.item() if isinstance(qz_div, torch.Tensor) else qz_div,
            proto_loss.item() if isinstance(proto_loss, torch.Tensor) else proto_loss,
        ))

    # ------------------------------------------------------------------ #
    #  Phase-2: Latent Flow Matching                                      #
    # ------------------------------------------------------------------ #

    def fm_init(self, lr: float = 1e-3, hidden_dim: int = 128,
                time_emb_dim: int = 32):
        """Initialize the flow-matching velocity network and optimizer.

        Call *after* Phase-1 training has converged.  The VAE encoder is
        frozen automatically — only the velocity network is trained.
        """
        latent_dim = self.nn.encoder.action_dim
        self.fm_net = FlowMatchingVelocity(
            latent_dim, hidden_dim, time_emb_dim,
        ).to(self.device)
        self.fm_optimizer = optim.Adam(self.fm_net.parameters(), lr=lr)
        self.fm_loss_history: list[float] = []

        # Freeze encoder (and full VAE) during FM training
        for p in self.nn.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _encode_targets(self, x_norm: np.ndarray) -> torch.Tensor:
        """Encode data to posterior means μ(x) (frozen encoder)."""
        x = torch.tensor(x_norm, dtype=torch.float32, device=self.device)
        if self.nn.encoder.use_ode:
            _, q_m, _, _ = self.nn.encoder(x)
        else:
            _, q_m, _ = self.nn.encoder(x)
        return q_m

    def fm_update(self, z_data: torch.Tensor) -> float:
        """One conditional-FM training step.

        Parameters
        ----------
        z_data : (B, D) tensor of target latents (posterior means).

        Returns
        -------
        loss : scalar FM loss value.
        """
        B, D = z_data.shape

        # Sample noise and time
        eps = torch.randn_like(z_data)
        t = torch.rand(B, device=z_data.device)

        # Linear OT interpolation: z_t = (1-t)·eps + t·z_data
        z_t = (1 - t).unsqueeze(-1) * eps + t.unsqueeze(-1) * z_data

        # Conditional vector field target: u_t = z_data - eps
        target = z_data - eps

        pred = self.fm_net(z_t, t)
        loss = F.mse_loss(pred, target)

        self.fm_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.fm_net.parameters(), self.grad_clip)
        self.fm_optimizer.step()

        loss_val = loss.item()
        self.fm_loss_history.append(loss_val)
        return loss_val

    @torch.no_grad()
    def take_fm_sample(self, n: int, steps: int = 100) -> np.ndarray:
        """Generate *n* new cell latents by integrating noise → data.

        Uses Euler integration of v_θ from t=0 to t=1.
        """
        D = self.nn.encoder.action_dim
        z = torch.randn(n, D, device=self.device)
        dt = 1.0 / steps
        for i in range(steps):
            t_val = torch.full((n,), i * dt, device=self.device)
            z = z + self.fm_net(z, t_val) * dt
        return z.cpu().numpy()

    @torch.no_grad()
    def take_fm_refined(self, state: np.ndarray, t_start: float = 0.5,
                        steps: int = 100) -> np.ndarray:
        """Refine existing latents by partial FM integration.

        Adds noise at level (1 - t_start) and integrates from t_start → 1.
        Higher t_start = less noise = more identity-preserving.

        Parameters
        ----------
        state : (N, G) raw gene-expression matrix (log1p-normalised).
        t_start : float in (0, 1]
            Flow starting time.  0.9 ≈ light smoothing, 0.5 ≈ heavy denoising.
        steps : int
            Euler integration steps from t_start to 1.
        """
        z_data = self._encode_targets(state)
        N = z_data.shape[0]
        eps = torch.randn_like(z_data)

        # Construct z at t_start on the OT path
        z = (1 - t_start) * eps + t_start * z_data
        dt = (1.0 - t_start) / steps
        for i in range(steps):
            t_val = torch.full((N,), t_start + i * dt, device=self.device)
            z = z + self.fm_net(z, t_val) * dt
        return z.cpu().numpy()
