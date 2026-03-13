"""Neural network modules for MoCoO."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Dict, Optional, Literal
from .mixin import NODEMixin


def _init_weights(m):
    """Initialize linear layers with Xavier uniform weights and zero biases."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Encoder(nn.Module):
    """Variational encoder: x -> q(z|x) with optional time prediction.
    
    Follows HSDE architecture: LayerNorm after hidden layers,
    clamped q_m/q_s for numerical stability.
    """
    
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int,
                 use_ode: bool = False, use_layer_norm: bool = True):
        super().__init__()
        self.use_ode = use_ode
        self.action_dim = action_dim
        self.use_layer_norm = use_layer_norm
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.latent_params = nn.Linear(hidden_dim, action_dim * 2)
        
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
        
        if use_ode:
            self.time_encoder = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
        
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor):
        # No log1p here — data is already log1p-normalized in preprocessing
        h = F.relu(self.ln1(self.fc1(x)) if self.use_layer_norm else self.fc1(x))
        h = F.relu(self.ln2(self.fc2(h)) if self.use_layer_norm else self.fc2(h))
        
        latent_output = self.latent_params(h)
        q_m, q_s = torch.split(latent_output, latent_output.size(-1) // 2, dim=-1)
        
        # Clamp for numerical stability (following HSDE)
        q_m = q_m.clamp(-10, 10)
        q_s = q_s.clamp(-10, 10)
        std = F.softplus(q_s).clamp(1e-6, 5.0)
        
        dist = Normal(q_m, std)
        q_z = dist.rsample()
        
        if self.use_ode:
            t = self.time_encoder(h).squeeze(-1)
            return q_z, q_m, q_s, t
        
        return q_z, q_m, q_s


class Decoder(nn.Module):
    """Generative decoder: z -> p(x|z) with count-based likelihoods.
    
    Follows HSDE architecture: LayerNorm after hidden layers.
    """
    
    VALID_MODES = ('mse', 'nb', 'zinb', 'poisson', 'zip')
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        loss_mode: Literal["mse", "nb", "zinb", "poisson", "zip"] = "nb",
        use_layer_norm: bool = True,
    ):
        super().__init__()
        if loss_mode not in self.VALID_MODES:
            raise ValueError(f"loss_mode must be one of {self.VALID_MODES}, got '{loss_mode}'")
        
        self.loss_mode = loss_mode
        self.use_layer_norm = use_layer_norm
        
        self.fc1 = nn.Linear(action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
        
        if loss_mode in ["nb", "zinb", "poisson", "zip"]:
            self.disp = nn.Parameter(torch.randn(state_dim))
            self.mean_decoder = nn.Sequential(
                nn.Linear(hidden_dim, state_dim),
                nn.Softmax(dim=-1)
            )
        else:
            self.mean_decoder = nn.Linear(hidden_dim, state_dim)
        
        if loss_mode in ["zinb", "zip"]:
            self.dropout_decoder = nn.Linear(hidden_dim, state_dim)
        
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor):
        h = F.relu(self.ln1(self.fc1(x)) if self.use_layer_norm else self.fc1(x))
        h = F.relu(self.ln2(self.fc2(h)) if self.use_layer_norm else self.fc2(h))
        mean = self.mean_decoder(h)
        
        if self.loss_mode in ["zinb", "zip"]:
            dropout_logits = self.dropout_decoder(h)
            return mean, dropout_logits
        
        return mean


class LatentODEfunc(nn.Module):
    """Time-conditioned ODE dynamics: dz/dt = f(t, z).
    
    Follows HSDE's BaseControlledSDE drift architecture with time conditioning.
    Supports 'concat', 'film', and 'add' modes.
    """
    
    def __init__(self, n_latent: int = 10, n_hidden: int = 25,
                 time_cond: str = 'concat'):
        super().__init__()
        self.time_cond = time_cond
        self.elu = nn.ELU()
        
        # Build time-conditioned network (following HSDE pattern)
        if time_cond == 'concat':
            self.fc1 = nn.Linear(n_latent + 1, n_hidden)
        elif time_cond == 'film':
            self.fc1 = nn.Linear(n_latent, n_hidden)
            self.time_scale = nn.Linear(1, n_hidden)
            self.time_shift = nn.Linear(1, n_hidden)
        else:  # 'add'
            self.fc1 = nn.Linear(n_latent, n_hidden)
            self.time_embed = nn.Linear(1, n_hidden)
        
        self.fc2 = nn.Linear(n_hidden, n_latent)
        
        self.apply(_init_weights)

    def _broadcast_time(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Broadcast scalar/1D time to match x's shape for concat/film/add."""
        if x.dim() == 1:
            # x is (latent_dim,) — from odeint with 1D z0
            return t.reshape(1) if t.dim() == 0 else t.view(-1)[:1]
        else:
            # x is (batch, latent_dim)
            batch_size = x.shape[0]
            if t.dim() == 0 or t.numel() == 1:
                return t.reshape(1, 1).expand(batch_size, 1)
            return t.view(-1, 1)
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t_bc = self._broadcast_time(t, x)
        
        if self.time_cond == 'concat':
            h = torch.cat([x, t_bc], dim=-1)
            h = self.fc1(h)
        elif self.time_cond == 'film':
            h = self.fc1(x)
            h = self.time_scale(t_bc) * h + self.time_shift(t_bc)
        else:  # 'add'
            h = self.fc1(x) + self.time_embed(t_bc)
        
        h = self.elu(h)
        return self.fc2(h)


class MoCo(nn.Module):
    """Enhanced Momentum Contrast integrating scAGCL + scGPCL strategies.
    
    Strategies from PanODE-LAB:
    - MoCo: Momentum encoder + memory queue
    - scAGCL: Symmetric contrastive with sim_11/sim_12/sim_22
    - scGPCL: Instance + Prototype contrastive
    - Topic-aware contrastive for ODE-VAE alignment
    """
    
    def __init__(
        self,
        encoder_q,
        encoder_k,
        state_dim,
        dim=128,
        K=65536,
        m=0.999,
        T=0.2,
        device=torch.device("cuda"),
        use_prototype=False,
        n_prototypes=10,
    ):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        self.device = device
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        self.use_prototype = use_prototype
        self.n_prototypes = n_prototypes
        
        latent_dim = encoder_q.action_dim
        
        # scAGCL-style 2-layer MLP with BatchNorm (per PanODE-LAB)
        self.proj_head_q = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, dim)
        ).to(device)
        
        self.proj_head_k = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, dim)
        ).to(device)
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        for param_q, param_k in zip(self.proj_head_q.parameters(), self.proj_head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        self.register_buffer("queue", torch.randn(dim, K, device=device))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long, device=device))
        
        # scGPCL-style prototypes
        if use_prototype:
            self.prototypes = nn.Parameter(torch.randn(n_prototypes, dim))
            nn.init.xavier_uniform_(self.prototypes)
    
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
        for param_q, param_k in zip(self.proj_head_q.parameters(), self.proj_head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        else:
            part1_size = self.K - ptr
            self.queue[:, ptr:] = keys[:part1_size].T
            self.queue[:, :batch_size - part1_size] = keys[part1_size:].T
        
        self.queue_ptr[0] = (ptr + batch_size) % self.K
    
    def forward(self, exp_q, exp_k):
        q_out = self.encoder_q(exp_q)
        q_m = q_out[1]
        q = self.proj_head_q(q_m)
        q = F.normalize(q, dim=1)
        
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k_out = self.encoder_k(exp_k)
            k_m = k_out[1]
            k = self.proj_head_k(k_m)
            k = F.normalize(k, dim=1)
        
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
        self._dequeue_and_enqueue(k)
        
        return logits, labels

    def cross_path_contrastive(self, q_z: torch.Tensor, q_z_ode: torch.Tensor) -> torch.Tensor:
        """VAE↔ODE same-cell positive pairs, symmetric cross-entropy."""
        batch_size = q_z.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=q_z.device)
        
        h_vae = F.normalize(self.proj_head_q(q_z), dim=1)
        h_ode = F.normalize(self.proj_head_q(q_z_ode), dim=1)
        
        sim = torch.mm(h_vae, h_ode.T) / self.T
        labels = torch.arange(batch_size, device=q_z.device)
        
        return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2

    def prototype_contrastive_loss(self, z: torch.Tensor, cluster_assignments: Optional[torch.Tensor] = None) -> torch.Tensor:
        """scGPCL-style prototype-level contrastive loss."""
        if not self.use_prototype:
            return torch.tensor(0.0, device=z.device)
        
        h = self.proj_head_q(z)
        h = F.normalize(h, dim=1)
        prototypes = F.normalize(self.prototypes.to(z.device), dim=1)
        
        # Similarity to prototypes
        sim = torch.mm(h, prototypes.T) / self.T
        
        if cluster_assignments is not None:
            # Supervised: use provided cluster assignments
            pos_mask = F.one_hot(cluster_assignments, num_classes=self.n_prototypes).float()
            pos_sim = (sim * pos_mask).sum(dim=1)
        else:
            # Self-supervised: assign to nearest prototype
            assignments = sim.argmax(dim=1)
            pos_mask = F.one_hot(assignments, num_classes=self.n_prototypes).float()
            pos_sim = (sim * pos_mask).sum(dim=1)
        
        loss = -pos_sim + torch.logsumexp(sim, dim=1)
        return loss.mean()


class VAE(nn.Module, NODEMixin):
    """VAE with optional Neural ODE and MoCo contrastive learning.
    
    Architecture follows HSDE for base VAE+ODE, with MoCoO's contrastive logic.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        i_dim: int,
        use_ode: bool,
        use_moco: bool,
        loss_mode: Literal["mse", "nb", "zinb", "poisson", "zip"] = "nb",
        moco_T: float = 0.2,
        moco_K: int = 4096,
        use_prototype: bool = False,
        n_prototypes: int = 10,
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    ):
        super().__init__()
        self.use_moco = use_moco
        self.use_ode = use_ode
        
        self.encoder = Encoder(state_dim, hidden_dim, action_dim, use_ode,
                               use_layer_norm=True).to(device)
        self.decoder = Decoder(state_dim, hidden_dim, action_dim, loss_mode,
                               use_layer_norm=True).to(device)
        
        if use_ode:
            self.ode_solver = LatentODEfunc(action_dim, time_cond='concat').to(device)
        
        if self.use_moco:
            self.encoder_k = Encoder(state_dim, hidden_dim, action_dim, use_ode,
                                     use_layer_norm=True).to(device)
            self.moco = MoCo(
                self.encoder,
                self.encoder_k,
                state_dim,
                dim=action_dim,
                K=moco_K,
                T=moco_T,
                device=device,
                use_prototype=use_prototype,
                n_prototypes=n_prototypes,
            )
        
        self.latent_encoder = nn.Linear(action_dim, i_dim).to(device)
    
    def _decode(self, z: torch.Tensor) -> dict:
        out = self.decoder(z)
        if self.decoder.loss_mode in ["zinb", "zip"]:
            return {'pred': out[0], 'dropout': out[1]}
        return {'pred': out, 'dropout': None}
    
    def forward(
        self,
        x: torch.Tensor,
        x_q: Optional[torch.Tensor] = None,
        x_k: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        result = {}
        
        if self.encoder.use_ode:
            q_z, q_m, q_s, t = self.encoder(x)
            
            # Sort by pseudotime (following HSDE)
            idxs = torch.argsort(t)
            t, q_z, q_m, q_s, x = t[idxs], q_z[idxs], q_m[idxs], q_s[idxs], x[idxs]
            
            # Remove duplicate time points using boolean mask (following HSDE)
            if len(t) > 1:
                unique_mask = torch.cat([
                    torch.tensor([True], device=t.device),
                    t[1:] != t[:-1]
                ])
                t = t[unique_mask]
                q_z = q_z[unique_mask]
                q_m = q_m[unique_mask]
                q_s = q_s[unique_mask]
                x = x[unique_mask]
                idxs = idxs[unique_mask]
            
            # Ensure strictly increasing times for ODE solver
            with torch.no_grad():
                min_dt = 1e-6
                t_fixed = t.clone()
                for i in range(1, len(t_fixed)):
                    if t_fixed[i] <= t_fixed[i-1] + min_dt:
                        t_fixed[i] = t_fixed[i-1] + min_dt
                t = t_fixed
            
            z0 = q_z[0]
            # ODE integrates from z0 along the learned time axis.
            # Gradients flow through ODE solver for joint encoder-ODE
            # training (dual-path reconstruction, following HSDE).
            q_z_ode = self.solve_ode(self.ode_solver, z0, t)
            velocity = self.ode_solver(t, q_z)
            
            # Dual-path reconstruction: encoder path + ODE path
            vae_dec = self._decode(q_z)          # encoder-derived latent
            ode_dec = self._decode(q_z_ode)      # ODE-derived latent
            
            result.update({
                'q_z': q_z, 'q_m': q_m, 'q_s': q_s,
                'x_sorted': x, 't': t,
                'sort_idx': idxs,
                'pred_x': vae_dec['pred'], 'dropout_x': vae_dec['dropout'],
                'pred_x_ode': ode_dec['pred'], 'dropout_x_ode': ode_dec['dropout'],
                'q_z_ode': q_z_ode,
                'velocity': velocity,
            })
            
            if self.use_moco and x_q is not None and x_k is not None:
                moco_logits, moco_labels = self.moco(x_q, x_k)
                # stop-gradient on q_z: ODE aligns to encoder, not vice versa
                cross_path_loss = self.moco.cross_path_contrastive(q_z.detach(), q_z_ode)
                result.update({
                    'moco_logits': moco_logits,
                    'moco_labels': moco_labels,
                    'cross_path_loss': cross_path_loss,
                })
        
        else:
            q_z, q_m, q_s = self.encoder(x)
            
            vae_dec = self._decode(q_z)
            
            result.update({
                'q_z': q_z, 'q_m': q_m, 'q_s': q_s,
                'pred_x': vae_dec['pred'], 'dropout_x': vae_dec['dropout'],
            })
            
            if self.use_moco and x_q is not None and x_k is not None:
                moco_logits, moco_labels = self.moco(x_q, x_k)
                result.update({
                    'moco_logits': moco_logits,
                    'moco_labels': moco_labels,
                })
        
        return result
