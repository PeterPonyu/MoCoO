"""
MoCoO Neural Network Modules - Encoder, Decoder, ODE, MoCo, VAE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Union, Literal, Optional
from .mixin import NODEMixin


class Encoder(nn.Module):
    """Variational encoder with optional ODE time prediction."""
    
    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int, use_ode: bool = False):
        super().__init__()
        self.use_ode = use_ode
        self.action_dim = action_dim
        
        self.base_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.latent_params = nn.Linear(hidden_dim, action_dim * 2)
        
        if use_ode:
            self.time_encoder = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )
        
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.01)
    
    def forward(self, x: torch.Tensor):
        x = torch.log1p(x)
        hidden = self.base_network(x)
        
        latent_output = self.latent_params(hidden)
        q_m, q_s = torch.split(latent_output, latent_output.size(-1) // 2, dim=-1)
        std = F.softplus(q_s)
        
        dist = Normal(q_m, std)
        q_z = dist.rsample()
        
        if self.use_ode:
            t = self.time_encoder(hidden).squeeze(-1)
            return q_z, q_m, q_s, t
        
        return q_z, q_m, q_s


class Decoder(nn.Module):
    """Decoder supporting MSE, NB, ZINB, Poisson, ZIP likelihoods."""
    
    VALID_MODES = ('mse', 'nb', 'zinb', 'poisson', 'zip')
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        action_dim: int,
        loss_mode: Literal["mse", "nb", "zinb", "poisson", "zip"] = "nb",
    ):
        super().__init__()
        if loss_mode not in self.VALID_MODES:
            raise ValueError(f"loss_mode must be one of {self.VALID_MODES}, got '{loss_mode}'")
        
        self.loss_mode = loss_mode
        
        self.base_network = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        if loss_mode in ["nb", "zinb", "poisson", "zip"]:
            self.register_buffer("_disp", torch.zeros(state_dim))
            self.disp_param = nn.Parameter(torch.randn(state_dim))
            self.mean_decoder = nn.Sequential(
                nn.Linear(hidden_dim, state_dim),
                nn.Softmax(dim=-1)
            )
        else:
            self.mean_decoder = nn.Linear(hidden_dim, state_dim)
        
        if loss_mode in ["zinb", "zip"]:
            self.dropout_decoder = nn.Linear(hidden_dim, state_dim)
        
        self.apply(self._init_weights)
    
    @property
    def disp(self):
        if hasattr(self, 'disp_param'):
            return self.disp_param
        return self._disp
    
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.01)
    
    def forward(self, x: torch.Tensor):
        hidden = self.base_network(x)
        mean = self.mean_decoder(hidden)
        
        if self.loss_mode in ["zinb", "zip"]:
            dropout_logits = self.dropout_decoder(hidden)
            return mean, dropout_logits
        
        return mean


class LatentODEfunc(nn.Module):
    """Neural ODE function for latent dynamics."""
    
    def __init__(self, n_latent: int = 10, n_hidden: int = 25):
        super().__init__()
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(n_latent, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_latent)
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        return out


class MoCo(nn.Module):
    """Momentum Contrast with projection head."""
    
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
    ):
        super().__init__()
        self.K = K
        self.m = m
        self.T = T
        self.device = device
        self.encoder_q = encoder_q
        self.encoder_k = encoder_k
        
        latent_dim = encoder_q.action_dim
        
        # Projection heads for contrastive learning
        self.proj_head_q = nn.Sequential(
            nn.Linear(latent_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        ).to(device)
        
        self.proj_head_k = nn.Sequential(
            nn.Linear(latent_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        ).to(device)
        
        # Initialize key encoder and projection
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        for param_q, param_k in zip(self.proj_head_q.parameters(), self.proj_head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        self.register_buffer("queue", torch.randn(dim, K, device=device))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long, device=device))
    
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
        # Query path
        q_out = self.encoder_q(exp_q)
        q_m = q_out[1]
        q = self.proj_head_q(q_m)
        q = F.normalize(q, dim=1)
        
        # Key path (momentum)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k_out = self.encoder_k(exp_k)
            k_m = k_out[1]
            k = self.proj_head_k(k_m)
            k = F.normalize(k, dim=1)
        
        # Contrastive logits
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
        
        self._dequeue_and_enqueue(k)
        
        return logits, labels


class VAE(nn.Module, NODEMixin):
    """VAE with ODE regularization and MoCo contrastive learning."""
    
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
        device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    ):
        super().__init__()
        self.use_moco = use_moco
        self.use_ode = use_ode
        
        self.encoder = Encoder(state_dim, hidden_dim, action_dim, use_ode).to(device)
        self.decoder = Decoder(state_dim, hidden_dim, action_dim, loss_mode).to(device)
        
        if use_ode:
            self.ode_solver = LatentODEfunc(action_dim).to(device)
        
        if self.use_moco:
            self.encoder_k = Encoder(state_dim, hidden_dim, action_dim, use_ode).to(device)
            self.moco = MoCo(
                self.encoder,
                self.encoder_k,
                state_dim,
                dim=action_dim,
                K=moco_K,
                T=moco_T,
                device=device,
            )
        
        # Information bottleneck (VAE path only, NOT ODE path)
        self.latent_encoder = nn.Linear(action_dim, i_dim).to(device)
        self.latent_decoder = nn.Linear(i_dim, action_dim).to(device)
    
    def forward(
        self,
        x: torch.Tensor,
        x_q: Optional[torch.Tensor] = None,
        x_k: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, ...]:
        
        if self.encoder.use_ode:
            q_z, q_m, q_s, t = self.encoder(x)
            
            # Sort by time
            idxs = torch.argsort(t)
            t = t[idxs]
            q_z = q_z[idxs]
            q_m = q_m[idxs]
            q_s = q_s[idxs]
            x = x[idxs]
            
            # Remove duplicates
            unique_mask = torch.ones_like(t, dtype=torch.bool)
            unique_mask[1:] = t[1:] != t[:-1]
            t, q_z, q_m, q_s, x = t[unique_mask], q_z[unique_mask], q_m[unique_mask], q_s[unique_mask], x[unique_mask]
            
            # ODE path (separate from bottleneck)
            z0 = q_z[0]
            q_z_ode = self.solve_ode(self.ode_solver, z0, t)
            
            # Bottleneck (VAE path only)
            le = self.latent_encoder(q_z)
            ld = self.latent_decoder(le)
            
            if self.decoder.loss_mode in ["zinb", "zip"]:
                pred_x, dropout_logits = self.decoder(q_z)
                pred_xl, dropout_logitsl = self.decoder(ld)
                pred_x_ode, dropout_logits_ode = self.decoder(q_z_ode)
                
                base = (q_z, q_m, q_s, x, pred_x, dropout_logits, le, pred_xl, dropout_logitsl, q_z_ode, pred_x_ode, dropout_logits_ode)
            else:
                pred_x = self.decoder(q_z)
                pred_xl = self.decoder(ld)
                pred_x_ode = self.decoder(q_z_ode)
                
                base = (q_z, q_m, q_s, x, pred_x, le, pred_xl, q_z_ode, pred_x_ode)
            
            if self.use_moco and x_q is not None and x_k is not None:
                logits, labels = self.moco(x_q, x_k)
                return base + (logits, labels)
            return base
        
        else:
            q_z, q_m, q_s = self.encoder(x)
            
            le = self.latent_encoder(q_z)
            ld = self.latent_decoder(le)
            
            if self.decoder.loss_mode in ["zinb", "zip"]:
                pred_x, dropout_logits = self.decoder(q_z)
                pred_xl, dropout_logitsl = self.decoder(ld)
                
                base = (q_z, q_m, q_s, pred_x, dropout_logits, le, pred_xl, dropout_logitsl)
            else:
                pred_x = self.decoder(q_z)
                pred_xl = self.decoder(ld)
                
                base = (q_z, q_m, q_s, pred_x, le, pred_xl)
            
            if self.use_moco and x_q is not None and x_k is not None:
                logits, labels = self.moco(x_q, x_k)
                return base + (logits, labels)
            return base