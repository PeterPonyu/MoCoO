"""Environment for MoCoO model."""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from typing import Optional, Tuple
from .model import MoCoOModel
from .mixin import envMixin


class Env(MoCoOModel, envMixin):
    """Training environment for MoCoO.

    Manages data loading, preprocessing, train/validation/test splitting,
    and the training loop with early stopping and validation scoring.
    """

    def __init__(
        self,
        adata,
        layer: str,
        recon: float,
        irecon: float,
        beta: float,
        dip: float,
        tc: float,
        info: float,
        hidden_dim: int,
        latent_dim: int,
        i_dim: int,
        use_ode: bool,
        use_moco: bool,
        loss_mode: str,
        lr: float,
        vae_reg: float,
        ode_reg: float,
        moco_weight: float,
        moco_T: float,
        moco_K: int,
        use_prototype: bool,
        n_prototypes: int,
        proto_weight: float,
        aug_prob: float,
        mask_prob: float,
        noise_prob: float,
        use_qm: bool,
        device: torch.device,
        grad_clip: float = 1.0,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        batch_size: int = 128,
        random_seed: int = 42,
        *args,
        **kwargs,
    ):
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.aug_prob = aug_prob
        self.mask_prob = mask_prob
        self.noise_prob = noise_prob
        
        self._register_anndata(adata, layer, latent_dim)
        self._create_dataloaders()
        
        super().__init__(
            recon=recon,
            irecon=irecon,
            beta=beta,
            dip=dip,
            tc=tc,
            info=info,
            state_dim=self.n_var,
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
            use_qm=use_qm,
            device=device,
            grad_clip=grad_clip,
        )
        
        self.train_losses = []
        self.val_losses = []
        self.val_scores = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def _register_anndata(self, adata, layer: str, latent_dim: int):
        if layer in adata.layers:
            X_raw = adata.layers[layer]
        elif layer == 'X':
            X_raw = adata.X
        else:
            raise ValueError(f"Layer '{layer}' not found")
        
        if issparse(X_raw):
            X_raw = X_raw.toarray()
        
        X_raw = X_raw.astype(np.float32)
        self.n_obs, self.n_var = adata.shape
        
        X_norm = np.log1p(X_raw).astype(np.float32)
        
        if 'cell_type' in adata.obs.columns:
            self.labels = LabelEncoder().fit_transform(adata.obs['cell_type'])
            print(f"✓ Using 'cell_type' labels: {len(np.unique(self.labels))} types")
        else:
            print(f"⚠ Generating KMeans pseudo-labels with {latent_dim} clusters...")
            self.labels = KMeans(
                n_clusters=latent_dim,
                n_init=10,
                max_iter=300,
                random_state=self.random_seed
            ).fit_predict(X_norm)
        
        np.random.seed(self.random_seed)
        indices = np.random.permutation(self.n_obs)
        
        n_train = int(self.train_size * self.n_obs)
        n_val = int(self.val_size * self.n_obs)
        
        self.train_idx = indices[:n_train]
        self.val_idx = indices[n_train:n_train + n_val]
        self.test_idx = indices[n_train + n_val:]
        
        self.X_train_norm = X_norm[self.train_idx]
        self.X_val_norm = X_norm[self.val_idx]
        self.X_test_norm = X_norm[self.test_idx]
        self.X_norm = X_norm
        
        self.X_train_raw = X_raw[self.train_idx]
        self.X_val_raw = X_raw[self.val_idx]
        self.X_test_raw = X_raw[self.test_idx]
        self.X_raw = X_raw
        
        self.X_train = self.X_train_norm
        self.X_val = self.X_val_norm
        self.X_test = self.X_test_norm
        self.X = self.X_norm
        
        self.labels_train = self.labels[self.train_idx]
        self.labels_val = self.labels[self.val_idx]
        self.labels_test = self.labels[self.test_idx]
        
        print(f"✓ Registered: {self.n_obs} cells × {self.n_var} genes")
        print(f"  Train: {len(self.train_idx):,} | Val: {len(self.val_idx):,} | Test: {len(self.test_idx):,}")
    
    def _create_dataloaders(self):
        train_dataset = TensorDataset(
            torch.FloatTensor(self.X_train_norm),
            torch.FloatTensor(self.X_train_raw),
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(self.X_val_norm),
            torch.FloatTensor(self.X_val_raw),
        )
        test_dataset = TensorDataset(torch.FloatTensor(self.X_test_norm))
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )
    
    def train_epoch(self):
        self.nn.train()
        epoch_losses = []
        
        for batch_norm, batch_raw in self.train_loader:
            batch_norm_np = batch_norm.numpy()
            batch_raw_np = batch_raw.numpy()
            
            if self.use_moco:
                batch_q_np = self._augment(batch_norm_np)
                batch_k_np = self._augment(batch_norm_np)
                self.update(batch_norm_np, batch_raw_np, batch_q_np, batch_k_np)
            else:
                self.update(batch_norm_np, batch_raw_np)
            
            if len(self.loss) > 0:
                epoch_losses.append(self.loss[-1][0])
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, track_metrics: bool = True) -> Tuple[float, Optional[tuple]]:
        self.nn.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch_norm, batch_raw in self.val_loader:
                batch_norm = batch_norm.to(self.device)
                batch_raw = batch_raw.to(self.device)
                
                assert len(batch_norm) == len(batch_raw), \
                    f"Batch size mismatch: norm={len(batch_norm)}, raw={len(batch_raw)}"
                
                if self.use_moco:
                    batch_np = batch_norm.cpu().numpy()
                    batch_q = torch.tensor(self._augment(batch_np), dtype=torch.float32).to(self.device)
                    batch_k = torch.tensor(self._augment(batch_np), dtype=torch.float32).to(self.device)
                    out = self.nn(batch_norm, batch_q, batch_k)
                else:
                    out = self.nn(batch_norm)
                
                loss = self._compute_validation_loss(out, batch_raw)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        self.val_losses.append(avg_val_loss)
        
        if track_metrics:
            all_latents = self.take_latent(self.X_val_norm)
            
            assert len(all_latents) == len(self.labels_val), \
                f"Latent/label mismatch: {len(all_latents)} != {len(self.labels_val)}"
            
            val_score = self._calc_score_with_labels(all_latents, self.labels_val)
            self.val_scores.append(val_score)
        else:
            val_score = None
        
        return avg_val_loss, val_score

    def _compute_validation_loss(self, out: dict, batch_raw: torch.Tensor) -> torch.Tensor:
        q_m, q_s = out['q_m'], out['q_s']
        
        # For ODE configs, sort raw data to match the forward pass ordering
        if 'sort_idx' in out:
            sort_idx = out['sort_idx']
            n_sorted = len(out['x_sorted'])
            x_target = batch_raw[sort_idx][:n_sorted]
        else:
            x_target = batch_raw
        
        recon_loss_ec = self._compute_recon_loss(x_target, out['pred_x'], out.get('dropout_x'))
        
        # Dual-path: add ODE reconstruction loss when available
        if 'pred_x_ode' in out:
            recon_loss_ode = self._compute_recon_loss(
                x_target, out['pred_x_ode'], out.get('dropout_x_ode'))
            recon_loss = recon_loss_ec + self.ode_reg * recon_loss_ode
        else:
            recon_loss = recon_loss_ec
        
        if self.irecon > 0:
            irecon_loss = self.irecon * self._compute_recon_loss(
                x_target, out['pred_xl'], out.get('dropout_xl')
            )
        else:
            irecon_loss = torch.tensor(0.0, device=self.device)
        
        p_m = torch.zeros_like(q_m)
        p_s = torch.zeros_like(q_s)
        kl_loss = self.beta * self._normal_kl(q_m, q_s, p_m, p_s).sum(dim=-1).mean()
        
        return self.recon * recon_loss + irecon_loss + kl_loss
    
    def check_early_stopping(self, val_loss: float, patience: int = 25) -> Tuple[bool, bool]:
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_model_state = {
                k: v.cpu().clone() for k, v in self.nn.state_dict().items()
            }
            self.patience_counter = 0
            return False, True
        else:
            self.patience_counter += 1
            return self.patience_counter >= patience, False
    
    def load_best_model(self):
        if hasattr(self, 'best_model_state') and self.best_model_state is not None:
            self.nn.load_state_dict(self.best_model_state)
    
    def _augment(self, profile: np.ndarray) -> np.ndarray:
        """Gene masking + Gaussian noise + feature swapping."""
        profile_aug = profile.copy().astype(np.float32)
        
        if np.random.rand() < self.aug_prob:
            mask_genes = np.random.choice(
                [True, False],
                self.n_var,
                p=[self.mask_prob, 1.0 - self.mask_prob]
            )
            profile_aug[:, mask_genes] = 0
            
            noise_genes = np.random.choice(
                [True, False],
                self.n_var,
                p=[self.noise_prob, 1.0 - self.noise_prob]
            )
            n_noise = np.sum(noise_genes)
            if n_noise > 0:
                noise = np.random.normal(0, 0.2, (profile_aug.shape[0], n_noise))
                profile_aug[:, noise_genes] += noise
            
            if profile_aug.shape[0] > 1:
                swap_prob = self.noise_prob * 0.5
                n_swap = max(1, int(self.n_var * swap_prob))
                swap_genes = np.random.choice(self.n_var, n_swap, replace=False)
                perm = np.random.permutation(profile_aug.shape[0])
                profile_aug[:, swap_genes] = profile_aug[perm][:, swap_genes]
        
        return np.clip(profile_aug, 0, None).astype(np.float32)
