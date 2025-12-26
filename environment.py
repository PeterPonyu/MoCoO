"""
Environment for MoCoO model with train/val/test splits and DataLoaders.
"""

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
    """
    Environment for MoCoO with train/val/test splits, normalized + raw dual data.
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
        # Data split configuration
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.aug_prob = aug_prob
        self.mask_prob = mask_prob
        self.noise_prob = noise_prob
        
        # Register and split data (creates X_norm, X_raw, and all splits)
        self._register_anndata(adata, layer, latent_dim)
        
        print(f"Batch size: {self.batch_size}")
        
        # Create DataLoaders (uses X_norm for input, but validation uses X_raw for loss)
        self._create_dataloaders()
        
        # Initialize parent model
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
            use_qm=use_qm,
            device=device,
            grad_clip=grad_clip,
        )
        
        # Training metrics
        self.train_losses = []
        self.val_losses = []
        self.val_scores = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def _register_anndata(self, adata, layer: str, latent_dim: int):
        """
        Register AnnData and create dual data splits (normalized + raw).
        
        - X_norm: log-transformed for model input
        - X_raw: raw counts for loss computation (NB/ZINB)
        """
        # Load raw count data
        if layer in adata.layers:
            X_raw = adata.layers[layer]
        elif layer == 'X':
            X_raw = adata.X
        else:
            raise ValueError(f"Layer '{layer}' not found in adata.layers or adata.X")
        
        if issparse(X_raw):
            X_raw = X_raw.toarray()
        
        X_raw = X_raw.astype(np.float32)
        self.n_obs, self.n_var = adata.shape
        
        # Create normalized version: log1p transformation
        X_norm = np.log1p(X_raw).astype(np.float32)
        
        # Get labels
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
        
        # Split data with consistent indices
        np.random.seed(self.random_seed)
        indices = np.random.permutation(self.n_obs)
        
        n_train = int(self.train_size * self.n_obs)
        n_val = int(self.val_size * self.n_obs)
        
        self.train_idx = indices[:n_train]
        self.val_idx = indices[n_train:n_train + n_val]
        self.test_idx = indices[n_train + n_val:]
        
        # Split normalized data (for model input)
        self.X_train_norm = X_norm[self.train_idx]
        self.X_val_norm = X_norm[self.val_idx]
        self.X_test_norm = X_norm[self.test_idx]
        self.X_norm = X_norm
        
        # Split raw data (for loss computation)
        self.X_train_raw = X_raw[self.train_idx]
        self.X_val_raw = X_raw[self.val_idx]
        self.X_test_raw = X_raw[self.test_idx]
        self.X_raw = X_raw
        
        # Alternative names for backward compatibility
        self.X_train = self.X_train_norm
        self.X_val = self.X_val_norm
        self.X_test = self.X_test_norm
        self.X = self.X_norm
        
        # Split labels
        self.labels_train = self.labels[self.train_idx]
        self.labels_val = self.labels[self.val_idx]
        self.labels_test = self.labels[self.test_idx]
        
        print(f"✓ Registered: {self.n_obs} cells × {self.n_var} genes")
        print(f"  Train: {len(self.train_idx):,} | Val: {len(self.val_idx):,} | Test: {len(self.test_idx):,}")
    
    def _create_dataloaders(self):
        """Create PyTorch DataLoaders with normalized data for training."""
        X_train_tensor = torch.FloatTensor(self.X_train_norm)
        X_val_tensor = torch.FloatTensor(self.X_val_norm)
        X_test_tensor = torch.FloatTensor(self.X_test_norm)
        
        train_dataset = TensorDataset(X_train_tensor)
        val_dataset = TensorDataset(X_val_tensor)
        test_dataset = TensorDataset(X_test_tensor)
        
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
        """Train for one complete pass through training data."""
        self.nn.train()
        epoch_losses = []
        
        for (batch_norm,) in self.train_loader:
            batch_norm = batch_norm.to(self.device)
            batch_np = batch_norm.cpu().numpy()
            
            if self.use_moco:
                # Create augmented views for MoCo
                batch_q = torch.tensor(self._augment(batch_np), dtype=torch.float32).to(self.device)
                batch_k = torch.tensor(self._augment(batch_np), dtype=torch.float32).to(self.device)
                self.update(batch_np, batch_q.cpu().numpy(), batch_k.cpu().numpy())
            else:
                self.update(batch_np)
            
            if len(self.loss) > 0:
                epoch_losses.append(self.loss[-1][0])
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self) -> Tuple[float, tuple]:
        """Evaluate model on validation set using raw counts for loss."""
        self.nn.eval()
        val_losses = []
        all_latents = []
        
        with torch.no_grad():
            for batch_idx, (batch_norm,) in enumerate(self.val_loader):
                batch_norm = batch_norm.to(self.device)
                
                # Get corresponding raw batch
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(self.X_val_raw))
                batch_raw = torch.FloatTensor(self.X_val_raw[start_idx:end_idx]).to(self.device)
                
                # Forward pass with normalized input
                if self.use_moco:
                    batch_np = batch_norm.cpu().numpy()
                    batch_q = torch.tensor(self._augment(batch_np), dtype=torch.float32).to(self.device)
                    batch_k = torch.tensor(self._augment(batch_np), dtype=torch.float32).to(self.device)
                    outputs = self.nn(batch_norm, batch_q, batch_k)
                else:
                    outputs = self.nn(batch_norm)
                
                # Parse outputs based on mode combination
                loss = self._compute_validation_loss(outputs, batch_raw)
                val_losses.append(loss.item())
                
                # Extract latent for metrics (first output is q_z)
                latent = outputs[0].cpu().numpy()
                all_latents.append(latent)
        
        avg_val_loss = np.mean(val_losses) if val_losses else float('inf')
        self.val_losses.append(avg_val_loss)
        
        # Compute metrics on validation latents
        all_latents = np.concatenate(all_latents, axis=0)
        val_score = self._calc_score_with_labels(all_latents, self.labels_val)
        self.val_scores.append(val_score)
        
        return avg_val_loss, val_score
    
    def _compute_validation_loss(self, outputs: tuple, batch_raw: torch.Tensor) -> torch.Tensor:
        """
        Compute validation loss from outputs (handles all mode combinations).
        
        Outputs structure varies by (use_ode, use_moco, loss_mode):
        - ODE=False, MoCo=False, ZINB: (q_z, q_m, q_s, pred_x, dropout_logits, le, pred_xl, dropout_logitsl)
        - ODE=False, MoCo=True, ZINB: (..., logits, labels)
        - ODE=False, MoCo=False, NB/MSE: (q_z, q_m, q_s, pred_x, le, pred_xl)
        - ODE=False, MoCo=True, NB/MSE: (..., logits, labels)
        - ODE=True, MoCo=False, ZINB: (..., x, ..., q_z_ode, pred_x_ode, dropout_logits_ode)
        - ODE=True, MoCo=True, ZINB: (..., logits, labels)
        - ODE=True, MoCo=False, NB/MSE: (..., q_z_ode, pred_x_ode)
        - ODE=True, MoCo=True, NB/MSE: (..., logits, labels)
        """
        q_z, q_m, q_s = outputs[0], outputs[1], outputs[2]
        
        # Extract predictions based on mode
        if self.use_ode:
            # ODE mode: (q_z, q_m, q_s, x, pred_x, ...)
            x_sorted = outputs[3]  # Sorted input
            pred_x_idx = 4
        else:
            # Non-ODE mode: (q_z, q_m, q_s, pred_x, ...)
            x_sorted = batch_raw
            pred_x_idx = 3
        
        if self.loss_mode == "zinb":
            pred_x = outputs[pred_x_idx]
            dropout_logits = outputs[pred_x_idx + 1]
            le = outputs[pred_x_idx + 2]
            pred_xl = outputs[pred_x_idx + 3]
            dropout_logitsl = outputs[pred_x_idx + 4]
            
            # Reconstruction loss (main path)
            l = x_sorted.sum(-1).view(-1, 1)
            pred_x_scaled = pred_x * l + 1e-8
            disp = torch.exp(self.nn.decoder.disp)
            recon_loss = -self._log_zinb(x_sorted, pred_x_scaled, disp, dropout_logits).sum(-1).mean()
            
            # Information bottleneck reconstruction loss
            if self.irecon > 0:
                pred_xl_scaled = pred_xl * l + 1e-8
                irecon_loss = self.irecon * (-self._log_zinb(x_sorted, pred_xl_scaled, disp, dropout_logitsl).sum(-1).mean())
            else:
                irecon_loss = torch.tensor(0.0, device=self.device)
        
        else:  # NB or MSE
            pred_x = outputs[pred_x_idx]
            le = outputs[pred_x_idx + 1]
            pred_xl = outputs[pred_x_idx + 2]
            
            if self.loss_mode == "nb":
                l = x_sorted.sum(-1).view(-1, 1)
                pred_x_scaled = pred_x * l + 1e-8
                disp = torch.exp(self.nn.decoder.disp)
                recon_loss = -self._log_nb(x_sorted, pred_x_scaled, disp).sum(-1).mean()
                
                if self.irecon > 0:
                    pred_xl_scaled = pred_xl * l + 1e-8
                    irecon_loss = self.irecon * (-self._log_nb(x_sorted, pred_xl_scaled, disp).sum(-1).mean())
                else:
                    irecon_loss = torch.tensor(0.0, device=self.device)
            
            else:  # MSE
                recon_loss = F.mse_loss(x_sorted, pred_x, reduction="none").sum(-1).mean()
                
                if self.irecon > 0:
                    irecon_loss = self.irecon * F.mse_loss(x_sorted, pred_xl, reduction="none").sum(-1).mean()
                else:
                    irecon_loss = torch.tensor(0.0, device=self.device)
        
        # KL divergence
        p_m = torch.zeros_like(q_m)
        p_s = torch.zeros_like(q_s)
        kl_loss = self.beta * self._normal_kl(q_m, q_s, p_m, p_s).sum(dim=-1).mean()
        
        # Total loss
        total_loss = self.recon * recon_loss + irecon_loss + kl_loss
        
        return total_loss
    
    def check_early_stopping(self, val_loss: float, patience: int = 25) -> Tuple[bool, bool]:
        """
        Check early stopping condition.
        
        Returns
        -------
        should_stop : bool
            Whether to stop training
        improved : bool
            Whether validation loss improved
        """
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
        """Restore best model from checkpoint."""
        if hasattr(self, 'best_model_state') and self.best_model_state is not None:
            self.nn.load_state_dict(self.best_model_state)
    
    def _augment(self, profile: np.ndarray) -> np.ndarray:
        """Apply data augmentation for MoCo."""
        profile_aug = profile.copy().astype(np.float32)
        
        if np.random.rand() < self.aug_prob:
            # Random masking
            mask_genes = np.random.choice(
                [True, False],
                self.n_var,
                p=[self.mask_prob, 1.0 - self.mask_prob]
            )
            profile_aug[:, mask_genes] = 0
            
            # Random noise
            noise_genes = np.random.choice(
                [True, False],
                self.n_var,
                p=[self.noise_prob, 1.0 - self.noise_prob]
            )
            n_noise = np.sum(noise_genes)
            if n_noise > 0:
                noise = np.random.normal(0, 0.2, (profile_aug.shape[0], n_noise))
                profile_aug[:, noise_genes] += noise
        
        return np.clip(profile_aug, 0, None).astype(np.float32)
    
    def take_latent(self, X_norm: np.ndarray) -> np.ndarray:
        """Extract latent representations."""
        self.nn.eval()
        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        
        with torch.no_grad():
            outputs = self.nn(X_tensor)
            q_z = outputs[0]  # First output is always q_z
        
        return q_z.cpu().numpy()
    
    def take_bottleneck(self, X_norm: np.ndarray) -> np.ndarray:
        """Extract information bottleneck representations."""
        self.nn.eval()
        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        
        with torch.no_grad():
            outputs = self.nn(X_tensor)
            
            # Extract bottleneck based on mode
            if self.use_ode:
                if self.loss_mode == "zinb":
                    le = outputs[6]  # (q_z, q_m, q_s, x, pred_x, dropout, le, ...)
                else:
                    le = outputs[5]  # (q_z, q_m, q_s, x, pred_x, le, ...)
            else:
                if self.loss_mode == "zinb":
                    le = outputs[5]  # (q_z, q_m, q_s, pred_x, dropout, le, ...)
                else:
                    le = outputs[4]  # (q_z, q_m, q_s, pred_x, le, ...)
        
        return le.cpu().numpy()
    
    def take_time(self, X_norm: np.ndarray) -> np.ndarray:
        """Extract pseudotime (ODE mode only)."""
        if not self.use_ode:
            raise RuntimeError("take_time() requires use_ode=True")
        
        self.nn.eval()
        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        
        with torch.no_grad():
            # Encoder returns (q_z, q_m, q_s, t) when ODE is enabled
            q_z, q_m, q_s, t = self.nn.encoder(X_tensor)
        
        return t.cpu().numpy()
    
    def take_grad(self, X_norm: np.ndarray) -> np.ndarray:
        """Extract velocity vectors dz/dt (ODE mode only)."""
        if not self.use_ode:
            raise RuntimeError("take_grad() requires use_ode=True")
        
        self.nn.eval()
        X_tensor = torch.FloatTensor(X_norm).to(self.device)
        
        with torch.no_grad():
            q_z, q_m, q_s, t = self.nn.encoder(X_tensor)
            
            # Compute velocities via ODE function
            velocities = self.nn.ode_solver(torch.zeros(1, device=self.device), q_z)
        
        return velocities.cpu().numpy()
    
    def take_transition(self, X_norm: np.ndarray, top_k: int = 30) -> np.ndarray:
        """Extract transition probabilities (ODE mode only)."""
        if not self.use_ode:
            raise RuntimeError("take_transition() requires use_ode=True")
        
        # Get velocities
        velocities = self.take_grad(X_norm)
        
        # Compute cosine similarity
        norms = np.linalg.norm(velocities, axis=1, keepdims=True) + 1e-12
        sim = (velocities @ velocities.T) / (norms @ norms.T)
        np.fill_diagonal(sim, 0.0)
        
        # Keep top-k neighbors
        transition = np.zeros_like(sim)
        for i in range(sim.shape[0]):
            top_indices = np.argsort(-sim[i])[:top_k]
            transition[i, top_indices] = sim[i, top_indices]
        
        # Normalize to probabilities
        transition = transition / (transition.sum(axis=1, keepdims=True) + 1e-12)
        
        return transition