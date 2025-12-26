"""
MoCoO: Momentum Contrast ODE-Regularized VAE for Single-Cell RNA Velocity
"""

from .environment import Env
from .mixin import VectorFieldMixin
import tqdm
import time
import torch
import numpy as np
from anndata import AnnData
from typing import Optional, Dict


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
    recon, irecon, beta, dip, tc, info : float
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
        irecon: float = 0.0,
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
        
        import random
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        
        super().__init__(
            adata=adata,
            layer=layer,
            recon=recon,
            irecon=irecon,
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
        """
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        with tqdm.tqdm(total=epochs, desc="Training", ncols=200) as pbar:
            for epoch in range(epochs):
                train_loss = self.train_epoch()
                
                if (epoch + 1) % val_every == 0 or epoch == 0:
                    val_loss, val_score = self.validate()
                    
                    should_stop, improved = self.check_early_stopping(val_loss, patience)
                    
                    pbar.set_postfix({
                        "Train": f"{train_loss:.2f}",
                        "Val": f"{val_loss:.2f}",
                        "ARI": f"{val_score[0]:.2f}",
                        "NMI": f"{val_score[1]:.2f}",
                        "ASW": f"{val_score[2]:.2f}",
                        "CAL": f"{val_score[3]:.2f}",
                        "DAV": f"{val_score[4]:.2f}",
                        "COR": f"{val_score[5]:.2f}",
                        "Best": f"{self.best_val_loss:.2f}",
                        "Pat": f"{self.patience_counter}/{patience}",
                        "Imp": "✓" if improved else "✗"
                    })
                    
                    if should_stop:
                        self.actual_epochs = epoch + 1
                        print(f"\n\nEarly stopping at epoch {epoch + 1}")
                        print(f"Best validation loss: {self.best_val_loss:.4f}")
                        self.load_best_model()
                        break
                
                pbar.update(1)
            else:
                self.actual_epochs = epochs
        
        self.train_time = time.time() - start_time
        self.peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0.0
        
        return self
    
    def get_latent(self) -> np.ndarray:
        """Extract latent representations."""
        return self.take_latent(self.X)
    
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