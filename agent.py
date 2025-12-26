"""
MoCoO: Momentum Contrast ODE-Regularized VAE for Single-Cell RNA Velocity
==========================================================================

Agent interface for MoCoO combining:
- Variational Autoencoder (VAE) for dimensionality reduction
- Information bottleneck for hierarchical feature extraction
- Neural ODE for trajectory inference (optional)
- Momentum Contrast (MoCo) for contrastive learning (optional)
- Multiple count-based likelihood functions (MSE, NB, ZINB)
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
    MoCoO: Momentum Contrast ODE-Regularized VAE for Single-Cell Analysis
    
    A unified framework combining variational autoencoders, neural ODEs for 
    trajectory inference, and momentum contrast for robust representation learning.
    
    Supports both scRNA-seq and scATAC-seq modalities with flexible loss modes
    (MSE, Negative Binomial, Zero-Inflated Negative Binomial).
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with raw count data in `adata.layers[layer]`
    layer : str, default='counts'
        Layer name containing raw unnormalized count data
    recon : float, default=1.0
        Weight for reconstruction loss
    irecon : float, default=0.0
        Weight for information bottleneck reconstruction loss
    beta : float, default=1.0
        Weight for KL divergence (β-VAE); >1 encourages disentanglement
    dip : float, default=0.0
        Weight for DIP-VAE disentanglement loss
    tc : float, default=0.0
        Weight for Total Correlation (β-TC-VAE) loss
    info : float, default=0.0
        Weight for Maximum Mean Discrepancy (InfoVAE) loss
    hidden_dim : int, default=128
        Hidden layer dimension in encoder/decoder
    latent_dim : int, default=10
        Primary latent space dimensionality
    i_dim : int, default=2
        Information bottleneck dimension (should be < latent_dim)
    use_ode : bool, default=False
        Enable Neural ODE for trajectory inference
    use_moco : bool, default=False
        Enable Momentum Contrast contrastive learning
    loss_mode : str, default='nb'
        Reconstruction likelihood model:
        - 'mse': Mean Squared Error
        - 'nb': Negative Binomial (recommended for count data)
        - 'zinb': Zero-Inflated Negative Binomial (high dropout)
    lr : float, default=1e-4
        Learning rate for Adam optimizer
    vae_reg : float, default=0.5
        Weight for VAE path in ODE mode (should sum to 1.0 with ode_reg)
    ode_reg : float, default=0.5
        Weight for ODE path in ODE mode
    moco_weight : float, default=1.0
        Weight for MoCo contrastive loss
    moco_T : float, default=0.2
        Temperature parameter for MoCo softmax
    aug_prob : float, default=0.5
        Probability of applying augmentation in MoCo
    mask_prob : float, default=0.1
        Probability of masking genes in augmentation
    noise_prob : float, default=0.1
        Probability of adding noise to genes in augmentation
    use_qm : bool, default=False
        Use mean (q_m) instead of sampled latent (q_z) for inference
    grad_clip : float, default=1.0
        Gradient clipping threshold for training stability
    train_size : float, default=0.7
        Proportion of cells for training set
    val_size : float, default=0.15
        Proportion of cells for validation set
    test_size : float, default=0.15
        Proportion of cells for test set (should sum to 1.0)
    batch_size : int, default=128
        Mini-batch size for training
    random_seed : int, default=42
        Random seed for reproducibility
    device : torch.device, optional
        Computation device (auto-detects CUDA if available)
    
    Attributes
    ----------
    nn : VAE
        The neural network model
    train_losses : list
        Training loss history
    val_losses : list
        Validation loss history
    val_scores : list
        Validation metrics history (ARI, NMI, ASW, etc.)
    best_val_loss : float
        Best validation loss achieved (for early stopping)
    train_time : float
        Total training time in seconds
    peak_memory_gb : float
        Peak GPU memory usage in GB
    actual_epochs : int
        Actual number of epochs trained (may be < epochs if early stopped)
    
    Examples
    --------
    >>> import scanpy as sc
    >>> from liora import MoCoO
    >>> 
    >>> # Load data
    >>> adata = sc.read_h5ad('data.h5ad')
    >>> 
    >>> # Basic VAE
    >>> model = MoCoO(
    ...     adata,
    ...     layer='counts',
    ...     loss_mode='nb',
    ...     batch_size=128
    ... )
    >>> model.fit(epochs=100)
    >>> latent = model.get_latent()
    >>> 
    >>> # With ODE trajectory inference
    >>> model = MoCoO(
    ...     adata,
    ...     use_ode=True,
    ...     latent_dim=10,
    ...     i_dim=2,
    ...     batch_size=128
    ... )
    >>> model.fit(epochs=400, patience=25)
    >>> latent = model.get_latent()
    >>> velocity = model.get_velocity()
    >>> pseudotime = model.get_time()
    >>> 
    >>> # With MoCo contrastive learning
    >>> model = MoCoO(
    ...     adata,
    ...     use_moco=True,
    ...     use_ode=True,
    ...     moco_weight=1.0,
    ...     aug_prob=0.5,
    ...     batch_size=256
    ... )
    >>> model.fit(epochs=400, patience=25)
    >>> latent = model.get_latent()
    >>> bottleneck = model.get_bottleneck()
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
        # Auto-detect device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Validate splits
        if not (0.99 <= train_size + val_size + test_size <= 1.01):
            raise ValueError(
                f"Split sizes must sum to 1.0, got {train_size + val_size + test_size}"
            )
        
        # Validate ODE weights
        if use_ode and not (0.99 <= vae_reg + ode_reg <= 1.01):
            raise ValueError(
                f"ODE weights must sum to 1.0, got {vae_reg + ode_reg}"
            )
        
        # Validate bottleneck dimension
        if i_dim >= latent_dim:
            raise ValueError(
                f"i_dim ({i_dim}) must be < latent_dim ({latent_dim})"
            )
        
        # Set random seed for reproducibility
        import random
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_seed)
        
        # Initialize parent environment
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
        
        # Resource tracking
        self.train_time = 0.0
        self.peak_memory_gb = 0.0
        self.actual_epochs = 0
        
        print(f"\n{'='*70}")
        print(f"MoCoO initialized on {device}")
        print(f"  ODE: {use_ode} | MoCo: {use_moco} | Loss: {loss_mode}")
        print(f"  Architecture: {self.n_var} → {hidden_dim} → {latent_dim} → {i_dim}")
        print(f"  Splits: Train {train_size*100:.0f}% | Val {val_size*100:.0f}% | Test {test_size*100:.0f}%")
        print(f"  Batch size: {batch_size}")
        print(f"{'='*70}\n")
    
    def fit(
        self,
        epochs: int = 400,
        patience: int = 25,
        val_every: int = 5,
    ) -> 'MoCoO':
        """
        Train MoCoO with validation-based early stopping.
        
        Training uses:
        - Adam optimizer with configurable learning rate
        - Early stopping based on validation loss
        - Gradient clipping for stability
        - Periodic validation every `val_every` epochs
        
        Parameters
        ----------
        epochs : int, default=400
            Maximum number of training epochs
        patience : int, default=25
            Early stopping patience (epochs without validation improvement)
        val_every : int, default=5
            Validation frequency (every N epochs)
        
        Returns
        -------
        self : MoCoO
            Returns self for method chaining
        """
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        with tqdm.tqdm(total=epochs, desc="Training", ncols=200) as pbar:
            for epoch in range(epochs):
                # Train one epoch
                train_loss = self.train_epoch()
                
                # Periodic validation
                if (epoch + 1) % val_every == 0 or epoch == 0:
                    val_loss, val_score = self.validate()
                    
                    should_stop, improved = self.check_early_stopping(val_loss, patience)
                    
                    # Update display only on validation epochs
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
        
        # Record resource usage
        self.train_time = time.time() - start_time
        self.peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0.0
        
        return self
    
    # =====================================================================
    # API Methods - HSDE Compatible
    # =====================================================================
    
    def get_latent(self) -> np.ndarray:
        """
        Extract latent representations for all cells.
        
        Returns
        -------
        latent : ndarray of shape (n_cells, latent_dim)
            Low-dimensional cell embeddings
        """
        return self.take_latent(self.X)
    
    def get_bottleneck(self) -> np.ndarray:
        """
        Extract information bottleneck representations.
        
        The bottleneck layer compresses latent features to a lower-dimensional
        space (i_dim), capturing the most essential information for reconstruction.
        
        Returns
        -------
        bottleneck : ndarray of shape (n_cells, i_dim)
            Compressed hierarchical representations
        """
        return self.take_bottleneck(self.X)
    
    def get_test_latent(self) -> np.ndarray:
        """
        Extract latent representations for test set only.
        
        Returns
        -------
        latent : ndarray of shape (n_test, latent_dim)
            Low-dimensional test cell embeddings
        """
        return self.take_latent(self.X_test)
    
    def get_time(self) -> np.ndarray:
        """
        Extract pseudotime for all cells (ODE mode only).
        
        Returns
        -------
        pseudotime : ndarray of shape (n_cells,)
            Predicted cell pseudotime values in [0, 1]
        
        Raises
        ------
        RuntimeError
            If use_ode=False
        """
        if not self.use_ode:
            raise RuntimeError("get_time() requires use_ode=True")
        return self.take_time(self.X)
    
    def get_velocity(self) -> np.ndarray:
        """
        Extract velocity vectors dz/dt (ODE mode only).
        
        Returns
        -------
        velocity : ndarray of shape (n_cells, latent_dim)
            Velocity vectors representing cell state transitions
        
        Raises
        ------
        RuntimeError
            If use_ode=False
        """
        if not self.use_ode:
            raise RuntimeError("get_velocity() requires use_ode=True")
        return self.take_grad(self.X)
    
    def get_transition(self, top_k: int = 30) -> np.ndarray:
        """
        Extract transition probabilities for all cells (ODE mode only).
        
        Computes cell-to-cell transition matrix based on predicted velocities.
        
        Parameters
        ----------
        top_k : int, default=30
            Number of nearest neighbors to keep in sparse transition matrix
        
        Returns
        -------
        transition : ndarray of shape (n_cells, n_cells)
            Cell-to-cell transition probability matrix
        
        Raises
        ------
        RuntimeError
            If use_ode=False
        """
        if not self.use_ode:
            raise RuntimeError("get_transition() requires use_ode=True")
        return self.take_transition(self.X, top_k=top_k)
    
    def get_resource_metrics(self) -> Dict[str, float]:
        """
        Get training resource usage metrics.
        
        Returns
        -------
        metrics : dict
            Dictionary containing:
            - 'train_time': Total training time in seconds
            - 'peak_memory_gb': Peak GPU memory usage in GB
            - 'actual_epochs': Number of epochs trained
        """
        return {
            'train_time': self.train_time,
            'peak_memory_gb': self.peak_memory_gb,
            'actual_epochs': self.actual_epochs,
        }
    
    # =====================================================================
    # History Methods
    # =====================================================================
    
    def get_loss_history(self) -> Dict[str, np.ndarray]:
        """
        Get loss history during training.
        
        Returns
        -------
        history : dict
            Dictionary containing:
            - 'total': Total loss per update
            - 'train': Training loss per epoch
            - 'val': Validation loss per epoch
        """
        if len(self.loss) == 0:
            return {}
        
        loss_array = np.array(self.loss)
        return {
            'total': loss_array[:, 0],
            'train': np.array(self.train_losses),
            'val': np.array(self.val_losses),
        }
    
    def get_metrics_history(self) -> Dict[str, np.ndarray]:
        """
        Get validation metrics history during training.
        
        Includes clustering quality metrics: ARI, NMI, ASW, CH, DB, Correlation.
        
        Returns
        -------
        history : dict
            Dictionary with arrays for each metric
        """
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