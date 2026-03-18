"""Uncertainty quantification via posterior sampling."""

import numpy as np
import torch
from typing import Dict


def posterior_uncertainty(
    encoder: torch.nn.Module,
    state: np.ndarray,
    n_samples: int = 50,
    device: torch.device = None,
    batch_size: int = 512,
) -> Dict[str, np.ndarray]:
    """Sample z from q(z|x) multiple times to quantify per-cell uncertainty.

    Each forward pass through the encoder draws a fresh reparametrised
    sample z = mu + softplus(sigma) * eps. Repeating this gives an
    empirical estimate of posterior spread.

    Parameters
    ----------
    encoder : nn.Module
        MoCoO encoder (returns (q_z, q_m, q_s[, t])).
    state : np.ndarray, shape (N, G)
        Log1p-normalised expression matrix.
    n_samples : int
        Number of posterior draws.
    device : torch.device
    batch_size : int

    Returns
    -------
    dict
        - ``latent_std`` : (N, D) per-cell per-dim std across samples.
        - ``uncertainty`` : (N,) mean std across latent dims.
    """
    if device is None:
        device = torch.device('cpu')

    encoder.eval()
    N = state.shape[0]

    all_samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            parts = []
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                x = torch.tensor(state[start:end], dtype=torch.float32, device=device)
                out = encoder(x)
                parts.append(out[0].cpu().numpy())   # q_z (reparametrised)
            all_samples.append(np.concatenate(parts, axis=0))

    samples = np.stack(all_samples, axis=1)          # (N, n_samples, D)
    latent_std = samples.std(axis=1)                  # (N, D)
    uncertainty = latent_std.mean(axis=1)              # (N,)

    return {'latent_std': latent_std, 'uncertainty': uncertainty}
