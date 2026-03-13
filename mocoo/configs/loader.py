"""Configuration loader for MoCoO experiments.

Provides functions to load YAML experiment configs and extract structured
parameter dictionaries for model training.  Falls back to a pure-Python
dict representation when PyYAML is not installed.

Public API
----------
load_config(name="default") -> dict
    Load a named YAML config from the ``mocoo/configs/`` directory.

get_shared_params(config) -> dict
    Extract the shared hyperparameters from a loaded config.

get_model_configs(config) -> dict[str, dict]
    Build the 6 ablation configurations, each ready to be unpacked as
    ``MoCoO(adata, **shared | config_params)``.  Automatically resolves
    the conditional moco_weight (1.0 for both ODE and non-ODE configs).

get_training_params(config) -> dict
    Extract training schedule parameters (epochs, patience, val_every).

get_dataset_paths(config) -> dict[str, dict]
    Resolve dataset paths using the MOCOO_DATA_DIR environment variable.

Examples
--------
>>> from mocoo.configs.loader import load_config, get_shared_params, get_model_configs
>>> cfg = load_config("default")
>>> shared = get_shared_params(cfg)
>>> models = get_model_configs(cfg)
>>> for name, params in models.items():
...     full_params = {**shared, **params}
...     model = MoCoO(adata, **full_params)
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# YAML loading (with pure-Python fallback)
# ---------------------------------------------------------------------------

_CONFIGS_DIR = Path(__file__).resolve().parent

try:
    import yaml

    def _load_yaml(path: Path) -> dict:
        """Load a YAML file using PyYAML."""
        with open(path, "r") as fh:
            return yaml.safe_load(fh)

except ImportError:
    yaml = None  # type: ignore[assignment]

    def _load_yaml(path: Path) -> dict:
        """Fallback YAML loader -- only works for the built-in configs
        that also have a pure-Python representation via _BUILTIN_CONFIGS."""
        raise ImportError(
            "PyYAML is not installed.  Install it with:\n"
            "    pip install pyyaml\n"
            "Or use load_config() with a built-in config name to use the "
            "pure-Python fallback."
        )


# ---------------------------------------------------------------------------
# Built-in fallback configs (used when PyYAML is unavailable)
# ---------------------------------------------------------------------------
# These mirror the canonical values in default.yaml exactly.

_BUILTIN_SHARED = dict(
    latent_dim=10,
    hidden_dim=128,
    i_dim=2,
    lr=1e-4,
    batch_size=128,
    beta=1.0,
    recon=1.0,
    loss_mode="nb",
    random_seed=42,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
)

_BUILTIN_MOCO = dict(
    moco_K=4096,
    moco_T=0.2,
    moco_m=0.999,
    n_prototypes=12,
)

_BUILTIN_LOSS_WEIGHTS = dict(
    vae_reg=0.5,
    ode_reg=0.5,
    moco_weight_with_ode=1.0,
    moco_weight_without_ode=1.0,
    proto_weight=1.0,
)

_BUILTIN_TRAINING = dict(
    epochs=400,
    patience=60,
    val_every=5,
)

_BUILTIN_DATASETS = {
    "IRALL": dict(
        path="LAB/scRL/IRALL.h5ad",
        cell_type_col="cell_type",
        batch_col="batch",
        description="Mouse haematopoiesis time-series (d0-d30)",
        max_cells=3000,
        hvg=3000,
    ),
    "dentate": dict(
        path="vGAE_LAB/data/dentate.h5ad",
        cell_type_col="Clusters",
        batch_col=None,
        description="Mouse dentate gyrus neurogenesis",
        max_cells=3000,
        hvg=3000,
    ),
    "endo": dict(
        path="vGAE_LAB/data/endo.h5ad",
        cell_type_col="clusters_fine",
        batch_col="day",
        description="Mouse endocrine pancreas development",
        max_cells=2500,
        hvg=3000,
    ),
    "paul": dict(
        path="LAB/data/paul.h5ad",
        cell_type_col="paul15_clusters",
        batch_col=None,
        description="Mouse myeloid/erythroid progenitor differentiation",
        max_cells=2700,
        hvg=3000,
    ),
    "spinoids": dict(
        path="LAB/data/spinoids.h5ad",
        cell_type_col="annotation",
        batch_col=None,
        description="Human spinal cord organoid development",
        max_cells=3000,
        hvg=3000,
    ),
}

_BUILTIN_CONFIGS = {
    "VAE": dict(
        use_ode=False, use_moco=False, use_prototype=False,
    ),
    "VAE+ODE": dict(
        use_ode=True, use_moco=False, use_prototype=False,
        vae_reg=0.5, ode_reg=0.5,
    ),
    "VAE+MoCo": dict(
        use_ode=False, use_moco=True, use_prototype=False,
        moco_weight=1.0, moco_T=0.2, moco_K=4096,
    ),
    "VAE+MoCo+Proto": dict(
        use_ode=False, use_moco=True, use_prototype=True,
        moco_weight=1.0, moco_T=0.2, moco_K=4096,
        n_prototypes=12, proto_weight=1.0,
    ),
    "VAE+ODE+MoCo": dict(
        use_ode=True, use_moco=True, use_prototype=False,
        vae_reg=0.5, ode_reg=0.5,
        moco_weight=1.0, moco_T=0.2, moco_K=4096,
    ),
    "Full": dict(
        use_ode=True, use_moco=True, use_prototype=True,
        vae_reg=0.5, ode_reg=0.5,
        moco_weight=1.0, moco_T=0.2, moco_K=4096,
        n_prototypes=12, proto_weight=1.0,
    ),
}


def _builtin_default() -> dict:
    """Return the built-in default config as a plain dict."""
    return {
        "shared": copy.deepcopy(_BUILTIN_SHARED),
        "moco": copy.deepcopy(_BUILTIN_MOCO),
        "loss_weights": copy.deepcopy(_BUILTIN_LOSS_WEIGHTS),
        "training": copy.deepcopy(_BUILTIN_TRAINING),
        "datasets": copy.deepcopy(_BUILTIN_DATASETS),
        "configs": copy.deepcopy(_BUILTIN_CONFIGS),
    }


def _builtin_beta_ablation() -> dict:
    """Return the built-in beta ablation config as a plain dict.

    Proper experimental settings for the beta ablation study (Tables I-V):
    - 400 epochs (sufficient for Full model convergence)
    - patience 60
    - All 6 model configurations
    """
    return {
        "shared": {k: v for k, v in _BUILTIN_SHARED.items() if k != "beta"},
        "training": dict(epochs=400, patience=60, val_every=5),
        "sweep": dict(parameter="beta", values=[0.01, 0.1, 1.0]),
        "loss_weights": copy.deepcopy(_BUILTIN_LOSS_WEIGHTS),
        "configs": copy.deepcopy(_BUILTIN_CONFIGS),
    }


_BUILTIN_REGISTRY: Dict[str, Any] = {
    "default": _builtin_default,
    "beta_ablation": _builtin_beta_ablation,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(name: str = "default", path: Optional[str] = None) -> dict:
    """Load a named experiment configuration.

    Parameters
    ----------
    name : str
        Config name.  Looks for ``<name>.yaml`` in the package configs
        directory.  Built-in names: ``"default"``, ``"beta_ablation"``.
    path : str, optional
        Explicit path to a YAML file.  Overrides *name* when provided.

    Returns
    -------
    dict
        The loaded configuration dictionary.

    Notes
    -----
    If PyYAML is not installed and *name* matches a built-in config,
    a pure-Python fallback is used.  For custom YAML files, PyYAML is
    required.
    """
    if path is not None:
        return _load_yaml(Path(path))

    yaml_path = _CONFIGS_DIR / f"{name}.yaml"

    # Try YAML first
    if yaml_path.exists() and yaml is not None:
        return _load_yaml(yaml_path)

    # Fall back to built-in Python dicts
    if name in _BUILTIN_REGISTRY:
        return _BUILTIN_REGISTRY[name]()

    # YAML file exists but no PyYAML
    if yaml_path.exists():
        raise ImportError(
            f"Config file '{yaml_path}' found but PyYAML is not installed.\n"
            f"Install it with: pip install pyyaml"
        )

    available = sorted(
        {p.stem for p in _CONFIGS_DIR.glob("*.yaml")} | set(_BUILTIN_REGISTRY)
    )
    raise FileNotFoundError(
        f"Unknown config '{name}'. Available configs: {available}"
    )


def get_shared_params(config: dict) -> dict:
    """Extract shared hyperparameters from a loaded config.

    Parameters
    ----------
    config : dict
        A config dict as returned by :func:`load_config`.

    Returns
    -------
    dict
        Shared parameters suitable for passing to ``MoCoO(adata, **params)``.
    """
    return copy.deepcopy(config.get("shared", {}))


def get_training_params(config: dict) -> dict:
    """Extract training schedule parameters.

    Parameters
    ----------
    config : dict
        A config dict as returned by :func:`load_config`.

    Returns
    -------
    dict
        Dictionary with keys ``epochs``, ``patience``, ``val_every``.
    """
    return copy.deepcopy(config.get("training", {}))


def get_moco_params(config: dict) -> dict:
    """Extract MoCo contrastive learning parameters.

    Parameters
    ----------
    config : dict
        A config dict as returned by :func:`load_config`.

    Returns
    -------
    dict
        Dictionary with keys ``moco_K``, ``moco_T``, ``moco_m``,
        ``n_prototypes``.
    """
    return copy.deepcopy(config.get("moco", {}))


def get_loss_weights(config: dict) -> dict:
    """Extract the loss weight definitions.

    Parameters
    ----------
    config : dict
        A config dict as returned by :func:`load_config`.

    Returns
    -------
    dict
        Dictionary with keys ``vae_reg``, ``ode_reg``,
        ``moco_weight_with_ode``, ``moco_weight_without_ode``,
        ``proto_weight``.
    """
    return copy.deepcopy(config.get("loss_weights", {}))


def get_model_configs(config: dict) -> Dict[str, dict]:
    """Build per-model configuration dicts from a loaded config.

    For each of the ablation configurations defined in ``config["configs"]``,
    returns a dict of model-specific parameters that can be merged with
    the shared params via ``{**shared, **model_cfg}``.

        The conditional ``moco_weight`` logic is handled automatically:
        - If a config has ``use_ode=True`` and ``use_moco=True`` but no
            explicit ``moco_weight``, it gets ``moco_weight_with_ode`` (1.0).
        - If a config has ``use_ode=False`` and ``use_moco=True`` but no
            explicit ``moco_weight``, it gets ``moco_weight_without_ode`` (1.0).

    Parameters
    ----------
    config : dict
        A config dict as returned by :func:`load_config`.

    Returns
    -------
    dict[str, dict]
        Mapping from config name (e.g. ``"Full"``) to its parameter dict.
    """
    raw_configs = config.get("configs", {})
    loss_weights = config.get("loss_weights", {})
    result = {}

    for name, cfg in raw_configs.items():
        resolved = copy.deepcopy(cfg)

        # Resolve conditional moco_weight if not explicitly set
        uses_moco = resolved.get("use_moco", False)
        uses_ode = resolved.get("use_ode", False)

        if uses_moco and "moco_weight" not in resolved:
            if uses_ode:
                resolved["moco_weight"] = loss_weights.get(
                    "moco_weight_with_ode", 1.0
                )
            else:
                resolved["moco_weight"] = loss_weights.get(
                    "moco_weight_without_ode", 1.0
                )

        result[name] = resolved

    return result


def get_dataset_paths(
    config: dict,
    base_dir: Optional[str] = None,
) -> Dict[str, dict]:
    """Resolve dataset paths using MOCOO_DATA_DIR.

    Parameters
    ----------
    config : dict
        A config dict as returned by :func:`load_config`.
    base_dir : str, optional
        Override for the base data directory.  If not provided, reads
        the ``MOCOO_DATA_DIR`` environment variable (falling back to
        ``data`` if unset, matching the dataset_registry default).

    Returns
    -------
    dict[str, dict]
        Mapping from dataset name to a dict with ``path`` (absolute),
        ``cell_type_col``, ``batch_col``, ``max_cells``, ``hvg``, and
        ``description``.
    """
    if base_dir is None:
        base_dir = os.environ.get("MOCOO_DATA_DIR", "data")

    datasets = config.get("datasets", {})
    resolved = {}

    for name, spec in datasets.items():
        entry = copy.deepcopy(spec)
        rel_path = entry.get("path", "")
        entry["path"] = os.path.join(base_dir, rel_path)
        resolved[name] = entry

    return resolved


def get_sweep_params(config: dict) -> Optional[dict]:
    """Extract sweep parameters if present (e.g. for beta_ablation.yaml).

    Parameters
    ----------
    config : dict
        A config dict as returned by :func:`load_config`.

    Returns
    -------
    dict or None
        Dictionary with ``parameter`` (str) and ``values`` (list) keys,
        or ``None`` if the config has no sweep section.
    """
    return copy.deepcopy(config.get("sweep", None))
