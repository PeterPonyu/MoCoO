"""MoCoO experiment configuration module.

Provides centralized, version-controlled experiment configurations for
the MoCoO ablation study.  All hyperparameters are defined once in YAML
files (``mocoo/configs/default.yaml``, ``mocoo/configs/beta_ablation.yaml``)
and loaded through a Python API.

Quick start
-----------
>>> from mocoo.configs import load_config, get_shared_params, get_model_configs
>>> cfg = load_config("default")
>>> shared = get_shared_params(cfg)
>>> models = get_model_configs(cfg)
>>> full_params = {**shared, **models["Full"]}

See Also
--------
mocoo.configs.loader : Full API documentation.
mocoo/configs/default.yaml : Canonical experiment configuration.
mocoo/configs/beta_ablation.yaml : Beta ablation study configuration.
"""

from .loader import (
    load_config,
    get_shared_params,
    get_model_configs,
    get_training_params,
    get_moco_params,
    get_loss_weights,
    get_dataset_paths,
    get_sweep_params,
)

__all__ = [
    "load_config",
    "get_shared_params",
    "get_model_configs",
    "get_training_params",
    "get_moco_params",
    "get_loss_weights",
    "get_dataset_paths",
    "get_sweep_params",
]
