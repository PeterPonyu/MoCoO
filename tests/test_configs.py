"""Tests for mocoo.configs subpackage."""
import pytest
from mocoo.configs import load_config, get_shared_params, get_model_configs


class TestConfigLoading:
    def test_load_default(self):
        cfg = load_config("default")
        assert isinstance(cfg, dict)
        assert "shared" in cfg
        assert "training" in cfg

    def test_load_beta_sweep(self):
        cfg = load_config("beta_sweep")
        assert isinstance(cfg, dict)

    def test_get_shared_params(self):
        cfg = load_config("default")
        shared = get_shared_params(cfg)
        assert isinstance(shared, dict)
        assert "latent_dim" in shared
        assert shared["latent_dim"] == 32

    def test_get_model_configs(self):
        cfg = load_config("default")
        configs = get_model_configs(cfg)
        assert isinstance(configs, dict)
        assert len(configs) == 6
        assert "VAE" in configs
        assert "Full" in configs

    def test_model_configs_have_required_keys(self):
        cfg = load_config("default")
        configs = get_model_configs(cfg)
        for name, params in configs.items():
            # Each config should have use_ode, use_moco, use_prototype flags
            assert "use_ode" in params, f"{name} missing 'use_ode'"
            assert "use_moco" in params, f"{name} missing 'use_moco'"
            assert "use_prototype" in params, f"{name} missing 'use_prototype'"
            # The configs should be dicts
            assert isinstance(params, dict)

    def test_invalid_config_name_raises(self):
        with pytest.raises((FileNotFoundError, ValueError, KeyError)):
            load_config("nonexistent_config_name_xyz")

    def test_full_config_has_all_components(self):
        cfg = load_config("default")
        configs = get_model_configs(cfg)
        full = configs["Full"]
        assert full["use_ode"] is True
        assert full["use_moco"] is True
        assert full["use_prototype"] is True

    def test_vae_config_is_minimal(self):
        cfg = load_config("default")
        configs = get_model_configs(cfg)
        vae = configs["VAE"]
        assert vae["use_ode"] is False
        assert vae["use_moco"] is False
        assert vae["use_prototype"] is False

    def test_moco_weight_resolution(self):
        cfg = load_config("default")
        configs = get_model_configs(cfg)
        # VAE+MoCo (no ODE) should get moco_weight_without_ode = 0.5
        vae_moco = configs["VAE+MoCo"]
        assert vae_moco["moco_weight"] == 0.5
        # Full (with ODE) should get moco_weight_with_ode = 0.3
        full = configs["Full"]
        assert full["moco_weight"] == 0.3

    def test_get_training_params(self):
        from mocoo.configs import get_training_params
        cfg = load_config("default")
        training = get_training_params(cfg)
        assert isinstance(training, dict)
        assert "epochs" in training
        assert "patience" in training

    def test_get_sweep_params_default(self):
        from mocoo.configs import get_sweep_params
        cfg = load_config("default")
        sweep = get_sweep_params(cfg)
        # Default config has no sweep section
        assert sweep is None

    def test_get_sweep_params_beta_sweep(self):
        from mocoo.configs import get_sweep_params
        cfg = load_config("beta_sweep")
        sweep = get_sweep_params(cfg)
        assert sweep is not None
        assert "parameter" in sweep
        assert sweep["parameter"] == "beta"
        assert "values" in sweep

    def test_get_dataset_paths(self):
        from mocoo.configs import get_dataset_paths
        cfg = load_config("default")
        datasets = get_dataset_paths(cfg, base_dir="/tmp/data")
        assert isinstance(datasets, dict)
        assert "IRALL" in datasets
        assert "dentate" in datasets
        assert datasets["IRALL"]["path"].startswith("/tmp/data")

    def test_beta_sweep_has_fewer_configs(self):
        cfg = load_config("beta_sweep")
        configs = get_model_configs(cfg)
        assert isinstance(configs, dict)
        assert len(configs) == 2
        assert "VAE" in configs
        assert "Full" in configs
