"""Tests for mocoo.visualization subpackage."""
import numpy as np
import pytest

# Matplotlib backend must be set before any import
import matplotlib
matplotlib.use("Agg")

from mocoo.visualization.style import (
    apply_style,
    get_base_config_order,
    get_config_colors,
    get_config_order,
    get_display_name,
)
from mocoo.visualization.plots import (
    plot_ablation_radar,
    plot_metric_bars,
    plot_umap_grid,
    plot_training_curves,
    plot_pseudotime_markers,
    plot_beta_sensitivity,
)
from mocoo.visualization.pipeline import FigurePipeline
import matplotlib.pyplot as plt


class TestStyle:
    def test_apply_style_idempotent(self):
        apply_style()
        apply_style()  # Should not error on second call

    def test_config_colors_has_all_configs(self):
        colors = get_config_colors()
        assert "VAE" in colors
        assert "Full" in colors
        assert len(colors) == 12
        assert "Full+FM" in colors
        assert "VAE+FM" in colors

    def test_config_order_is_list(self):
        order = get_config_order()
        assert isinstance(order, list)
        assert len(order) == 12

    def test_base_config_order(self):
        base = get_base_config_order()
        assert isinstance(base, list)
        assert len(base) == 6
        for c in base:
            assert "+FM" not in c

    def test_config_display_maps(self):
        assert get_display_name("VAE") == "VAE"
        assert get_display_name("Full") == "MoCoO"
        assert get_display_name("Full+FM") == "MoCoO+FM"
        assert get_display_name("VAE_ODE") == "VAE+ODE"


class TestPlots:
    def setup_method(self):
        np.random.seed(42)
        apply_style()

    def test_ablation_radar_basic(self):
        data = {
            "VAE": {"ARI": 0.3, "NMI": 0.4, "ASW": 0.05},
            "Full": {"ARI": 0.4, "NMI": 0.5, "ASW": 0.08},
        }
        fig = plot_ablation_radar(data, metrics=["ARI", "NMI", "ASW"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ablation_radar_empty_data(self):
        fig = plot_ablation_radar({})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_metric_bars_basic(self):
        data = {
            "VAE": {"ARI": 0.3, "NMI": 0.4},
            "Full": {"ARI": 0.4, "NMI": 0.5},
        }
        fig = plot_metric_bars(data, metric_names=["ARI", "NMI"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_umap_grid_basic(self):
        latents = {
            "VAE": np.random.randn(100, 2),
            "Full": np.random.randn(100, 2),
        }
        labels = np.random.randint(0, 3, 100)
        fig = plot_umap_grid(latents, labels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_umap_grid_single_config(self):
        latents = {
            "VAE": np.random.randn(50, 2),
        }
        labels = np.random.randint(0, 3, 50)
        fig = plot_umap_grid(latents, labels)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_training_curves_basic(self):
        data = {
            "VAE": {"train": [1.0, 0.8, 0.6], "val": [1.1, 0.9, 0.7]},
        }
        fig = plot_training_curves(data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_pseudotime_markers_basic(self):
        data = {
            "GeneA": {
                "pseudotime": np.linspace(0, 1, 50),
                "expression": np.random.randn(50),
                "correlation": 0.8,
            },
        }
        fig = plot_pseudotime_markers(data)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_pseudotime_markers_empty(self):
        fig = plot_pseudotime_markers({})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_beta_sensitivity_basic(self):
        data = {
            "VAE": {"beta": [1.0, 0.1, 0.01], "ARI": [0.2, 0.3, 0.4]},
            "Full": {"beta": [1.0, 0.1, 0.01], "ARI": [0.2, 0.28, 0.3]},
        }
        fig = plot_beta_sensitivity(data, metric_name="ARI")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_beta_sensitivity_single_config(self):
        data = {
            "VAE": {"beta": [0.01, 0.1, 1.0], "NMI": [0.5, 0.4, 0.3]},
        }
        fig = plot_beta_sensitivity(data, metric_name="NMI")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestFigurePipeline:
    def test_instantiation(self, tmp_path):
        pipe = FigurePipeline(str(tmp_path), str(tmp_path / "output"))
        assert pipe is not None

    def test_available_figures_empty(self, tmp_path):
        pipe = FigurePipeline(str(tmp_path), str(tmp_path / "output"))
        pipe.load_results()
        available = pipe.available_figures
        assert isinstance(available, (list, set, tuple))

    def test_repr(self, tmp_path):
        pipe = FigurePipeline(str(tmp_path), str(tmp_path / "output"))
        r = repr(pipe)
        assert "FigurePipeline" in r
        assert "not loaded" in r

    def test_loaded_state(self, tmp_path):
        pipe = FigurePipeline(str(tmp_path), str(tmp_path / "output"))
        pipe.load_results()
        r = repr(pipe)
        assert "loaded" in r

    def test_configs_property(self, tmp_path):
        pipe = FigurePipeline(str(tmp_path), str(tmp_path / "output"))
        pipe.load_results()
        configs = pipe.configs
        assert isinstance(configs, list)
