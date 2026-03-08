"""
High-level pipeline that generates all MoCoO paper figures from benchmark
results stored on disk.

The :class:`FigurePipeline` scans a *results_dir* for the CSV / JSON / NPZ
files produced by the benchmark suite, loads them into memory, and exposes
:meth:`generate_all` and :meth:`generate_figure` methods that delegate to
the core plotting functions in :mod:`mocoo.visualization.plots`.

Usage
-----
::

    from mocoo.visualization.pipeline import FigurePipeline

    pipe = FigurePipeline("benchmarks/results/IRALL", "figures/")
    pipe.load_results()
    pipe.generate_all()           # all six figure groups
    pipe.generate_figure("ablation")  # just the ablation panel
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore[assignment]

from . import style as _style
from . import plots as _plots

# ---------------------------------------------------------------------------
# Metric key normalization (mirrors the helper in the standalone scripts)
# ---------------------------------------------------------------------------
_METRIC_KEY_MAP = {
    "full_ARI": "ARI", "full_NMI": "NMI", "full_ASW": "ASW",
    "full_CH": "CAL", "full_DB": "DAV", "corr": "COR",
    "CH": "CAL", "DB": "DAV",
    "LSE_overall": "LSE_overall_quality",
    "DRE_UMAP_overall": "DRE_umap_overall_quality",
    "DRE_tSNE_overall": "DRE_tsne_overall_quality",
}


def _unify_metric_keys(m: dict) -> dict:
    """Normalise raw JSON metric keys to short canonical names."""
    for src, dst in _METRIC_KEY_MAP.items():
        if src in m and dst not in m:
            m[dst] = m[src]
    return m


# All recognised figure names
FIGURE_NAMES: List[str] = [
    "ablation",
    "comparison",
    "dynamics",
    "batch",
    "trajectory",
    "biovalidation",
]


class FigurePipeline:
    """End-to-end generator for MoCoO paper figures.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing benchmark output files (``benchmark_data.npz``,
        per-config JSON metric files, optional CSVs).
    output_dir : str or Path
        Directory where generated figures will be saved.
    datasets : sequence of str, optional
        If the results directory contains sub-directories per dataset
        (e.g. ``IRALL/``, ``dentate/``, ``endo/``), specify which to
        process.  When *None*, the pipeline scans for any NPZ / JSON
        files directly in *results_dir*.
    """

    def __init__(
        self,
        results_dir: str | Path,
        output_dir: str | Path,
        datasets: Optional[Sequence[str]] = None,
    ) -> None:
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.datasets = list(datasets) if datasets is not None else None

        # Populated by load_results()
        self._configs: List[str] = []
        self._metrics: Dict[str, dict] = {}
        self._latents: Dict[str, np.ndarray] = {}
        self._labels: Dict[str, np.ndarray] = {}
        self._train_losses: Dict[str, np.ndarray] = {}
        self._val_losses: Dict[str, np.ndarray] = {}
        self._val_scores: Dict[str, np.ndarray] = {}
        self._extra: Dict[str, Any] = {}  # catch-all for CSVs etc.
        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_results(self) -> "FigurePipeline":
        """Scan *results_dir* and load all benchmark artefacts.

        Populates internal state so that :meth:`generate_all` /
        :meth:`generate_figure` can produce figures without further I/O.

        Returns *self* for chaining.
        """
        _style.apply_style()
        dirs_to_scan: List[Path] = []

        if self.datasets is not None:
            for ds in self.datasets:
                d = self.results_dir / ds
                if d.is_dir():
                    dirs_to_scan.append(d)
                else:
                    warnings.warn(f"Dataset directory not found: {d}")
        else:
            dirs_to_scan.append(self.results_dir)

        for rdir in dirs_to_scan:
            self._load_from_dir(rdir)

        self._loaded = True
        return self

    def _load_from_dir(self, rdir: Path) -> None:
        """Load benchmark artefacts from a single directory."""
        # 1. NPZ bundle (latents, labels, losses, val_scores)
        npz_path = rdir / "benchmark_data.npz"
        if npz_path.exists():
            npz = np.load(npz_path, allow_pickle=True)
            configs = [str(c) for c in npz["configs"]]
            self._configs = configs

            # Latents & labels (from quant_comparison / ablation runs)
            if "latents" in npz:
                latents = list(npz["latents"])
                labels = list(npz["labels"])
                for i, cfg in enumerate(configs):
                    self._latents[cfg] = np.asarray(latents[i], dtype=np.float32)
                    self._labels[cfg] = np.asarray(labels[i])

            # Loss histories (from training_dynamics runs)
            if "train_losses" in npz:
                for i, cfg in enumerate(configs):
                    self._train_losses[cfg] = np.asarray(
                        npz["train_losses"][i], dtype=np.float32
                    )
            if "val_losses" in npz:
                for i, cfg in enumerate(configs):
                    self._val_losses[cfg] = np.asarray(
                        npz["val_losses"][i], dtype=np.float32
                    )
            if "val_scores" in npz:
                for i, cfg in enumerate(configs):
                    self._val_scores[cfg] = np.asarray(
                        npz["val_scores"][i], dtype=np.float32
                    )

        # 2. Per-config JSON metrics
        for jf in sorted(rdir.glob("*.json")):
            # Skip non-config JSON files
            if jf.stem in ('benchmark_data', 'summary', 'meta_analysis'):
                continue
            try:
                with open(jf) as f:
                    raw = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            # Infer config name from filename  (VAE_ODE.json -> VAE+ODE)
            cfg_key = jf.stem.replace("_", "+")
            raw = _unify_metric_keys(raw)
            self._metrics[cfg_key] = raw

        # 3. CSV files (sweep results, cross-dataset comparisons, etc.)
        if pd is not None:
            for csv in sorted(rdir.glob("*.csv")):
                try:
                    df = pd.read_csv(csv)
                    self._extra[csv.stem] = df
                except Exception:
                    continue

        # Use canonical config list if NPZ was not present
        if not self._configs:
            self._configs = _style.get_config_order()

    # ------------------------------------------------------------------
    # Figure generation
    # ------------------------------------------------------------------

    def generate_all(self) -> Dict[str, Any]:
        """Generate all paper figures and save to *output_dir*.

        Returns
        -------
        dict[str, Figure]
            Mapping of figure name to the matplotlib Figure object.
        """
        self._ensure_loaded()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        results = {}
        for name in FIGURE_NAMES:
            try:
                fig = self.generate_figure(name)
                if fig is not None:
                    results[name] = fig
            except Exception as exc:
                warnings.warn(f"Skipping figure '{name}': {exc}")
        return results

    def generate_figure(self, name: str) -> Any:
        """Generate a specific figure by name.

        Parameters
        ----------
        name : str
            One of: ``"ablation"``, ``"comparison"``, ``"dynamics"``,
            ``"batch"``, ``"trajectory"``, ``"biovalidation"``.

        Returns
        -------
        matplotlib.figure.Figure or None
            The generated figure, or *None* when required data is missing.
        """
        self._ensure_loaded()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        dispatch = {
            "ablation": self._fig_ablation,
            "comparison": self._fig_comparison,
            "dynamics": self._fig_dynamics,
            "batch": self._fig_batch,
            "trajectory": self._fig_trajectory,
            "biovalidation": self._fig_biovalidation,
        }
        handler = dispatch.get(name)
        if handler is None:
            raise ValueError(
                f"Unknown figure name '{name}'. "
                f"Choose from: {', '.join(FIGURE_NAMES)}"
            )
        return handler()

    # ------------------------------------------------------------------
    # Individual figure handlers
    # ------------------------------------------------------------------

    def _fig_ablation(self):
        """Figure 5 -- ablation study radar + metric bars."""
        if not self._metrics:
            warnings.warn("No metric data loaded; skipping ablation figure.")
            return None

        configs = [c for c in self._configs if c in self._metrics]
        outpath = self.output_dir / "fig_ablation_radar.png"
        fig = _plots.plot_ablation_radar(
            metrics_df=self._metrics,
            configs=configs,
            outpath=outpath,
        )
        return fig

    def _fig_comparison(self):
        """Figure 2 -- quantitative latent space comparison."""
        figs = {}

        # UMAP grid (needs latents)
        if self._latents:
            configs = [c for c in self._configs if c in self._latents]
            outpath = self.output_dir / "fig_comparison_umap.png"
            labels = {c: self._labels[c] for c in configs}
            figs["umap"] = _plots.plot_umap_grid(
                latents_dict=self._latents,
                labels=labels,
                configs=configs,
                outpath=outpath,
            )

        # Metric bars
        if self._metrics:
            configs = [c for c in self._configs if c in self._metrics]
            outpath = self.output_dir / "fig_comparison_bars.png"
            figs["bars"] = _plots.plot_metric_bars(
                metrics_df=self._metrics,
                metric_names=["ARI", "NMI", "ASW"],
                configs=configs,
                outpath=outpath,
            )

        if not figs:
            warnings.warn("No comparison data loaded; skipping.")
            return None

        # Return the first generated figure for convenience
        return next(iter(figs.values()))

    def _fig_dynamics(self):
        """Figure 4 -- training dynamics & convergence."""
        if not self._train_losses and not self._val_losses:
            warnings.warn("No loss history data loaded; skipping dynamics figure.")
            return None

        configs = [
            c for c in self._configs
            if c in self._train_losses or c in self._val_losses
        ]
        loss_histories = {}
        for cfg in configs:
            entry: Dict[str, Any] = {}
            if cfg in self._train_losses:
                entry["train"] = self._train_losses[cfg]
            if cfg in self._val_losses:
                entry["val"] = self._val_losses[cfg]
            if cfg in self._val_scores:
                entry["val_scores"] = self._val_scores[cfg]
            loss_histories[cfg] = entry

        outpath = self.output_dir / "fig_training_dynamics.png"
        return _plots.plot_training_curves(
            loss_histories=loss_histories,
            configs=configs,
            outpath=outpath,
        )

    def _fig_batch(self):
        """Figure 7 -- batch integration & cross-dataset generalization.

        Uses metric bars for iLISI / bASW / cLISI if those keys are present
        in the loaded metrics. Falls back to a standard metric bar chart.
        """
        if not self._metrics:
            warnings.warn("No metric data loaded; skipping batch figure.")
            return None

        batch_metrics = ["iLISI", "bASW", "cLISI"]
        # Check which batch metrics are actually available
        available = [
            m for m in batch_metrics
            if any(m in self._metrics.get(c, {}) for c in self._configs)
        ]
        if not available:
            # Fall back to standard clustering metrics
            available = ["ARI", "NMI", "ASW"]

        configs = [c for c in self._configs if c in self._metrics]
        outpath = self.output_dir / "fig_batch_integration.png"
        return _plots.plot_metric_bars(
            metrics_df=self._metrics,
            metric_names=available,
            configs=configs,
            outpath=outpath,
        )

    def _fig_trajectory(self):
        """Figure 6 -- ODE-driven pseudotime & trajectory analysis.

        If a ``pseudotime_correlations`` CSV was loaded, use it for the
        marker-gene plot. Otherwise, generate a beta-sensitivity plot from
        sweep data if available.
        """
        # Pseudotime markers
        pt_df = self._extra.get("pseudotime_correlations")
        if pt_df is not None and pd is not None:
            outpath = self.output_dir / "fig_trajectory_markers.png"
            return _plots.plot_pseudotime_markers(
                correlations_df=pt_df,
                outpath=outpath,
            )

        # Beta sweep fallback
        sweep_df = self._extra.get("beta_sweep")
        if sweep_df is not None:
            outpath = self.output_dir / "fig_trajectory_beta.png"
            return _plots.plot_beta_sensitivity(
                sweep_results=sweep_df,
                outpath=outpath,
            )

        warnings.warn(
            "No pseudotime or beta-sweep data found; skipping trajectory figure."
        )
        return None

    def _fig_biovalidation(self):
        """Figure 3 -- biological validation (gene-expression correlation).

        Uses a metric bar chart for bio-relevant metrics when full
        validation CSVs are not available.
        """
        bio_df = self._extra.get("biological_validation")
        if bio_df is not None and pd is not None:
            # If there is a dedicated bio-validation CSV with gene/pseudotime
            # columns, treat it as marker correlation data
            required_cols = {"gene", "pseudotime", "expression"}
            if required_cols.issubset(set(bio_df.columns)):
                outpath = self.output_dir / "fig_biovalidation_markers.png"
                return _plots.plot_pseudotime_markers(
                    correlations_df=bio_df,
                    outpath=outpath,
                )

        # Fall back to metric bars for bio-relevant metrics
        if self._metrics:
            bio_metrics = ["COR", "LSE_overall_quality", "DRE_umap_overall_quality"]
            available = [
                m for m in bio_metrics
                if any(m in self._metrics.get(c, {}) for c in self._configs)
            ]
            if not available:
                available = ["ARI", "NMI", "ASW"]
            configs = [c for c in self._configs if c in self._metrics]
            outpath = self.output_dir / "fig_biovalidation.png"
            return _plots.plot_metric_bars(
                metrics_df=self._metrics,
                metric_names=available,
                configs=configs,
                outpath=outpath,
            )

        warnings.warn("No bio-validation data found; skipping.")
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load_results()

    @property
    def configs(self) -> List[str]:
        """Currently loaded config names."""
        return list(self._configs)

    @property
    def available_figures(self) -> List[str]:
        """Figure names that can be generated given the loaded data."""
        available = []
        if self._metrics:
            available.extend(["ablation", "batch", "biovalidation"])
        if self._latents or self._metrics:
            available.append("comparison")
        if self._train_losses or self._val_losses:
            available.append("dynamics")
        if (
            "pseudotime_correlations" in self._extra
            or "beta_sweep" in self._extra
        ):
            available.append("trajectory")
        return sorted(set(available) & set(FIGURE_NAMES))

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return (
            f"FigurePipeline(results_dir={str(self.results_dir)!r}, "
            f"output_dir={str(self.output_dir)!r}, status={status}, "
            f"configs={self._configs})"
        )
