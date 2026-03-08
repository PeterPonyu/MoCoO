## Repository Map

- Package code lives in `mocoo/` and exposes the main Python library implementation.
- Tests live in `tests/` and are configured through `setup.cfg` with `pytest` discovery under `tests`.
- Benchmark automation and plotting live under `benchmarks/scripts/`.
- Benchmark result artifacts live under `benchmarks/results/`.
- Top-level documentation includes `README.md`, `docs/PIPELINE.md`, and manuscript-style notes such as `MoCoO_JBHI_Article.md`.
- Existing chat customizations live under `.github/prompts/`.
- The repository now also supports a workspace custom agent under `.github/agents/`.
- Root `AGENTS.md` now defines a tool-agnostic continuity contract for Claude and other agents.
- `.github/copilot-instructions.md` bridges Copilot's automatic repo instructions back to `AGENTS.md`.
- `scripts/install_agent_bundle.sh` installs the portable agent workflow into other repositories.

## Build And Test Signals

- Python package metadata is defined in `pyproject.toml`.
- Development extras include `pytest`, `pytest-cov`, `black`, `isort`, `flake8`, `mypy`, `jupyter`, and `ipykernel`.
- Common validation path for code changes is `pytest`.

## Environment Expectations

- Python 3.8+ is required by package metadata.
- The project depends on PyTorch, torchdiffeq, AnnData, Scanpy, NumPy, SciPy, scikit-learn, tqdm, and pandas.

## Architectural Notes

- The repository mixes package code, benchmark pipelines, analysis scripts, plotting utilities, and paper-supporting artifacts.
- Custom prompt behavior was already documented in `.github/prompts/`; the new agent extends that with reusable agent selection support.

## Recurring Pitfalls

- Open-ended tasks may touch code, docs, and benchmark artifacts at the same time, so validation needs to match the artifact type rather than defaulting to code-only assumptions.
- VS Code prompt files are helpful workflow references, but cross-agent continuity should live in repository Markdown under `agent-context/`.
- The intended discovery path is now `AGENTS.md` first, then `agent-context/`, then any editor-specific helper files.
