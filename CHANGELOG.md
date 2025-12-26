# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of MoCoO package
- VAE with multiple likelihood modes (MSE, NB, ZINB, Poisson, ZIP)
- Neural ODE for trajectory modeling
- Momentum Contrast (MoCo) for representation learning
- Information bottleneck regularization
- Disentanglement losses (DIP-VAE, β-TC-VAE, InfoVAE)
- Vector field analysis for RNA velocity
- Comprehensive test suite
- GitHub Actions CI/CD pipeline
- Automated PyPI publishing

### Changed
- Restructured codebase into proper Python package
- Updated to modern packaging standards (pyproject.toml)
- Improved documentation and examples

### Fixed
- Batch size mismatch in validation
- ODE path handling in latent extraction
- Memory efficiency improvements

## [0.0.1] - 2025-12-26

### Added
- Initial implementation of MoCoO framework
- Core VAE, ODE, and MoCo components
- Basic training and inference functionality
- Package structure and configuration files