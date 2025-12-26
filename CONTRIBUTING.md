# Contributing to MoCoO

Thank you for your interest in contributing to MoCoO! This document provides guidelines for contributors.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PeterPonyu/MoCoO.git
   cd MoCoO
   ```

2. **Install in development mode:**
   ```bash
   pip install -e ".[dev]"
   ```

3. **Run tests:**
   ```bash
   pytest tests/
   ```

## Code Style

This project uses:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Format code before committing:
```bash
black mocoo/ tests/
isort mocoo/ tests/
flake8 mocoo/ tests/
mypy mocoo/
```

## Testing

- Write tests for new features in `tests/`
- Ensure all tests pass: `pytest tests/ -v`
- Test on multiple Python versions when possible

## Pull Request Process

1. **Fork** the repository
2. **Create a feature branch:** `git checkout -b feature/your-feature`
3. **Make your changes** following the code style
4. **Add tests** for new functionality
5. **Ensure tests pass:** `pytest tests/`
6. **Update documentation** if needed
7. **Commit your changes:** `git commit -m "Add your feature"`
8. **Push to your fork:** `git push origin feature/your-feature`
9. **Create a Pull Request** with a clear description

## Release Process

Releases are automated via GitHub Actions:

1. **Version bumping** is done via `python release.py [patch|minor|major]`
2. **GitHub releases** trigger automatic PyPI publishing
3. **Semantic versioning** is followed

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Provide a clear description and minimal reproducible example
- Include your environment details (Python version, OS, etc.)

## License

By contributing to MoCoO, you agree that your contributions will be licensed under the MIT License.