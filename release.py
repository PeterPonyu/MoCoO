#!/usr/bin/env python3
"""
Release script for MoCoO package.

Usage:
    python release.py patch  # 0.0.1 -> 0.0.2
    python release.py minor  # 0.0.1 -> 0.1.0
    python release.py major  # 0.0.1 -> 1.0.0
    python release.py 1.2.3   # Set specific version
"""

import re
import sys
from pathlib import Path

def get_current_version():
    """Get current version from pyproject.toml"""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()
    match = re.search(r'version = "([^"]+)"', content)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")
    return match.group(1)

def bump_version(current, bump_type):
    """Bump version according to semantic versioning"""
    major, minor, patch = map(int, current.split('.'))

    if bump_type == 'patch':
        patch += 1
    elif bump_type == 'minor':
        minor += 1
        patch = 0
    elif bump_type == 'major':
        major += 1
        minor = 0
        patch = 0
    else:
        raise ValueError(f"Unknown bump type: {bump_type}")

    return f"{major}.{minor}.{patch}"

def update_version(new_version):
    """Update version in pyproject.toml"""
    pyproject_path = Path("pyproject.toml")
    content = pyproject_path.read_text()
    updated = re.sub(r'version = "[^"]*"', f'version = "{new_version}"', content)
    pyproject_path.write_text(updated)
    print(f"✓ Updated version to {new_version} in pyproject.toml")

    # Update __init__.py
    init_path = Path("mocoo/__init__.py")
    content = init_path.read_text()
    updated = re.sub(r'__version__ = "[^"]*"', f'__version__ = "{new_version}"', content)
    init_path.write_text(updated)
    print(f"✓ Updated version to {new_version} in mocoo/__init__.py")

def main():
    if len(sys.argv) != 2 or sys.argv[1] in ['-h', '--help']:
        print(__doc__)
        sys.exit(1)

    arg = sys.argv[1]

    if arg in ['patch', 'minor', 'major']:
        current = get_current_version()
        new_version = bump_version(current, arg)
    else:
        # Assume it's a specific version
        new_version = arg

    print(f"Bumping version from {get_current_version()} to {new_version}")
    update_version(new_version)

    print("\nNext steps:")
    print("1. Commit changes: git add -A && git commit -m 'Bump version to", new_version + "'")
    print("2. Create tag: git tag v" + new_version)
    print("3. Push: git push && git push --tags")
    print("4. Create GitHub release to trigger PyPI publish")

if __name__ == "__main__":
    main()