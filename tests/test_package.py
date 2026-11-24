"""Smoke tests for the repo skeleton.

They check three things:
1. The package and every planned subpackage can be imported.
2. The version in the code matches the version in pyproject.toml.
3. The .gitignore keeps dataset files out of Git (VoD license: no data on GitHub).
"""

import importlib
import tomllib
from pathlib import Path

import pytest

import senseguard

REPO_ROOT = Path(__file__).resolve().parents[1]

# Must match the layout in the project brief, section 9.
EXPECTED_SUBPACKAGES = {
    "data",
    "geometry",
    "faults",
    "perception",
    "health",
    "safety",
    "eval",
}


def test_subpackage_list_matches_brief():
    assert set(senseguard.SUBPACKAGES) == EXPECTED_SUBPACKAGES


@pytest.mark.parametrize("name", sorted(EXPECTED_SUBPACKAGES))
def test_subpackage_imports(name):
    module = importlib.import_module(f"senseguard.{name}")
    assert module.__doc__, f"senseguard.{name} needs a docstring saying what goes in it"


def test_version_matches_pyproject():
    with open(REPO_ROOT / "pyproject.toml", "rb") as f:
        pyproject = tomllib.load(f)
    assert senseguard.__version__ == pyproject["project"]["version"]


@pytest.mark.parametrize(
    "pattern",
    ["datasets/", "*.bin", "*.pcd", "*.db3", "*.pt"],
)
def test_gitignore_blocks_dataset_and_weight_files(pattern):
    lines = (REPO_ROOT / ".gitignore").read_text().splitlines()
    assert pattern in lines, f".gitignore must contain '{pattern}'"
