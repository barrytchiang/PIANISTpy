"""Every third-party package the package imports is a declared dependency."""

import ast
import pathlib
import sys

import pytest

tomllib = pytest.importorskip("tomllib")  # Python 3.11 and later

ROOT = pathlib.Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "pianistpy"
# Distributions whose import name differs from the dependency's name.
IMPORT_OF = {"jax": {"jax", "jaxlib"}}


def _third_party_imports() -> set[str]:
    found = set()
    for path in PACKAGE.rglob("*.py"):
        if "tests" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top != "pianistpy" and top not in sys.stdlib_module_names:
                    found.add(top)
    return found


def test_every_third_party_import_is_a_declared_dependency():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = set()
    for requirement in project["project"]["dependencies"]:
        name = (
            requirement.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip()
        )
        declared |= IMPORT_OF.get(name, {name})
    imported = _third_party_imports()
    assert imported, "the package imports nothing third party?"
    assert imported <= declared, sorted(imported - declared)
