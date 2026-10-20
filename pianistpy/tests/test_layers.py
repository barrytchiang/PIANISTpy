"""The three layers of the package, held by its imports.

1. Leaves: numerics, closed forms and input objects, with no state.
2. Solvers: the stages, which take settings and return arrays or result
   objects, with no file access.
3. Products and drivers: result files, refusals and the public calls.

A lower layer never imports a higher one, and neither a leaf nor a stage
touches a file. Every module is assigned to a layer here; a new module
fails the test until it is.
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parents[1]

# Modules every layer may use: they hold no physics and open no file.
BASE = {"exceptions", "typing", "validation", "identity", "_version", "runtime"}
# The session services (messages, timing, workers) serve every layer, and
# write a log file only where the user asks for one.
SERVICES = {"runtime"}

LEAVES = {
    "numerics",
    "profiles",
    "anisotropy",
    "units",
    "df.asymptotics",
    "df.central",
    "df.inversion",
    "df.edge_law",
    "df.edge",
    "df.inputs",
    "observables.aperture",
    "observables._angle_rule",
    "observables._panels",
}
SOLVERS = {
    "settings",
    "df.stages",
    "df.checks",
    "df.lookup",
    "df.truncation",
    "df.poisson",
    "df.center",
    "df.fill",
    "df.validate",
    "df.gates",
    "df.form",
    "df.grid",
    "df.report",
    "df.top",
    "df.rows",
    "df.derivatives",
    "df.pair",
    "df.profile_checks",
    "particles.stages",
    "particles.speed",
    "observables.stages",
    "observables._dispersion",
    "observables._projection",
    "observables._tables",
    "observables._aperture",
    "observables._ellipse_rules",
    "observables._angle",
    "observables._layout",
    "observables._ellipse_df",
    "observables._ellipse",
    "observables._kept",
    "observables._panel_table",
    "observables._losvd_table",
    "observables._aperture_losvd",
    "observables._ellipse_model",
    "observables._losvd_stages",
    "observables._speed",
    "observables._table_vjp",
    "observables._rows_theta",
}
DRIVERS = {
    "",
    "io",
    "io.core",
    "io.text",
    "df",
    "df.core",
    "df.results",
    "df.invert",
    "df.gradient",
    "df.truncated",
    "df.target",
    "df.tabulated",
    "df.analytic",
    "particles",
    "particles.core",
    "particles.writers",
    "observables",
    "observables.core",
    "observables.projected",
    "observables._differentiable",
    "superposition",
    "superposition.core",
    "inference",
    "inference.core",
    "inference._binned",
}

# What may not be imported below the third layer: file access.
FILE_MODULES = {"h5py", "os", "pathlib", "shutil", "io", "tempfile", "pickle"}


def _modules():
    """Return the dotted name of every module of the package, without tests."""
    names = []
    for path in sorted(PACKAGE.rglob("*.py")):
        parts = path.relative_to(PACKAGE).with_suffix("").parts
        if "tests" in parts:
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append((".".join(parts), path))
    return names


def _layer(name):
    """Return the layer of a module: its own entry, or that of its package."""
    probe = name
    while True:
        for layer, members in (
            ("base", BASE),
            ("leaf", LEAVES),
            ("solver", SOLVERS),
            ("driver", DRIVERS),
        ):
            if probe in members:
                # a package entry in DRIVERS names its __init__ only
                if layer == "driver" and probe != name:
                    return None
                return layer
        if "." not in probe:
            return None
        probe = probe.rsplit(".", 1)[0]


def _imports(name, path):
    """Return the package modules and the outside modules a module imports."""
    is_package = path.name == "__init__.py"
    here = name.split(".") if name else []
    inside, outside = set(), set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            outside.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                outside.add((node.module or "").split(".")[0])
                continue
            up = node.level - (1 if is_package else 0)
            anchor = here[: len(here) - up] if up else here
            if not is_package and node.level == 1:
                anchor = here[:-1]
            target = anchor + (node.module.split(".") if node.module else [])
            inside.add(".".join(target))
    return inside, outside


MODULES = _modules()


@pytest.mark.parametrize(
    ("name", "path"), MODULES, ids=[m[0] or "root" for m in MODULES]
)
def test_every_module_has_a_layer(name, path):
    assert _layer(name) is not None, f"assign pianistpy.{name} to a layer"


@pytest.mark.parametrize(
    ("name", "path"), MODULES, ids=[m[0] or "root" for m in MODULES]
)
def test_no_layer_imports_a_higher_one(name, path):
    layer = _layer(name)
    inside, outside = _imports(name, path)
    if layer in ("base", "leaf", "solver") and name not in SERVICES:
        touched = outside & FILE_MODULES
        assert not touched, f"pianistpy.{name} ({layer}) opens files: {touched}"
    allowed = {
        "base": {"base"},
        "leaf": {"base", "leaf"},
        "solver": {"base", "leaf", "solver"},
        "driver": {"base", "leaf", "solver", "driver"},
    }[layer]
    for target in inside:
        target_layer = _layer(target)
        # a stage may name the result objects it takes and returns
        if layer == "solver" and target == "df":
            continue
        assert target_layer in allowed, (
            f"pianistpy.{name} ({layer}) imports pianistpy.{target} ({target_layer})"
        )
