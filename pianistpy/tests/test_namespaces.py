"""Tests of the package's namespaces.

A name bound in a package's namespace (a function, a class or a constant)
that equals the name of one of the package's own submodules shadows that
submodule on attribute access: ``package.name`` is then the object, not the
module, and ``import package.name as module`` fails. The test walks every
package of ``pianistpy`` and requires that no bound name shadows a
submodule.
"""

import importlib
import pkgutil
import types

import pianistpy


def _packages():
    yield "pianistpy"
    for info in pkgutil.walk_packages(pianistpy.__path__, "pianistpy."):
        if info.ispkg:
            yield info.name


def test_no_bound_name_shadows_a_submodule():
    shadowed = []
    for name in _packages():
        package = importlib.import_module(name)
        submodules = {
            info.name.rsplit(".", 1)[-1]
            for info in pkgutil.iter_modules(package.__path__)
        }
        for attr, value in vars(package).items():
            if attr in submodules and not isinstance(value, types.ModuleType):
                shadowed.append(f"{name}.{attr} ({type(value).__name__})")
    assert shadowed == [], shadowed
