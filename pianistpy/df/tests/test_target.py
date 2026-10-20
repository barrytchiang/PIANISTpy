"""The truncation-energy search's failures: the solve's own errors mark an
energy inadmissible; any other error is raised."""

import pytest

from pianistpy import Anisotropy, Component, DoublePowerLaw
from pianistpy.df import compute_df, target


def test_an_error_that_is_not_the_solves_is_raised(monkeypatch):
    # release audit 317 A8: the search reads the package's exceptions only
    df = compute_df(Component(DoublePowerLaw.hernquist(), Anisotropy()))

    def broken(df, z):
        raise TypeError("a fault, not an inadmissible K_t")

    monkeypatch.setattr(target, "truncation_chain", broken)
    with pytest.raises(TypeError, match="a fault"):
        target.solve_target(df, "radius", 10.0)
