"""Tests of the Poisson solve's inner extension.

Where the series' deficit at the solve's first radius exceeds the solver's
tolerance, the grid is extended inward to the radius where the two meet,
(sqrt(ode_atol) / A)^(1 / (2 - gamma)). At gamma 1.99 that radius lies far
below the double's range, and the solve refuses by name instead of dividing by
the 0 the power rounds to.
"""

import pytest

from pianistpy import Anisotropy, DoublePowerLaw, Settings, Truncation
from pianistpy.df import core
from pianistpy.df.inputs import Component
from pianistpy.exceptions import InputError


def test_an_inner_floor_below_the_double_is_refused():
    settings = Settings(truncation={"log10_r_min": -32.0})
    component = Component(DoublePowerLaw(1.0, 4.0, 1.99), Anisotropy(r_a=0.3))
    df = core.compute_df((component,), settings)
    with pytest.raises(
        InputError,
        match=r"the series meets the solver's absolute tolerance only at R = "
        r"\(sqrt\(ode_atol\) / A\)\^\(1 / \(2 - gamma\)\) = 10\^-5\d\d\.\d .*"
        r"gamma = 1\.990\), below the double's range",
    ):
        core.truncate(df, Truncation(k_t=0.01))
