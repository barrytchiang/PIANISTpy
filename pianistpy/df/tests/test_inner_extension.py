r"""Tests of the inward extension of the radial grid of the Poisson solve.

Near the center, the central series gives the potential deficit
:math:`u = P_0 - \Phi(R) = A R^{2 - \gamma}`, with :math:`P_0` the central
potential, :math:`\Phi(R)` the potential at radius R and :math:`\gamma` the
inner density slope. Let :math:`\epsilon` be the solver's absolute tolerance
`ode_atol`. Where u at the solve's first radius exceeds
:math:`\sqrt{\epsilon}`, the solve extends the radial grid inward to the
radius :math:`R = (\sqrt{\epsilon} / A)^{1 / (2 - \gamma)}`, at which u
equals :math:`\sqrt{\epsilon}`. At :math:`\gamma = 1.99` the extended radius
lies below the smallest normal double, and the solve raises InputError,
which states the extended radius as a power of ten.
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
