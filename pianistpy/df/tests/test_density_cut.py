r"""Tests of `density_cut`, the inner radius below which the density is not computed.

`density_cut` returns the radius at which the cusp density
:math:`C R^{-\gamma}` reaches the largest float64, with :math:`C` the profile's
normalization and :math:`\gamma` its inner slope. The tests compare
`density_cut` with the fixed cut :math:`R = 10^{-13}` (``OLD_CUT``). For the
profiles :math:`(\alpha, \beta, \gamma) = (2, 4, \gamma)`, the center's window
(depths from :math:`10^{-4}` to :math:`3 \times 10^{-4}` below :math:`P_0`)
starts to cross :math:`R = 10^{-13}` between :math:`\gamma = 1.70` and
:math:`\gamma = 1.705`, and lies wholly inside :math:`R = 10^{-13}` between
:math:`\gamma = 1.735` and :math:`\gamma = 1.74`. A center's window that lies
wholly inside the cut raises AccuracyError.
"""

import numpy as np
import pytest

from pianistpy.df.center import center_amplitude
from pianistpy.df.poisson import density_cut, profile_density
from pianistpy.df.truncation import INNER_CEILING
from pianistpy.exceptions import AccuracyError
from pianistpy.profiles import DoublePowerLaw

OLD_CUT = 1e-13


def _window_radius(gamma, u):
    r"""Return the leading-order radius at depth `u` for inner slope `gamma`.

    The depth `u` is measured below the central potential :math:`P_0`. The
    radius is :math:`[u (2 - \gamma) (3 - \gamma) / C]^{1 / (2 - \gamma)}`,
    with :math:`\gamma` the inner slope `gamma` and C the normalization of
    the profile (2, 4, `gamma`).
    """
    c = float(DoublePowerLaw(2.0, 4.0, gamma).rho_norm)
    return (u * (2.0 - gamma) * (3.0 - gamma) / c) ** (1.0 / (2.0 - gamma))


def test_the_cut_is_where_the_cusp_leaves_float64():
    log_big = np.log(np.finfo(np.float64).max)
    for c, g in ((0.1, 1.9), (0.26, 1.74), (1.0, 1.0)):
        r = density_cut(c, g)
        assert r > 0.0
        assert np.log(c) - g * np.log(r) == pytest.approx(log_big, rel=1e-14)
        assert np.log(c) - g * np.log(2.0 * r) < log_big
    assert density_cut(1.0, 0.0) == 0.0


@pytest.mark.parametrize(
    "gamma, inner_crossed, all_crossed",
    [
        (1.70, False, False),
        (1.705, True, False),
        (1.735, True, False),
        (1.74, True, True),
    ],
)
def test_the_measured_edges_lie_above_the_new_cut(gamma, inner_crossed, all_crossed):
    r_in = _window_radius(gamma, INNER_CEILING)
    r_out = _window_radius(gamma, 3.0 * INNER_CEILING)
    assert (r_in <= OLD_CUT) == inner_crossed
    assert (r_out <= OLD_CUT) == all_crossed
    c = float(DoublePowerLaw(2.0, 4.0, gamma).rho_norm)
    assert r_in > 1e100 * density_cut(c, gamma)


def test_rows_below_the_old_cut_keep_their_density():
    q = np.linspace(0.01, 0.99, 50)
    p_rhs = q**2
    r = np.array([1e-20, 1e-10])
    phi = np.array([0.5, 0.5])
    c = float(DoublePowerLaw(2.0, 4.0, 1.74).rho_norm)
    args = (r, phi, q, p_rhs, 1.0, 0.0, "constant")
    new = np.asarray(profile_density(*args, order="linear", r_cut=density_cut(c, 1.74)))
    old = np.asarray(profile_density(*args, order="linear", r_cut=OLD_CUT))
    assert np.all(new > 0.0)
    assert old[0] == 0.0 and old[1] == new[1]


def test_a_window_below_the_cut_is_refused():
    p_0 = 0.7
    u = np.geomspace(1e-5, 1e-2, 200)
    r = np.geomspace(1e-200, 1e-190, 200)
    rho = np.zeros_like(r)
    with pytest.raises(AccuracyError, match="density's cut"):
        center_amplitude(
            r,
            p_0 - u,
            rho,
            0.3,
            p_0,
            1.0,
            np.linspace(0.01, 1.0, 10),
            np.ones(10),
            1.0,
            -0.5,
            (0.26, 2.0, 4.0, 1.74),
            (None, 1.0),
        )
