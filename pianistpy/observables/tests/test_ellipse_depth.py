r"""Tests of the depth :math:`P_0 - \Phi` below the central potential that the
ellipse route reads (`PotentialReading.depth`) and of the cumulative DF read
by the depth (`EllipseDF.G`).

:math:`P_0` is the model's relative potential at the center and :math:`\Phi`
is the profile's relative potential at radius r. The test model is a truncated
isotropic Hernquist model with truncation parameter ``k_t = 0.05``.
"""

import numpy as np
import pytest

from pianistpy import Anisotropy, Component, DoublePowerLaw, Truncation
from pianistpy.df import compute_df, truncate
from pianistpy.observables._ellipse_model import PotentialReading, ellipse_route
from pianistpy.observables._losvd_stages import ellipse_rules


@pytest.fixture(scope="module")
def model():
    """Return the isotropic Hernquist model truncated by
    ``Truncation(k_t=0.05)``."""
    df = compute_df(Component(DoublePowerLaw.hernquist(), Anisotropy(), name="hern"))
    return truncate(df, Truncation(k_t=0.05))


def test_the_depth_is_the_solves_own_column_near_the_center(model):
    r"""Assert where `PotentialReading.depth` reads the solve's column u_trunc.

    At the profile's radii where u_trunc is below :math:`P_0 / 2`, the depth
    equals u_trunc to a relative :math:`10^{-14}`. At the innermost radius,
    u_trunc is below :math:`10^{-6} P_0`. At the geometric midpoint of two
    neighboring radii where u_trunc is below :math:`P_0 / 2`, the depth lies
    strictly between u_trunc at the two radii. Where u_trunc lies between
    :math:`0.1 P_0` and :math:`0.4 P_0`, the depth equals :math:`P_0 - \Phi`
    to a relative :math:`10^{-12}`. Where u_trunc is above :math:`0.6 P_0`,
    the depth equals :math:`P_0 - \Phi` exactly. A `PotentialReading` built
    without :math:`P_0` returns a NaN depth at the radii where u_trunc is
    below :math:`P_0 / 2`.
    """
    profile = model.profile
    p_0 = float(model.form.p_0)
    reading = PotentialReading(profile, p_0)
    u = np.asarray(profile.u_trunc, dtype=np.float64)
    r = np.asarray(profile.r, dtype=np.float64)
    deep = np.flatnonzero(u < 0.5 * p_0)
    # At the profile's radii where u_trunc < P_0 / 2, the difference P_0 - Phi
    # loses digits to cancellation, so the depth is read from u_trunc. The
    # depth there is exp(ln u_trunc), which differs from u_trunc by a few units
    # in the last place.
    np.testing.assert_allclose(reading.depth(r[deep]), u[deep], rtol=1e-14, atol=0.0)
    assert u[0] < 1e-6 * p_0
    # Between two neighboring radii where u_trunc < P_0 / 2, the depth is a
    # cubic Hermite interpolant of ln u_trunc in ln r.
    mid = np.sqrt(r[deep][:-1] * r[deep][1:])
    d = reading.depth(mid)
    assert np.all(d > u[deep][:-1]) and np.all(d < u[deep][1:])
    # Where u_trunc lies between 0.1 P_0 and 0.4 P_0, P_0 - Phi keeps its
    # digits, so the Hermite interpolant and P_0 - Phi agree.
    both = (u > 0.1 * p_0) & (u < 0.4 * p_0)
    np.testing.assert_allclose(
        reading.depth(r[both]), p_0 - profile.potential(r[both]), rtol=1e-12
    )
    # Where u_trunc > 0.6 P_0, the depth is P_0 - Phi itself. A
    # PotentialReading built without P_0 holds P_0 = NaN, so the
    # PotentialReading's depth is NaN.
    far = r[u > 0.6 * p_0]
    assert np.array_equal(reading.depth(far), p_0 - profile.potential(far))
    assert np.all(np.isnan(PotentialReading(profile).depth(r[deep])))


def test_g_read_by_the_depth_is_g_read_by_the_energy_where_both_hold(model):
    r"""Assert that G read by the depth matches G read by the energy.

    G is the cumulative DF :math:`\int_0^Q F\,{\rm d}K` of the ellipse
    route's `EllipseDF`. The test asserts that the `EllipseDF` has
    ``depth_end`` set, which means that the top energy
    :math:`Q_{\rm end}` (the attribute ``Qend``) is the model's :math:`P_0`.
    For energies Q between :math:`0.6 Q_{\rm end}` and :math:`0.95 Q_{\rm end}`,
    G called with the depth :math:`Q_{\rm end} - Q` matches G called with Q
    alone to a relative :math:`10^{-14}`. G called with a depth of NaN equals
    G called with Q alone exactly.
    """
    df = ellipse_route(model, ellipse_rules(model.settings)).df
    assert df is not None and df.depth_end
    q = np.linspace(0.6, 0.95, 8) * df.Qend
    np.testing.assert_allclose(df.G(q, df.Qend - q), df.G(q), rtol=1e-14)
    # A depth of NaN tells G that the caller holds no depth, so G uses the
    # difference Q_end - Q.
    assert np.array_equal(df.G(q, np.full(q.shape, np.nan)), df.G(q))
