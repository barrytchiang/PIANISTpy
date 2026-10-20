"""Tests of the ellipse route's depth below the central potential."""

import numpy as np
import pytest

from pianistpy import Anisotropy, Component, DoublePowerLaw, Truncation
from pianistpy.df import compute_df, truncate
from pianistpy.observables._ellipse_model import PotentialReading, ellipse_route
from pianistpy.observables._losvd_stages import ellipse_rules


@pytest.fixture(scope="module")
def model():
    df = compute_df(Component(DoublePowerLaw.hernquist(), Anisotropy(), name="hern"))
    return truncate(df, Truncation(k_t=0.05))


def test_the_depth_is_the_solves_own_column_near_the_center(model):
    profile = model.profile
    p_0 = float(model.form.p_0)
    reading = PotentialReading(profile, p_0)
    u = np.asarray(profile.u_trunc, dtype=np.float64)
    r = np.asarray(profile.r, dtype=np.float64)
    deep = np.flatnonzero(u < 0.5 * p_0)
    # on the rows: the column (to the few ulps of exp(ln u)), where P_0 - Phi has
    # lost the digits of the depth
    np.testing.assert_allclose(reading.depth(r[deep]), u[deep], rtol=1e-14, atol=0.0)
    assert u[0] < 1e-6 * p_0
    # between the rows near the center: the Hermite of ln u, smooth in ln r
    mid = np.sqrt(r[deep][:-1] * r[deep][1:])
    d = reading.depth(mid)
    assert np.all(d > u[deep][:-1]) and np.all(d < u[deep][1:])
    # where the difference keeps its digits, the two readings agree
    both = (u > 0.1 * p_0) & (u < 0.4 * p_0)
    np.testing.assert_allclose(
        reading.depth(r[both]), p_0 - profile.potential(r[both]), rtol=1e-12
    )
    # beyond the deep rows, and with no central potential, the difference itself
    far = r[u > 0.6 * p_0]
    assert np.array_equal(reading.depth(far), p_0 - profile.potential(far))
    assert np.all(np.isnan(PotentialReading(profile).depth(r[deep])))


def test_g_read_by_the_depth_is_g_read_by_the_energy_where_both_hold(model):
    df = ellipse_route(model, ellipse_rules(model.settings)).df
    assert df is not None and df.depth_end
    q = np.linspace(0.6, 0.95, 8) * df.Qend
    np.testing.assert_allclose(df.G(q, df.Qend - q), df.G(q), rtol=1e-14)
    # NaN where no depth is held: the energy's own reading
    assert np.array_equal(df.G(q, np.full(q.shape, np.nan)), df.G(q))
