r"""Tests of the degree of the polynomial interpolant of the central deficit.

`truncate` raises the interpolant's degree through `DEFICIT_DEGREES` until the
interpolant's residual is below :math:`10^{-7}`. The model records the degree in
``model.inversion.deficit_degree``. When the last degree of `DEFICIT_DEGREES`
leaves a residual above :math:`10^{-7}`, `truncate` raises AccuracyError.
"""

import numpy as np
import pytest

from pianistpy import Anisotropy, DoublePowerLaw, Settings, Truncation
from pianistpy.df import center, core
from pianistpy.df.inputs import Component
from pianistpy.exceptions import AccuracyError


@pytest.fixture(scope="module")
def steep():
    return core.compute_df(
        (Component(DoublePowerLaw(1.0, 4.0, 1.5), Anisotropy()),), Settings()
    )


@pytest.fixture(scope="module")
def hernquist():
    return core.compute_df(
        (Component(DoublePowerLaw(1.0, 4.0, 1.0), Anisotropy()),), Settings()
    )


def test_the_steep_center_raises_its_degree(steep):
    model = core.truncate(steep, Truncation(k_t=0.5))
    (degree,) = model.inversion.deficit_degree
    assert degree in center.DEFICIT_DEGREES and degree > 8
    assert model.validation.tier == "silent"


def test_a_degree_that_stays_8_is_unchanged(hernquist, monkeypatch):
    model = core.truncate(hernquist, Truncation(k_t=0.2))
    assert model.inversion.deficit_degree == (8,)
    monkeypatch.setattr(center, "DEFICIT_DEGREES", (8,))
    fixed = core.truncate(hernquist, Truncation(k_t=0.2))
    for name in ("r", "p_trunc", "rho_trunc", "mass_trunc"):
        a = np.asarray(getattr(model.profile, name))
        b = np.asarray(getattr(fixed.profile, name))
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
    for name in ("k", "log_df", "log_df_slope"):
        a = np.asarray(getattr(model.table, name))
        b = np.asarray(getattr(fixed.table, name))
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
    assert model.profile.central_deficit == fixed.profile.central_deficit


def test_past_the_last_degree_the_center_is_refused(steep, monkeypatch):
    monkeypatch.setattr(center, "DEFICIT_DEGREES", (8,))
    with pytest.raises(
        AccuracyError,
        match=r"the deficit's interpolant misses it by .* at degree 8, the last "
        r"of the ladder \(limit 0.0000001\)",
    ):
        core.truncate(steep, Truncation(k_t=0.5))


def test_the_deficit_series_is_the_integral_at_a_depth(hernquist):
    # The coefficients D_k that central_deficit returns at P_0 are the
    # derivatives of the deficit integral in the depth u. The deficit at depth
    # v below P_0, computed by central_deficit at P_0 - v, therefore equals
    # sum_k (-1)^k D_k v^k for k = 0 to 4 to within 1e-9. The two-term sum
    # D_0 - D_1 v differs from the deficit at depth v by more than 1e-7, since
    # the two-term sum omits the term D_2 v^2.
    t = hernquist.table
    support = slice(int(t.i_first[0]), int(t.i_last[0]) + 1)
    q = np.asarray(t.k)[0, support]
    f = 10.0 ** np.asarray(t.log_df)[0, support]
    d = center.central_deficit(q, f, 0.2, 1.0, 0.0)
    assert len(d) == 5
    for v in (1e-3, 1e-2):
        at_v = center.central_deficit(q, f, 0.2, 1.0 - v, 0.0)[0]
        series = sum((-1) ** k * d[k] * v**k for k in range(5))
        assert abs(series / at_v - 1.0) < 1e-9
        assert abs((d[0] - d[1] * v) / at_v - 1.0) > 1e-7
