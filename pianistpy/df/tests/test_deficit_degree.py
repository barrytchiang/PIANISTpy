"""Tests of the degree of the central deficit's interpolant.

The degree is raised through `DEFICIT_DEGREES` until the residual, read at the
midpoints between the fit's nodes, is within the limit; the degree is in the
model's record, and past the last degree the center is refused by name.
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
    assert model.validation.tier == "goal"


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
        r"of the ladder \(limit a tenth of the goal\)",
    ):
        core.truncate(steep, Truncation(k_t=0.5))


def test_the_deficit_series_is_the_integral_at_a_depth(hernquist):
    # central_deficit's D_k are the u-derivatives of its own integral: its value
    # at depth v (the same integral at P_0 - v) is sum_k (-1)^k D_k v^k through
    # D4 to the series' remainder, where the linear pair misses by D2 v^2
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
