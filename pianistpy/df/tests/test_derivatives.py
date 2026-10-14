"""The exact derivatives of the auxiliary density, held against mpmath."""

import mpmath as mp
import numpy as np
import pytest

from pianistpy.df.derivatives import ExactDerivs
from pianistpy.profiles import DoublePowerLaw
from pianistpy.validation import GOAL

# Exponents off the small rationals: a rounded exponent is another model.
ALPHA, BETA, GAMMA, BETA_C = 0.9, 3.0, 1.137, 0.3333


def _mp_derivatives(profile, r, orders):
    """rho_aux and its first two derivatives in P at r, from mpmath's own
    derivatives in R: D = h d/dR with h = -R^2/M, M the integral of R^2 rho,
    so D f = h f' and D^2 f = h (h' f' + h f'')."""
    a, b, g, c = (mp.mpf(x) for x in (ALPHA, BETA, GAMMA, profile.rho_norm))

    def rho(s):
        return c * s ** (-g) * (1 + s**a) ** ((g - b) / a)

    def rho_aux(s):
        return rho(s) * s ** (2 * mp.mpf(BETA_C))

    with mp.workdps(30):
        s = mp.mpf(r)
        m = mp.quad(lambda t: t**2 * rho(t), [0, s])
        h = -(s**2) / m
        dh = -(2 * s * m - s**4 * rho(s)) / m**2
        f0, f1, f2 = (mp.diff(rho_aux, s, n) for n in (0, 1, 2))
        values = (f0, h * f1, h * (dh * f1 + h * f2))
        return [float(values[n]) for n in orders]


@pytest.mark.parametrize("r", [0.05, 0.7, 3.0, 40.0])
def test_the_derivatives_hold_against_mpmath_off_the_small_rationals(r):
    profile = DoublePowerLaw(ALPHA, BETA, GAMMA)
    source = ExactDerivs(profile, beta_c=BETA_C, model="constant", max_order=2)
    want = _mp_derivatives(profile, r, (0, 1, 2))
    got = [float(source._eval(n, np.array([r]))[0]) for n in (0, 1, 2)]
    np.testing.assert_allclose(got, want, rtol=GOAL)


def test_the_cached_construction_is_the_same_to_the_bit(tmp_path, monkeypatch):
    monkeypatch.setenv("PIANISTPY_COMPILATION_CACHE", str(tmp_path / "jax"))
    profile = DoublePowerLaw.plummer()
    fresh = ExactDerivs(profile, beta_c=-0.25, model="constant", max_order=4)
    assert len(list((tmp_path / "derivatives").iterdir())) == 1
    cached = ExactDerivs(profile, beta_c=-0.25, model="constant", max_order=4)
    r = np.logspace(-6, 6, 25)
    with np.errstate(all="ignore"):
        for n in range(5):
            np.testing.assert_array_equal(fresh.funcs[n](r), cached.funcs[n](r))
            with mp.workdps(30):
                assert fresh.funcs_mp[n](mp.mpf(3e-3)) == cached.funcs_mp[n](
                    mp.mpf(3e-3)
                )


def test_no_cached_construction_where_the_cache_is_off(tmp_path, monkeypatch):
    monkeypatch.setenv("PIANISTPY_COMPILATION_CACHE", "off")
    monkeypatch.setenv("HOME", str(tmp_path))
    ExactDerivs(DoublePowerLaw.hernquist(), max_order=2)
    assert not any(tmp_path.rglob("*.json"))
