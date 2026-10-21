"""Tests of the stages of the line-of-sight velocity distribution (LOSVD) and of
the ellipse route's readers of a model.

The tests cover the ellipse route's rules read from the settings, the
derivatives of the logarithm of the potential, the bounds of the search for
the extrema along a line of sight, the cumulative DF, the Gauss-Hermite
moments, the LOSVD rows, table and aperture LOSVD of the isotropic Plummer
model, the floor of the DF, and the process pools that the stages start.
"""

import math
import re

import numpy as np
import pytest
from scipy.integrate import quad

from pianistpy import Anisotropy, DoublePowerLaw, InputError, Settings
from pianistpy.df.analytic import analytic_model
from pianistpy.df.lookup import prepare_lookup
from pianistpy.observables import _ellipse_model, _losvd_stages
from pianistpy.observables._ellipse_model import (
    LogPotentialSlopes,
    cumulative_df,
    ellipse_route,
)
from pianistpy.observables._ellipse_rules import EllipseRules
from pianistpy.observables._losvd_stages import ellipse_rules, shared_pool, start_pool
from pianistpy.observables.aperture import Aperture
from pianistpy.observables.stages import (
    ApertureLosvd,
    aperture_dispersion,
    aperture_losvd,
    dispersion_profile,
    gauss_hermite_moments,
    gauss_hermite_table,
    kept_ranges,
    losvd_row,
    losvd_table,
    projected_dispersion,
    validate_aperture,
    validate_table,
)
from pianistpy.profiles import TruncatedProfile
from pianistpy.validation import TIER_BOUNDS

pytestmark = pytest.mark.slow

# Grids smaller than the grids of the default settings, equal to the grids in
# test_stages.py.
SETTINGS = Settings(
    observables={"n_radii": 400, "n_jeans_nodes": 200, "n_projection_nodes": 200}
)


@pytest.fixture(scope="module")
def plummer():
    """Return the untruncated isotropic Plummer model and the model's projected
    dispersion (`projected_dispersion`) at the settings SETTINGS."""
    model = analytic_model("plummer")
    projected = projected_dispersion(
        model, dispersion_profile(model, SETTINGS), SETTINGS
    )
    return model, projected


@pytest.fixture(scope="module")
def plummer_table(plummer):
    """Return the LOSVD table (`losvd_table`) of the Plummer model of the
    `plummer` fixture."""
    model, projected = plummer
    return losvd_table(
        model, kept_ranges(model, projected, SETTINGS), projected, SETTINGS
    )


def test_the_default_settings_are_the_working_codes_rules():
    """Assert that `ellipse_rules` of the default settings equals the default
    `EllipseRules`."""
    assert ellipse_rules(Settings()) == EllipseRules()


def test_the_rows_take_the_kernel_by_default_and_numpy_as_the_reference():
    """Assert the angle rule's route that `ellipse_rules` reads from the
    setting ``observables.rows_route``.

    The default settings and the quick settings select the JAX kernel. The
    value ``"numpy"`` selects the numpy route. The value ``"gpu"`` raises
    InputError.
    """
    assert ellipse_rules(Settings()).kernel
    assert ellipse_rules(Settings(quick=True)).kernel
    assert not ellipse_rules(Settings(observables={"rows_route": "numpy"})).kernel
    with pytest.raises(InputError, match="rows_route"):
        ellipse_rules(Settings(observables={"rows_route": "gpu"}))


def test_the_log_potential_slopes_are_the_hermites_own():
    r"""Assert that `LogPotentialSlopes` returns the derivatives of the
    profile's cubic Hermite of ln P in ln r.

    Inside the knot intervals, the first derivative matches a central
    difference of step :math:`10^{-5}` to a relative :math:`10^{-6}`, and the
    second derivative matches a second difference to a relative
    :math:`10^{-3}` with an absolute tolerance of :math:`10^{-4}`. At a knot,
    the left-hand first derivative and the profile's `potential_log_slope`
    equal the right-hand first derivative to a relative :math:`10^{-12}`. At a
    knot, the left-hand second derivative equals the second derivative
    :math:`10^{-9}` left of the knot, and the right-hand second derivative
    equals the second derivative :math:`10^{-9}` right of the knot, each to a
    relative :math:`10^{-6}`.
    """
    profile = analytic_model("hernquist", n_nodes=201).profile
    slopes = LogPotentialSlopes(profile)
    x = slopes.x
    # Inside each knot interval, the derivatives are compared with central
    # differences of ln P in ln r of step h. The second difference carries a
    # rounding error of about 1e-16 / h^2 = 1e-6, which the atol of 1e-4 covers.
    y = 0.5 * (x[1:-1] + x[2:])[::40]
    h = 1e-5
    lp = lambda y: np.log(profile.potential(np.exp(y)))  # noqa: E731
    np.testing.assert_allclose(
        slopes(y, 1), (lp(y + h) - lp(y - h)) / (2.0 * h), rtol=1e-6
    )
    np.testing.assert_allclose(
        slopes(y, 2),
        (lp(y + h) - 2.0 * lp(y) + lp(y - h)) / h**2,
        rtol=1e-3,
        atol=1e-4,
    )
    # At a knot, the first derivative is continuous: both sides equal the
    # profile's potential_log_slope. At a knot, the second derivative jumps:
    # the left-hand value is the left interval's, the right-hand value the
    # right interval's.
    knots = x[5:-5:50]
    np.testing.assert_allclose(slopes(knots, 1), slopes.left(knots, 1), rtol=1e-12)
    np.testing.assert_allclose(
        slopes(knots, 1), profile.potential_log_slope(np.exp(knots)), rtol=1e-12
    )
    eps = 1e-9
    np.testing.assert_allclose(slopes(knots, 2), slopes(knots + eps, 2), rtol=1e-6)
    np.testing.assert_allclose(slopes.left(knots, 2), slopes(knots - eps, 2), rtol=1e-6)


@pytest.mark.parametrize(
    ("sgn", "r_a"),
    [(0.0, math.inf), (1.0, 3.0), (1.0, 0.3), (-1.0, 1e3), (-1.0, 10.0)],
)
def test_the_extrema_bounds_drop_only_dead_intervals(sgn, r_a):
    """Assert that the knot-interval bounds of `EllipseRoute.f_extrema` change
    no number.

    A route with the bounds (`EllipseRoute._live_bound`) and a route with
    every knot interval kept return the same extrema (`f_extrema_many`) and
    the same breaks and break positions (`breaks_many` with ``with_t=True``),
    bit for bit, at 300 projected radii. The cases set the route's anisotropy
    sign SGN to 0, +1 or -1 and the route's anisotropy radius RA to `r_a`.
    Every case with SGN other than 0 has at least one extremum.
    """
    # The test sets the route's SGN and RA directly. EllipseRoute._live_bound
    # reads SGN and RA.
    model = analytic_model("hernquist", n_nodes=201)

    def route():
        r = ellipse_route(model, ellipse_rules(SETTINGS), with_df=False)
        r.SGN, r.RA = sgn, r_a
        return r

    prof = model.profile
    r2 = np.geomspace(float(prof.r[0]) * (1 + 1e-9), 0.5 * prof.r_table_max, 300)
    bounded, every = route(), route()
    live = every._live_bound(every.psi.spline)
    every._live = np.ones_like(live)
    every._top = np.full(live.size, np.inf)
    a, b = bounded.f_extrema_many(r2), every.f_extrema_many(r2)
    assert sum(e.size for e in b) > 0 or sgn == 0.0
    for x, y in zip(a, b, strict=True):
        assert x.tobytes() == y.tobytes()
    bounded, every2 = route(), route()
    every2._live, every2._top = every._live, every._top
    for x, y in zip(
        bounded.breaks_many(r2, with_t=True),
        every2.breaks_many(r2, with_t=True),
        strict=True,
    ):
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes()


def _finite(r_t=2.0, n=400):
    """Return a TruncatedProfile of a uniform-density sphere of radius `r_t`
    on `n` radii, with an edge row at `r_t`.

    The density and the relative potential at the edge row are zero. The
    knot interval from the last interior radius to `r_t` is the Kepler
    interval of `LogPotentialSlopes`.
    """
    r = np.geomspace(1e-3, r_t, n)
    rho_0 = 3.0 / r_t**3
    mass = rho_0 * r**3 / 3.0
    p = rho_0 * (r_t**2 - r**2) / 6.0
    rho = np.full(n, rho_0)
    rho[-1], p[-1] = 0.0, 0.0
    return TruncatedProfile(
        r=r,
        p_trunc=p,
        rho_trunc=rho[np.newaxis, :],
        mass_trunc=mass,
        k_t=0.3,
        r_t=r_t,
        mass_t=1.0,
        spatially_finite=True,
        source=DoublePowerLaw.hernquist(),
        source_anisotropy=Anisotropy(),
        model="test-finite",
        edge_exponent=(2.5,),
    )


def test_the_kepler_interval_reads_its_closed_forms():
    r"""Assert that `LogPotentialSlopes` reads the Kepler interval of the
    uniform sphere `_finite` (:math:`r_{\rm t} = 2`) by the interval's closed
    forms.

    The last knot is ln 2. Between the last interior radius and
    :math:`r_{\rm t}`, the first derivative of ln P in ln r is
    :math:`-2/(2 - r)` and the second derivative is :math:`-2 r/(2 - r)^2`,
    each to a relative :math:`10^{-12}`. At the last interior radius, the
    right-hand first derivative is the Kepler value and the left-hand first
    derivative is the profile's `potential_log_slope`. At :math:`r_{\rm t}`,
    the left-hand first derivative is minus infinity.
    """
    profile = _finite()
    slopes = LogPotentialSlopes(profile)
    assert slopes.x[-1] == pytest.approx(math.log(2.0))
    r_last = float(profile.r[-2])
    r = np.geomspace(r_last * 1.0001, 1.999, 7)
    np.testing.assert_allclose(slopes(np.log(r), 1), -2.0 / (2.0 - r), rtol=1e-12)
    np.testing.assert_allclose(
        slopes(np.log(r), 2), -2.0 * r / (2.0 - r) ** 2, rtol=1e-12
    )
    # At the last interior radius, the right-hand limit of the first derivative
    # is the Kepler value, and the left-hand limit is the cubic Hermite's value.
    y = math.log(r_last)
    assert float(slopes(y, 1)) == pytest.approx(-2.0 / (2.0 - r_last), rel=1e-12)
    assert float(slopes.left(y, 1)) == pytest.approx(
        float(profile.potential_log_slope(r_last)), rel=1e-12
    )
    assert float(slopes.left(math.log(2.0), 1)) == -np.inf


def test_the_cumulative_df_is_the_lookups_integral():
    r"""Assert that `cumulative_df` at an energy Q equals the integral of the DF
    lookup from 0 to Q, to a relative :math:`10^{-6}`.

    For Q above the lookup's first energy K_0, the reference is
    `cumulative_df` at K_0 plus an adaptive quadrature of the DF lookup from
    K_0 to Q.
    """
    model = analytic_model("hernquist", n_nodes=201)
    arrays, kinds = prepare_lookup(model.table, model.form, 0)

    def f(k):
        from pianistpy.df.lookup import evaluate_lookup

        return float(10.0 ** evaluate_lookup(arrays, kinds, np.array([k]))[0])

    k_0 = float(arrays.params[0])
    for q in (0.5 * k_0, 1e-3, 0.3, 0.9):
        lo = min(q, k_0)
        exact = cumulative_df(arrays, kinds, lo)
        if q > k_0:
            exact += quad(f, k_0, q, limit=500, epsabs=0.0, epsrel=1e-12)[0]
        assert cumulative_df(arrays, kinds, q) == pytest.approx(
            exact, rel=TIER_BOUNDS[0]
        )


def test_an_unconverged_cumulative_df_is_refused_with_cause(monkeypatch):
    """Assert that `cumulative_df` and `_cumulative_order` raise AccuracyError,
    with a message that gives the energy Q, when the cumulative DF at Q does
    not converge. The test sets CUMULATIVE_TOLERANCE to 0."""
    from pianistpy.exceptions import AccuracyError
    from pianistpy.observables._table_vjp import _cumulative_order

    model = analytic_model("hernquist", n_nodes=201)
    arrays, kinds = prepare_lookup(model.table, model.form, 0)
    monkeypatch.setattr(_ellipse_model, "CUMULATIVE_TOLERANCE", 0.0)
    for read in (cumulative_df, _cumulative_order):
        with pytest.raises(AccuracyError, match=r"Q = 3\.0+e-01 is not converged"):
            read(arrays, kinds, 0.3)


def _hermite_d(v, h3, h4):
    """Return the unit Gaussian in `v` times 1 + h3 H_3(v) + h4 H_4(v).

    H_3 and H_4 are the normalized Hermite polynomials of van der Marel and
    Franx (1993).
    """
    w = v
    h3_ = (2.0 * w**3 - 3.0 * w) / math.sqrt(3.0)
    h4_ = (4.0 * w**4 - 12.0 * w**2 + 3.0) / (2.0 * math.sqrt(6.0))
    return np.exp(-0.5 * w * w) / math.sqrt(2.0 * math.pi) * (1.0 + h3 * h3_ + h4 * h4_)


def test_gauss_hermite_moments_of_a_gaussian_and_of_a_perturbed_one():
    r"""Assert `gauss_hermite_moments` of a unit Gaussian and of perturbed
    Gaussians (`_hermite_d`).

    For the unit Gaussian, the absolute mean and the absolute h3 are below
    :math:`10^{-14}`, the absolute h4 is below :math:`10^{-13}`, and sigma
    equals 1 and s4 equals 3, each to a relative :math:`10^{-13}`. For
    perturbations with h3 and h4 of size :math:`10^{-5}`, the relations
    :math:`s_3 = 4\sqrt{3}\,h_3` and :math:`s_4 - 3 = 8\sqrt{6}\,h_4` of
    Chiang+(2026d), Eq. (B66), hold to a relative :math:`10^{-3}`. For
    perturbations with h3 and h4 of size :math:`10^{-3}`, the moments return
    the perturbation's h3 and h4 to an absolute
    :math:`2 \times 10^{-5}`.
    """
    x, w = np.polynomial.legendre.leggauss(400)
    v, w = 12.0 * x, 12.0 * w
    g = gauss_hermite_moments(v, w, _hermite_d(v, 0.0, 0.0))
    assert abs(g["mean"]) < 1e-14 and g["sigma"] == pytest.approx(1.0, rel=1e-13)
    assert abs(g["h3"]) < 1e-14 and abs(g["h4"]) < 1e-13
    assert g["s4"] == pytest.approx(3.0, rel=1e-13)
    # Chiang+(2026d), Eq. (B66) holds to first order in h3 and h4. The
    # perturbations of size 1e-5 keep the second-order terms below the rtol
    # of 1e-3.
    for h3, h4 in ((1e-5, 0.0), (0.0, 1e-5), (-2e-5, 1.5e-5)):
        g = gauss_hermite_moments(v, w, _hermite_d(v, h3, h4))
        assert g["s3"] == pytest.approx(4.0 * math.sqrt(3.0) * g["h3"], rel=1e-3)
        assert g["s4"] - 3.0 == pytest.approx(8.0 * math.sqrt(6.0) * g["h4"], rel=1e-3)
    # The moments return the h3 and h4 of each perturbation.
    for h3, h4 in ((1e-3, 0.0), (0.0, 1e-3), (-2e-3, 1.5e-3)):
        g = gauss_hermite_moments(v, w, _hermite_d(v, h3, h4))
        assert g["h3"] == pytest.approx(h3, abs=2e-5)
        assert g["h4"] == pytest.approx(h4, abs=2e-5)


def test_rows_of_plummer_agree_with_its_projection(plummer):
    r"""Assert that LOSVD rows of the isotropic Plummer model agree with the
    model's projected dispersion at :math:`R_{\rm 2D}` = 0.01, 1 and 30.

    The Plummer model has no kept velocity ranges. Each row's records of the
    normalization against the surface density and of the second moment
    against the projected dispersion are below :math:`10^{-6}`. Twice the
    integral of each row over its nodes in :math:`V \ge 0` is 1 to a
    relative :math:`10^{-14}`. The row mirrored to :math:`V < 0` has a mean
    below :math:`10^{-12}` times :math:`\sigma_{\rm los}`, an h3 below
    :math:`10^{-12}`, and a sigma equal to :math:`\sigma_{\rm los}` to a
    relative :math:`10^{-6}`.
    """
    model, projected = plummer
    kept = kept_ranges(model, projected, SETTINGS)
    assert kept.v_rad.shape == (0, 2) and kept.v_k.shape == (0, 2)
    for r_2d in (1e-2, 1.0, 30.0):
        row = losvd_row(model, r_2d, kept, projected, SETTINGS)
        assert abs(row.record["norm_against_surface_density"]) < TIER_BOUNDS[0]
        assert abs(row.record["m2_against_projected_dispersion"]) < TIER_BOUNDS[0]
        assert row.losvd.shape == (1, len(row.v_nodes))
        assert 2.0 * float(np.dot(row.v_weights, row.losvd[0])) == pytest.approx(
            1.0, rel=1e-14
        )
        # The row holds V >= 0 only. Mirrored to V < 0, the row has mean zero
        # and dispersion sigma_los.
        v = np.concatenate([-row.v_nodes[::-1], row.v_nodes])
        w = np.concatenate([row.v_weights[::-1], row.v_weights])
        d = np.concatenate([row.losvd[0][::-1], row.losvd[0]])
        g = gauss_hermite_moments(v, w, d)
        sigma_los = float(projected.read(np.array([r_2d]))[1][0, 0])
        assert abs(g["mean"]) < 1e-12 * sigma_los
        assert g["sigma"] == pytest.approx(sigma_los, rel=TIER_BOUNDS[0])
        assert abs(g["h3"]) < 1e-12


def test_a_row_states_its_norm_through_the_dfs_own_factor(plummer, monkeypatch):
    r"""Assert that a DF read with a constant factor leaves the row unchanged.

    The test sets LOG10_DF_CEILING to 7 below the largest log10 value of the
    DF, so the ellipse route reads the DF times the factor `EllipseDF.scale`
    of :math:`10^{-7}`. The row's records of the normalization and of the
    second moment change by less than :math:`10^{-12}`. The row's normalized
    LOSVD changes by a relative :math:`10^{-12}` at most.
    """
    model, projected = plummer
    kept = kept_ranges(model, projected, SETTINGS)
    rows = [losvd_row(model, 1.0, kept, projected, SETTINGS)]
    arrays, _ = prepare_lookup(model.table, model.form, 0)
    log_df = np.asarray(arrays.log_df[: int(arrays.n)])
    top = float(np.max(log_df[np.isfinite(log_df)]))
    monkeypatch.setattr(_ellipse_model, "LOG10_DF_CEILING", top - 7.0)
    df = ellipse_route(model, ellipse_rules(SETTINGS)).df
    assert df is not None and df.scale == pytest.approx(1e-7, rel=1e-12)
    rows.append(losvd_row(model, 1.0, kept, projected, SETTINGS))
    for key in ("norm_against_surface_density", "m2_against_projected_dispersion"):
        assert rows[1].record[key] == pytest.approx(rows[0].record[key], abs=1e-12)
    assert np.allclose(rows[1].losvd, rows[0].losvd, rtol=1e-12, atol=0.0)


def _floor_at(monkeypatch, k_0):
    """Patch the ellipse route's `prepare_lookup` so that every DF lookup has
    the floor K_0 = `k_0` and the rule "zero", which sets the DF to zero below
    K_0."""
    real = _ellipse_model.prepare_lookup

    def prepare(table, form, component):
        arrays, kinds = real(table, form, component)
        params = np.array(arrays.params, dtype=np.float64)
        params[_ellipse_model.K_0] = k_0
        return arrays._replace(params=params), kinds._replace(below_floor="zero")

    monkeypatch.setattr(_ellipse_model, "prepare_lookup", prepare)


def test_the_rows_end_where_the_potential_reaches_the_dfs_floor(plummer, monkeypatch):
    r"""Assert that the rows end at the radius where the potential P equals
    the DF's floor K_0.

    Without a floor, `floor_radius` is the profile's last radius. With K_0
    set to P at r = 5, `floor_radius` is 5 to a relative :math:`10^{-12}`.
    The route's :math:`R_{\rm t}` equals `floor_radius`. The last base row is
    EDGE_FRACTION times `floor_radius`. The last row lies inside
    `floor_radius`, at a potential P above K_0.
    """
    model, _ = plummer
    profile = model.profile
    r_f = 5.0
    k_0 = float(profile.potential(np.array([r_f]))[0])
    assert _ellipse_model.floor_radius(model) == float(profile.r_table_max)
    _floor_at(monkeypatch, k_0)
    end = _ellipse_model.floor_radius(model)
    assert end == pytest.approx(r_f, rel=1e-12)
    rules = ellipse_rules(SETTINGS)
    route = ellipse_route(model, rules, with_df=False)
    assert route.R_t == end
    base = _losvd_stages.base_rows(model, SETTINGS)
    assert base[-1] == pytest.approx(end * _losvd_stages.EDGE_FRACTION, rel=1e-12)
    r_rows, _ = route.rows_with_r_breaks(base)
    assert r_rows[-1] < end
    assert float(profile.potential(r_rows[-1:])[0]) > k_0


def test_a_floor_inside_the_first_row_is_refused_with_cause(plummer, monkeypatch):
    """Assert that `floor_radius` and `base_rows` raise ResultError when the
    potential reaches the DF's floor K_0 at or inside the profile's first
    radius. The message gives K_0 and the profile's first radius."""
    from pianistpy.exceptions import ResultError

    model, _ = plummer
    profile = model.profile
    k_0 = float(profile.potential(profile.r[:1])[0]) * (1.0 + 1e-9)
    _floor_at(monkeypatch, k_0)
    message = (
        f"floor K_0 = {k_0:.6e} (the rule 'zero'), which the potential reaches "
        f"at or inside the profile's first tabulated radius "
        f"r = {float(profile.r[0]):.6e}"
    )
    with pytest.raises(ResultError, match=re.escape(message)):
        _ellipse_model.floor_radius(model)
    with pytest.raises(ResultError, match=re.escape(message)):
        _losvd_stages.base_rows(model, SETTINGS)


def test_table_and_aperture_of_plummer(plummer, plummer_table):
    r"""Assert the LOSVD table and the aperture LOSVD of the Plummer model in
    a centered circle of radius 1.

    The table's validation report has level 0, the tier "silent". The
    aperture's check has level 0, the tier "silent". The projected mass
    inside the circle is :math:`2\pi` to a relative :math:`10^{-6}`.
    """
    model, projected = plummer
    table = plummer_table
    report = validate_table(table, projected)
    assert report.level == 0, report.to_dict()
    aperture = Aperture.circle(1.0)
    result = aperture_losvd(model, table, projected, aperture, SETTINGS)
    assert isinstance(result, ApertureLosvd)
    sigma_los = aperture_dispersion(projected, aperture, SETTINGS)
    check = validate_aperture(result, sigma_los)
    assert check.level == 0, check.to_dict()
    # In internal units, the Plummer model's surface density is
    # Sigma = 4 (1 + R^2)^-2, so the projected mass inside R is
    # 4 pi R^2 / (1 + R^2), which is 2 pi at R = 1.
    assert result.mass[0] == pytest.approx(2.0 * math.pi, rel=TIER_BOUNDS[0])


def test_a_row_whose_t_rule_did_not_converge_refuses_the_table(plummer, plummer_table):
    """Assert that a row whose rule along the line of sight did not converge
    gives the table's validation report the tier "refused".

    The test marks the middle row's record with ``t_converged=False``. The
    report's check ``t_converged`` then has an infinite error at the middle
    row's projected radius. The unmodified table's check ``t_converged`` has
    error 0.
    """
    import dataclasses

    _, projected = plummer
    rows = list(plummer_table.rows)
    k = len(rows) // 2
    rows[k] = dataclasses.replace(
        rows[k], record={**rows[k].record, "t_converged": False}
    )
    report = validate_table(
        dataclasses.replace(plummer_table, rows=tuple(rows)), projected
    )
    assert report.tier == "refused"
    check = report.checks["t_converged"]
    assert check.error == float("inf") and check.where == rows[k].r_2d
    assert validate_table(plummer_table, projected).checks["t_converged"].error == 0


def test_an_aperture_beyond_the_table_is_refused_in_one_statement(
    plummer, plummer_table
):
    """Assert that an aperture outside the table's projected radii raises
    InputError with one message.

    `aperture_dispersion` and `aperture_losvd`, each for a circle and for a
    rectangle offset by 1000 times the outermost projected radius, raise
    InputError. The four messages are equal once the bracketed radius ranges
    are masked.
    """
    model, projected = plummer
    far = 1e3 * float(projected.r_2d[-1])
    messages = set()
    for aperture in (
        Aperture.circle(1.0, r_off=far),
        Aperture.rectangle(2.0, 2.0, r_off=far),
    ):
        with pytest.raises(InputError) as dispersion:
            aperture_dispersion(projected, aperture, SETTINGS)
        with pytest.raises(InputError) as distribution:
            aperture_losvd(model, plummer_table, projected, aperture, SETTINGS)
        messages |= {
            re.sub(r"\[[^]]*\]", "[...]", str(e.value))
            for e in (dispersion, distribution)
        }
    assert messages == {
        "The aperture lies outside the tabulated projected radii [...], "
        "so the aperture encloses no surface density."
    }, messages


def test_a_shared_scope_starts_one_pool_a_key(monkeypatch):
    """Assert that `shared_pool` keeps one pool per key.

    Inside the scope, two `start_pool` calls with the key "a" return the same
    pool, and a call with the key "b" returns a different pool. After the
    scope ends, a map on the first pool raises ValueError, and no scope
    remains open.
    """
    # One core, so that each pool starts one process.
    monkeypatch.setattr(_losvd_stages, "get_cores", lambda: 1)
    with shared_pool():
        with start_pool(None, key="a") as first:
            assert first.map(abs, [-1, 2]) == [1, 2]
        with start_pool(None, key="a") as again:
            assert again is first
        with start_pool(None, key="b") as other:
            assert other is not first
    # The end of the scope terminates the scope's pools.
    with pytest.raises(ValueError):
        first.map(abs, [1])
    assert not _losvd_stages._SCOPES


def test_the_gauss_hermite_moments_read_across_the_table(plummer, plummer_table):
    r"""Assert the readings of the Gauss-Hermite moments of the Plummer table
    (`gauss_hermite_table`).

    At the rows' radii, `read` returns the rows' moments bit for bit. Under
    jax.jit, `read_traced` returns the rows' moments bit for bit on the CPU
    and within 4 units in the last place on any other platform. At the
    geometric midpoints between rows, the traced sigma and h4 match `read`
    to a relative :math:`10^{-13}` with an absolute tolerance of
    :math:`10^{-16}`. At the midpoints inside 0.9 times the last row's
    radius, sigma matches the projected dispersion to a relative
    :math:`10^{-6}`. At every midpoint, the absolute mean is below
    :math:`10^{-12}`. A
    radius beyond the last row or inside the first row raises InputError.
    """
    import jax
    import jax.numpy as jnp

    _, projected = plummer
    gh = gauss_hermite_table(plummer_table)
    r = gh.r_2d
    # At the rows' radii, read returns the rows' own moments bit for bit.
    # read_traced runs in the platform's own arithmetic: bit for bit on the
    # CPU, and within 4 units in the last place on any other platform.
    host = gh.read(r)
    traced = jax.jit(lambda x: gh.read_traced(x, jnp))(jnp.asarray(r))
    platform = jax.default_backend()
    ulps = 0 if platform == "cpu" else 4
    for k in ("mean", "sigma", "h3", "h4"):
        np.testing.assert_array_equal(host[k], gh.moments[k])
        off = np.abs(np.asarray(traced[k]) - gh.moments[k]) / np.spacing(
            np.abs(gh.moments[k])
        )
        assert np.all(off <= ulps), (
            f"{k} on {platform}: {off.max():.0f} ulp, {ulps} allowed"
        )
    # Between rows, the traced reading matches read, and sigma matches the
    # projected dispersion.
    mid = np.sqrt(r[:-1] * r[1:])
    between = gh.read(mid)
    traced = jax.jit(lambda x: gh.read_traced(x, jnp))(jnp.asarray(mid))
    for k in ("sigma", "h4"):
        np.testing.assert_allclose(
            np.asarray(traced[k]), between[k], rtol=1e-13, atol=1e-16
        )
    inside = mid < 0.9 * r[-1]
    np.testing.assert_allclose(
        between["sigma"][inside], projected.read(mid[inside])[1][0], rtol=1e-6
    )
    assert np.all(np.abs(between["mean"]) < 1e-12)
    # A radius beyond the last row or inside the first row raises InputError.
    with pytest.raises(InputError, match="beyond the outermost projected radius"):
        gh.read(np.array([2.0 * r[-1]]))
    with pytest.raises(InputError):
        gh.read(np.array([0.5 * r[0]]))


def test_a_pools_workers_run_on_the_cpu(monkeypatch):
    """Assert that a pool's processes run on the CPU while the main process's
    environment selects the GPU.

    The main process sets JAX_PLATFORMS to "cuda,cpu" and PIANISTPY_PLATFORM
    to "gpu". A process of `new_pool` reports the platform "cpu". After the
    pool ends, both environment variables of the main process hold the
    values the test set.
    """
    import os

    from pianistpy import platform
    from pianistpy.observables._losvd_table import new_pool

    monkeypatch.setenv("JAX_PLATFORMS", "cuda,cpu")
    monkeypatch.setenv("PIANISTPY_PLATFORM", "gpu")
    with new_pool(1, None) as pool:
        assert pool.apply(platform) == "cpu"
    assert os.environ["JAX_PLATFORMS"] == "cuda,cpu"
    assert os.environ["PIANISTPY_PLATFORM"] == "gpu"


def _worker_route(_: object) -> bytes:
    """Return the pickled bytes of the route that a pool process holds
    (``WORKER["route"]``)."""
    from multiprocessing.reduction import ForkingPickler

    from pianistpy.observables._kept import WORKER

    return bytes(ForkingPickler.dumps(WORKER["route"]))


def test_a_pools_route_is_read_from_a_file_removed_with_the_pool(plummer):
    """Assert that the processes of `new_pool` read the route from a file that
    the pool's end removes.

    The route held by a process of a plain spawn pool, which receives the
    route as the initializer's argument, pickles to the reference bytes.
    The route held by each process of `new_pool` pickles to the same bytes.
    While the pool runs, the route's folder holds the one file route.pickle.
    After the pool ends, the folder no longer exists.
    """
    # The reference bytes come from the route pickled and loaded once, as a
    # pool process loads the route. The route in the main process holds array
    # views and reversed strides, which a load makes contiguous, so the route
    # in the main process is not the reference.
    import multiprocessing as mp
    import os
    import pickle
    from multiprocessing.reduction import ForkingPickler

    from pianistpy.observables._losvd_table import (
        START_METHOD,
        init_worker,
        new_pool,
    )
    from pianistpy.runtime import worker_environment

    model, _ = plummer
    route = ellipse_route(model, ellipse_rules(SETTINGS))
    loaded = bytes(ForkingPickler.dumps(pickle.loads(ForkingPickler.dumps(route))))
    with worker_environment():
        given = mp.get_context(START_METHOD).Pool(1, init_worker, (route,))
    with given:
        assert given.map(_worker_route, range(1)) == [loaded]
    with new_pool(2, route) as pool:
        folder = pool.remove_route.peek()[2][0]
        assert os.listdir(folder) == ["route.pickle"]
        assert pool.map(_worker_route, range(4), chunksize=1) == [loaded] * 4
    assert not os.path.exists(folder)


def test_a_second_aperture_on_a_table_forms_no_extrema(
    plummer, plummer_table, monkeypatch
):
    """Assert that a second aperture on a table forms no extrema along the
    line of sight and starts no pool.

    With one core, `aperture_losvd` forms the extrema in the main process.
    With two cores, `aperture_losvd` forms the extrema in a pool and keeps
    the extrema and the one pool on the table. A second call with two cores
    on the same table forms no extrema, starts no pool and reuses the
    table's pool. The results of the three calls agree bit for bit in the
    velocity nodes, weights, breaks, LOSVD and mass, and the records are
    equal.
    """
    import dataclasses

    from pianistpy import runtime
    from pianistpy.observables import stages
    from pianistpy.observables._ellipse import EllipseRoute

    model, projected = plummer
    aperture = Aperture.circle(1.0)
    fields = ("v_nodes", "v_weights", "v_breaks", "losvd", "mass")
    saved = runtime._cores
    try:
        runtime.set_cores(1)
        alone = aperture_losvd(
            model,
            dataclasses.replace(plummer_table, extrema={}),
            projected,
            aperture,
            SETTINGS,
        )
        runtime.set_cores(2)
        table = dataclasses.replace(plummer_table, extrema={})
        pooled = aperture_losvd(model, table, projected, aperture, SETTINGS)
        assert table.extrema
        # The table keeps the one pool that the call with two cores started.
        assert len(table.pools) == 1
        kept_pool = next(iter(table.pools.values()))
        formed, started = [], []
        chunk = EllipseRoute._f_extrema_chunk
        monkeypatch.setattr(
            EllipseRoute,
            "_f_extrema_chunk",
            lambda self, *a: formed.append(1) or chunk(self, *a),
        )
        start = stages.new_pool
        monkeypatch.setattr(
            stages, "new_pool", lambda *a, **k: started.append(1) or start(*a, **k)
        )
        again = aperture_losvd(model, table, projected, aperture, SETTINGS)
        assert next(iter(table.pools.values())) is kept_pool
        table.close_pools()
        assert not table.pools
    finally:
        runtime.set_cores(saved)
    assert not formed and not started
    for result in (pooled, again):
        for name in fields:
            a, b = np.asarray(getattr(alone, name)), np.asarray(getattr(result, name))
            assert a.tobytes() == b.tobytes(), name
        assert result.record == alone.record


def test_a_layout_that_returns_between_two_samples_is_found():
    r"""Assert that `rows_with_r_breaks` finds every change of the layout.

    The model has the density profile of a Hernquist model truncated by
    ``Truncation(k_t=0.05)`` and the anisotropy :math:`\beta_{\rm c} = -0.2`,
    :math:`r_{\rm a} = 1.5`, with the quick settings. The test asserts that an
    R-break lies between :math:`R_{\rm 2D} = 23.31699` and
    :math:`R_{\rm 2D} = 23.31700`, where the velocity lattice merges V_k into
    V_end. The test asserts that the rows between two neighboring
    R-breaks share one layout code.
    """
    from pianistpy import Component, Truncation, compute_df, solve, truncate
    from pianistpy.observables._losvd_stages import base_rows

    quick = Settings(quick=True)
    parent = solve(Component(DoublePowerLaw.hernquist()), Truncation(k_t=0.05), quick)
    model = truncate(
        compute_df(
            Component(
                profile=parent.profile, anisotropy=Anisotropy(beta_c=-0.2, r_a=1.5)
            ),
            quick,
        )
    )
    projected = projected_dispersion(model, dispersion_profile(model, quick), quick)
    kept = kept_ranges(model, projected, quick)
    route = ellipse_route(model, ellipse_rules(quick), with_df=False)
    route.vrad_kept, route.vk_kept = kept.v_rad, kept.v_k
    rows, breaks = route.rows_with_r_breaks(base_rows(model, quick))
    assert np.any((breaks > 23.31699) & (breaks < 23.31700))
    code, stretch = route.layout_many(rows), np.searchsorted(breaks, rows)
    for st in np.unique(stretch):
        assert np.unique(code[stretch == st]).size == 1, st


def test_a_pool_whose_worker_ends_raises_by_name():
    """Assert that a pool of `new_pool` raises WorkerError when one of the
    pool's processes ends, and serves the next call.

    A task that ends its own process with exit code 3 makes `map` raise
    WorkerError with a message that gives the exit code, memory and
    PIANISTPY_CORES. A process ended from outside the pool, namely the
    process that the pool started in place of the first ended process, makes
    `imap_unordered` raise WorkerError. After each WorkerError, the pool's
    next call of `abs` returns the absolute values of the call's inputs.
    """
    import os
    import time

    from pianistpy.exceptions import WorkerError
    from pianistpy.observables._losvd_table import new_pool

    from ._pools import bounded

    with new_pool(2, None) as pool, bounded():
        assert pool.map(abs, [-1, -2, 3]) == [1, 2, 3]
        with pytest.raises(WorkerError, match="exit code 3") as ended:
            pool.map(os._exit, [3])
        assert "memory" in str(ended.value)
        assert "PIANISTPY_CORES" in str(ended.value)
        assert pool.map(abs, [-4, 5]) == [4, 5]
        while len(pool.workers_started) < 3:
            time.sleep(0.01)
        replacement = pool.workers_started[-1]
        replacement.kill()
        while replacement.exitcode is None:
            time.sleep(0.01)
        with pytest.raises(WorkerError, match="A parallel process ended"):
            list(pool.imap_unordered(time.sleep, [5, 5]))
        assert sorted(pool.imap_unordered(abs, [-6, 7])) == [6, 7]


def test_a_pool_whose_task_pipe_lock_is_lost_terminates(caplog):
    """Assert that a pool's termination returns when the lock of the pool's
    task pipe is never released.

    A pool process takes the lock of the task pipe to read a task. A process
    ended while holding the lock never releases the lock. The test takes
    the lock itself while both processes are busy with a task or still
    starting, ends one process, and never releases the lock. The pool's
    termination returns within `_pools.BOUND` seconds, logs nothing, and leaves every
    process ended. The termination of the standard library's pool would
    wait on the lock without end.
    """
    import time

    from pianistpy.observables._losvd_table import new_pool

    from ._pools import BOUND, bounded

    with bounded():
        with new_pool(2, None) as pool:
            pool.map_async(time.sleep, [30, 30], chunksize=1)
            assert pool._inqueue._rlock.acquire(timeout=BOUND)
            worker = pool.workers_started[0]
            worker.kill()
            while worker.exitcode is None:
                time.sleep(0.01)
            caplog.clear()
        assert not caplog.records
        assert all(p.exitcode is not None for p in pool.workers_started)


def test_the_builds_pool_raises_when_a_worker_ends(plummer, plummer_table, monkeypatch):
    """Assert that `losvd_table` raises WorkerError when the first process of
    the pool that `start_pool` starts has ended, and that the call starts
    one pool."""
    from pianistpy import runtime
    from pianistpy.exceptions import WorkerError

    from ._pools import bounded, end_first_worker

    model, projected = plummer
    started = end_first_worker(_losvd_stages, monkeypatch)
    saved = runtime._cores
    try:
        runtime.set_cores(2)
        with bounded(), pytest.raises(WorkerError):
            losvd_table(model, plummer_table.kept, projected, SETTINGS)
    finally:
        runtime.set_cores(saved)
    assert len(started) == 1


def test_the_apertures_pool_raises_when_a_worker_ends(
    plummer, plummer_table, monkeypatch
):
    """Assert that `aperture_losvd` raises WorkerError when the first process
    of the pool that `table_pool` starts for the extrema has ended, and that
    the call starts one pool."""
    import dataclasses

    from pianistpy import runtime
    from pianistpy.exceptions import WorkerError
    from pianistpy.observables import stages

    from ._pools import bounded, end_first_worker

    model, projected = plummer
    started = end_first_worker(stages, monkeypatch)
    table = dataclasses.replace(plummer_table, extrema={})
    saved = runtime._cores
    try:
        runtime.set_cores(2)
        with bounded(), pytest.raises(WorkerError):
            aperture_losvd(model, table, projected, Aperture.circle(1.0), SETTINGS)
    finally:
        table.close_pools()
        runtime.set_cores(saved)
    assert len(started) == 1


def test_the_rows_own_pool_raises_when_a_worker_ends(
    plummer_table, plummer, monkeypatch
):
    """Assert that `losvd_rows`, given no pool, raises WorkerError when the
    first process of the pool that `losvd_rows` starts has ended, and that
    the call starts one pool."""
    from pianistpy import runtime
    from pianistpy.exceptions import WorkerError
    from pianistpy.observables import _losvd_table

    from ._pools import bounded, end_first_worker

    model, _ = plummer
    kept = plummer_table.kept
    route = ellipse_route(model, ellipse_rules(SETTINGS))
    route.vrad_kept, route.vk_kept = kept.v_rad, kept.v_k
    r_2d = np.array([row.r_2d for row in plummer_table.rows[:4]])
    started = end_first_worker(_losvd_table, monkeypatch)
    saved = runtime._cores
    try:
        runtime.set_cores(2)
        with bounded(), pytest.raises(WorkerError):
            _losvd_table.losvd_rows(
                route,
                r_2d,
                None,
                None,
                vrad_kept=kept.v_rad,
                vk_kept=kept.v_k,
                certify=False,
            )
    finally:
        runtime.set_cores(saved)
    assert len(started) == 1
