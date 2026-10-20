"""Tests of the LOSVD stages and of the ellipse route's readers of a model."""

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
from pianistpy.validation import GOAL

SETTINGS = Settings(
    observables={"n_radii": 400, "n_jeans_nodes": 200, "n_projection_nodes": 200}
)


@pytest.fixture(scope="module")
def plummer():
    """The untruncated isotropic Plummer model and its projection."""
    model = analytic_model("plummer")
    projected = projected_dispersion(
        model, dispersion_profile(model, SETTINGS), SETTINGS
    )
    return model, projected


@pytest.fixture(scope="module")
def plummer_table(plummer):
    """Its table of line-of-sight velocity distributions."""
    model, projected = plummer
    return losvd_table(
        model, kept_ranges(model, projected, SETTINGS), projected, SETTINGS
    )


def test_the_default_settings_are_the_working_codes_rules():
    assert ellipse_rules(Settings()) == EllipseRules()


def test_the_rows_take_the_kernel_by_default_and_numpy_as_the_reference():
    assert ellipse_rules(Settings()).kernel
    assert ellipse_rules(Settings(quick=True)).kernel
    assert not ellipse_rules(Settings(observables={"rows_route": "numpy"})).kernel
    with pytest.raises(InputError, match="rows_route"):
        ellipse_rules(Settings(observables={"rows_route": "gpu"}))


def test_the_log_potential_slopes_are_the_hermites_own():
    profile = analytic_model("hernquist", n_nodes=201).profile
    slopes = LogPotentialSlopes(profile)
    x = slopes.x
    # inside each interval, against central differences of ln P in ln r (the
    # second difference carries rounding of 1e-16 / h^2, hence its atol)
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
    # at a row the first derivative is continuous (both sides give the
    # Poisson slope), the second jumps, and each side is its interval's
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
    # f_extrema with its knot-interval bounds against the same route
    # with every knot interval kept, which is the bisection alone: the extrema
    # and V_end to the bit, isotropic, OM I and OM II (the route reads only
    # the sign and r_a of the anisotropy here)
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
    """A uniform sphere with its edge row: the last interval is Kepler's."""
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
    profile = _finite()
    slopes = LogPotentialSlopes(profile)
    assert slopes.x[-1] == pytest.approx(math.log(2.0))
    r_last = float(profile.r[-2])
    r = np.geomspace(r_last * 1.0001, 1.999, 7)
    np.testing.assert_allclose(slopes(np.log(r), 1), -2.0 / (2.0 - r), rtol=1e-12)
    np.testing.assert_allclose(
        slopes(np.log(r), 2), -2.0 * r / (2.0 - r) ** 2, rtol=1e-12
    )
    # the right-hand limit at the last row is Kepler's, the left the Hermite's
    y = math.log(r_last)
    assert float(slopes(y, 1)) == pytest.approx(-2.0 / (2.0 - r_last), rel=1e-12)
    assert float(slopes.left(y, 1)) == pytest.approx(
        float(profile.potential_log_slope(r_last)), rel=1e-12
    )
    assert float(slopes.left(math.log(2.0), 1)) == -np.inf


def test_the_cumulative_df_is_the_lookups_integral():
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
        assert cumulative_df(arrays, kinds, q) == pytest.approx(exact, rel=GOAL)


def _hermite_d(v, h3, h4):
    w = v
    h3_ = (2.0 * w**3 - 3.0 * w) / math.sqrt(3.0)
    h4_ = (4.0 * w**4 - 12.0 * w**2 + 3.0) / (2.0 * math.sqrt(6.0))
    return np.exp(-0.5 * w * w) / math.sqrt(2.0 * math.pi) * (1.0 + h3 * h3_ + h4 * h4_)


def test_gauss_hermite_moments_of_a_gaussian_and_of_a_perturbed_one():
    x, w = np.polynomial.legendre.leggauss(400)
    v, w = 12.0 * x, 12.0 * w
    g = gauss_hermite_moments(v, w, _hermite_d(v, 0.0, 0.0))
    assert abs(g["mean"]) < 1e-14 and g["sigma"] == pytest.approx(1.0, rel=1e-13)
    assert abs(g["h3"]) < 1e-14 and abs(g["h4"]) < 1e-13
    assert g["s4"] == pytest.approx(3.0, rel=1e-13)
    # the relations of Chiang+(2026d) hold to first order in h: off by a part in 1e4 of
    # themselves at h = 1e-5 (2 percent at h4 = 1e-3, a second-order term)
    for h3, h4 in ((1e-5, 0.0), (0.0, 1e-5), (-2e-5, 1.5e-5)):
        g = gauss_hermite_moments(v, w, _hermite_d(v, h3, h4))
        assert g["s3"] == pytest.approx(4.0 * math.sqrt(3.0) * g["h3"], rel=1e-3)
        assert g["s4"] - 3.0 == pytest.approx(8.0 * math.sqrt(6.0) * g["h4"], rel=1e-3)
    # and the coefficients of the perturbation are recovered
    for h3, h4 in ((1e-3, 0.0), (0.0, 1e-3), (-2e-3, 1.5e-3)):
        g = gauss_hermite_moments(v, w, _hermite_d(v, h3, h4))
        assert g["h3"] == pytest.approx(h3, abs=2e-5)
        assert g["h4"] == pytest.approx(h4, abs=2e-5)


def test_rows_of_plummer_agree_with_its_projection(plummer):
    model, projected = plummer
    kept = kept_ranges(model, projected, SETTINGS)
    assert kept.v_rad.shape == (0, 2) and kept.v_k.shape == (0, 2)
    for r_2d in (1e-2, 1.0, 30.0):
        row = losvd_row(model, r_2d, kept, projected, SETTINGS)
        assert abs(row.record["norm_against_surface_density"]) < GOAL
        assert abs(row.record["m2_against_projected_dispersion"]) < GOAL
        assert row.losvd.shape == (1, len(row.v_nodes))
        assert 2.0 * float(np.dot(row.v_weights, row.losvd[0])) == pytest.approx(
            1.0, rel=1e-14
        )
        # mirrored, the row's own mean is zero and its dispersion sigma_los
        v = np.concatenate([-row.v_nodes[::-1], row.v_nodes])
        w = np.concatenate([row.v_weights[::-1], row.v_weights])
        d = np.concatenate([row.losvd[0][::-1], row.losvd[0]])
        g = gauss_hermite_moments(v, w, d)
        sigma_los = float(projected.read(np.array([r_2d]))[1][0, 0])
        assert abs(g["mean"]) < 1e-12 * sigma_los
        assert g["sigma"] == pytest.approx(sigma_los, rel=GOAL)
        assert abs(g["h3"]) < 1e-12


def test_a_row_states_its_norm_through_the_dfs_own_factor(plummer, monkeypatch):
    """A DF read with a constant factor (its largest value above the
    ceiling) leaves the row's record and its normalized D where they were."""
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
    """The lookups of the ellipse route prepared with the floor K_0 = `k_0`
    and the rule "zero" below it."""
    real = _ellipse_model.prepare_lookup

    def prepare(table, form, component):
        arrays, kinds = real(table, form, component)
        params = np.array(arrays.params, dtype=np.float64)
        params[_ellipse_model.K_0] = k_0
        return arrays._replace(params=params), kinds._replace(below_floor="zero")

    monkeypatch.setattr(_ellipse_model, "prepare_lookup", prepare)


def test_the_rows_end_where_the_potential_reaches_the_dfs_floor(plummer, monkeypatch):
    """Where the DF is zero below a floor K_0 the potential reaches inside
    the profile, the rows end at the radius where P = K_0, read before any
    row is built: no row and no line of sight beyond it."""
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


def test_a_floor_inside_the_first_row_is_refused_by_name(plummer, monkeypatch):
    """Where the potential reaches the DF's floor at or inside the profile's
    first row, the model is refused naming the floor and that row."""
    from pianistpy.exceptions import ResultError

    model, _ = plummer
    profile = model.profile
    k_0 = float(profile.potential(profile.r[:1])[0]) * (1.0 + 1e-9)
    _floor_at(monkeypatch, k_0)
    message = (
        f"floor K_0 = {k_0:.6e} (the rule 'zero'), which the potential reaches "
        f"at or inside the profile's first row r = {float(profile.r[0]):.6e}"
    )
    with pytest.raises(ResultError, match=re.escape(message)):
        _ellipse_model.floor_radius(model)
    with pytest.raises(ResultError, match=re.escape(message)):
        _losvd_stages.base_rows(model, SETTINGS)


def test_table_and_aperture_of_plummer(plummer, plummer_table):
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
    # the projected mass inside R of Plummer, 4 pi R^2 / (1 + R^2) in
    # internal units (Sigma = 4 (1 + R^2)^-2)
    assert result.mass[0] == pytest.approx(2.0 * math.pi, rel=GOAL)


def test_a_row_whose_t_rule_did_not_converge_refuses_the_table(plummer, plummer_table):
    # the same table with one row's rule along the line of sight marked as
    # having run out of doublings: the report is refused by that check, at
    # that row, whatever the other checks read
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
    # the dispersion and the distribution, a circle and a rectangle: one
    # statement, each naming the range of the tables it reads
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
        "The aperture lies outside the tabulated projected radii [...]: "
        "it encloses no surface density."
    }, messages


def test_a_shared_scope_starts_one_pool_a_key(monkeypatch):
    # one worker, so that the test pays one process start
    monkeypatch.setattr(_losvd_stages, "get_workers", lambda: 1)
    with shared_pool():
        with start_pool(None, key="a") as first:
            assert first.map(abs, [-1, 2]) == [1, 2]
        with start_pool(None, key="a") as again:
            assert again is first
        with start_pool(None, key="b") as other:
            assert other is not first
    # the scope has closed its pools; outside one, a pool is the caller's
    with pytest.raises(ValueError):
        first.map(abs, [1])
    assert not _losvd_stages._SCOPES


def test_the_gauss_hermite_moments_read_across_the_table(plummer, plummer_table):
    import jax
    import jax.numpy as jnp

    _, projected = plummer
    gh = gauss_hermite_table(plummer_table)
    r = gh.r_2d
    # at the rows, the rows' own moments: to the bit on the host; traced, to
    # the bit on the CPU and elsewhere at rounding, the platform's own
    # arithmetic (measured: the mean and h3 one value of 711 apart on a GPU, at
    # 4.4e-16 relative, two epsilon, which is up to 4 ulp of the value)
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
    # between rows the traced reading is the host's; sigma is the projected
    # dispersion's, h4 small for an isotropic Plummer sphere
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
    # beyond the last row nothing; inside the first, refused
    assert gh.read(np.array([2.0 * r[-1]]))["sigma"][0] == 0.0
    with pytest.raises(InputError):
        gh.read(np.array([0.5 * r[0]]))


def test_a_pools_workers_run_on_the_cpu(monkeypatch):
    # the main process's platform names the GPU; its workers take the CPU at
    # spawn, and the main process's variables are as they were after it
    import os

    from pianistpy import platform
    from pianistpy.observables._losvd_table import new_pool

    monkeypatch.setenv("JAX_PLATFORMS", "cuda,cpu")
    monkeypatch.setenv("PIANISTPY_PLATFORM", "gpu")
    with new_pool(1, None) as pool:
        assert pool.apply(platform) == "cpu"
    assert os.environ["JAX_PLATFORMS"] == "cuda,cpu"
    assert os.environ["PIANISTPY_PLATFORM"] == "gpu"


def test_a_second_aperture_on_a_table_forms_no_extrema(
    plummer, plummer_table, monkeypatch
):
    # the extrema of the line of sight are kept on the table: one
    # worker forms them in the main process, more form them in the pool, and
    # a second call on the same table and radii forms none; the same bytes
    import dataclasses

    from pianistpy import runtime
    from pianistpy.observables import stages
    from pianistpy.observables._ellipse import EllipseRoute

    model, projected = plummer
    aperture = Aperture.circle(1.0)
    fields = ("v_nodes", "v_weights", "v_breaks", "losvd", "mass")
    saved = runtime._workers
    try:
        runtime.set_workers(1)
        alone = aperture_losvd(
            model,
            dataclasses.replace(plummer_table, extrema={}),
            projected,
            aperture,
            SETTINGS,
        )
        runtime.set_workers(2)
        table = dataclasses.replace(plummer_table, extrema={})
        pooled = aperture_losvd(model, table, projected, aperture, SETTINGS)
        assert table.extrema
        # the table keeps the one pool the aperture started
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
        runtime.set_workers(saved)
    assert not formed and not started
    for result in (pooled, again):
        for name in fields:
            a, b = np.asarray(getattr(alone, name)), np.asarray(getattr(result, name))
            assert a.tobytes() == b.tobytes(), name
        assert result.record == alone.record


def test_a_layout_that_returns_between_two_samples_is_found():
    # near the edge of this member the layout runs 3, 2, 4, 2 between
    # the samples at R_2D 22.70 and 23.36; each change is an R-break, so the
    # rows of every stretch share their layout (the panel table's rule)
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
    # the lattice merges V_k into V_end between 23.31699 and 23.31700
    assert np.any((breaks > 23.31699) & (breaks < 23.31700))
    code, stretch = route.layout_many(rows), np.searchsorted(breaks, rows)
    for st in np.unique(stretch):
        assert np.unique(code[stretch == st]).size == 1, st
