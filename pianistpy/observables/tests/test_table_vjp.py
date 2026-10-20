"""Tests of the transposes of the DF path: the spline's slopes, the lookup and
the speed distribution, against SciPy and against central differences of
their own forward with every decision held."""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from pianistpy import (
    Anisotropy,
    Component,
    DoublePowerLaw,
    InputError,
    ResultError,
    Settings,
    Truncation,
    compute_df,
    platform,
    solve,
    truncate,
)
from pianistpy.df.lookup import evaluate_lookup, prepare_lookup
from pianistpy.observables._ellipse_model import ellipse_route
from pianistpy.observables._losvd_stages import ellipse_rules
from pianistpy.observables._speed import angle_integral, angle_pieces, speed_model
from pianistpy.observables._table_vjp import (
    _lookup_reader,
    _read_sums,
    _row_by_levels,
    _row_levels,
    _row_support,
    aperture_losvd_vjp,
    lookup_vjp,
    losvd_read_vjp,
    losvd_rows_vjp,
    speed_distribution_vjp,
    spline_slopes_vjp,
)
from pianistpy.observables.stages import (
    _speed_point,
    dispersion_profile,
    kept_ranges,
    losvd_row,
    projected_dispersion,
    speed_distribution,
)

QUICK = Settings(quick=True)
THETA = dict(beta_c=0.25, r_a=5.0)
# the rows' transpose holds each row recomputed level by level to the forward's
# row bit for bit, which a GPU's reduction order does not promise
CPU_ONLY = pytest.mark.skipif(
    platform() != "cpu",
    reason="the transpose holds its recompute to the forward bit for bit, which "
    "a GPU's reduction order does not promise",
)
# the pool's workers and the main process give the same bits only where they
# share one backend and its reduction order: the CPU
SAME_BITS_IN_THE_POOL = pytest.mark.skipif(
    platform() != "cpu",
    reason="the pooled transpose equals the serial one bit for bit only where the "
    "workers and the main process share one backend and its reduction order",
)


@pytest.fixture(scope="module")
def parent():
    return solve(Component(DoublePowerLaw.hernquist()), Truncation(k_t=0.05), QUICK)


def _child(parent, **anisotropy):
    component = Component(profile=parent.profile, anisotropy=Anisotropy(**anisotropy))
    return truncate(compute_df(component, QUICK))


@pytest.fixture(scope="module")
def member(parent):
    """A COM Type I Method II child of the truncated Hernquist model, a member
    of the likelihood's own kind; its table holds no slopes."""
    return _child(parent, **THETA)


def _direction(model, seed):
    """A random direction in the table's values and the edge law's numbers."""
    rng = np.random.default_rng(seed)
    table = model.table
    d_y = np.zeros(table.log_df.shape)
    first, last = table.i_first[0], table.i_last[0]
    d_y[0, first : last + 1] = rng.normal(size=last + 1 - first)
    d_floor = rng.normal(size=len(model.form.below_floor_coefficients[0]))
    return d_y, d_floor


def _moved(model, d_y, d_floor, h):
    table = dataclasses.replace(model.table, log_df=model.table.log_df + h * d_y)
    coefficients = np.asarray(model.form.below_floor_coefficients[0]) + h * d_floor
    form = dataclasses.replace(
        model.form, below_floor_coefficients=(tuple(coefficients),)
    )
    return dataclasses.replace(model, table=table, form=form)


def _richardson(f, h):
    """The central difference of f at 0, extrapolated from steps h and h/2,
    and the extrapolation's own error."""
    coarse = (f(h) - f(-h)) / (2.0 * h)
    fine = (f(0.5 * h) - f(-0.5 * h)) / h
    return (4.0 * fine - coarse) / 3.0, abs(fine - coarse) / 3.0


@pytest.mark.parametrize("n", [4, 5, 9, 300])
def test_the_spline_transpose_is_scipys_slopes_transposed(n):
    rng = np.random.default_rng(n)
    x = np.sort(rng.uniform(-10.0, 3.0, n))
    y, c = rng.normal(size=n), rng.normal(size=n)
    slopes = CubicSpline(x, y, bc_type="not-a-knot")(x, 1)
    lhs = float(c @ slopes)
    assert float(spline_slopes_vjp(x, c) @ y) == pytest.approx(
        lhs, rel=1e-13, abs=1e-13 * float(np.abs(c) @ np.abs(slopes))
    )


def test_the_lookup_transpose_against_differences(member):
    assert not member.table.slopes_exact[0]
    d_y, d_floor = _direction(member, 1)
    rng = np.random.default_rng(2)
    # energies inside the table, below it (the edge law) and above it
    k = np.concatenate(
        [rng.uniform(1e-9, 1e-6, 50), np.geomspace(1e-5, 0.9, 400), [0.949, 0.9499]]
    )
    c = rng.normal(size=k.size)

    def read(h):
        arrays, kinds = prepare_lookup(
            _moved(member, d_y, d_floor, h).table,
            _moved(member, d_y, d_floor, h).form,
        )
        return float(c @ 10.0 ** evaluate_lookup(arrays, kinds, k))

    jd, error = _richardson(read, 1e-4)
    cot = lookup_vjp(member, k, c)
    jtc = float(np.sum(cot.log_df * d_y)) + float(
        np.dot(cot.form["below_floor_coefficients"], d_floor)
    )
    assert abs(jtc - jd) <= 3.0 * error + 1e-10 * abs(jd)


def test_the_lookup_transpose_reads_the_shift_inside_the_exponent(member):
    # a table whose F passes the float64 range (log10 F raised by 400) has a
    # finite transpose with the route's shift inside the exponent, the
    # unshifted table's to rounding; read unshifted it is not finite
    arrays, kinds = prepare_lookup(member.table, member.form, 0)
    leaves, read = _lookup_reader(arrays, kinds)
    table = member.table
    k = np.asarray(table.k[0])[table.i_first[0] : table.i_last[0] + 1]
    k = 0.5 * (k[1:] + k[:-1])
    c = np.random.default_rng(12).normal(size=k.size)
    up = (leaves[0] + 400.0, *leaves[1:])
    base = _read_sums(leaves, read, k, c)
    shifted = _read_sums(up, read, k, c, 0.0, 400.0)
    assert all(np.all(np.isfinite(x)) for x in shifted)
    # the raised exponent rounds at 400 eps, about 2e-13 of F, and some of the
    # leaves' sums cancel: held against the size of the sums, not entry by entry
    for a, b in zip(shifted[:2], base[:2], strict=True):
        np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-11 * np.max(np.abs(b)))
    assert not np.all(np.isfinite(_read_sums(up, read, k, c)[0]))


def test_the_speed_distribution_transpose_against_differences(member):
    r = 1.0
    result = speed_distribution(member, r, QUICK)
    rng = np.random.default_rng(3)
    c = rng.normal(size=result.pdf.shape)
    n_nodes = int(result.record["n_angle_nodes"])
    n_panels = int(result.record["n_angle_panels"])

    def frozen(model):
        sm = speed_model(model)
        psi, xi, scale = _speed_point(model, sm, r)
        v = result.v_nodes
        integral = angle_integral(sm, psi, xi, v, n_nodes, n_panels)
        return scale * v ** (2.0 - 2.0 * sm.b) * integral

    def with_theta(name, h):
        a = dataclasses.replace(
            member.components[0].anisotropy, **{name: THETA[name] + h}
        )
        component = dataclasses.replace(member.components[0], anisotropy=a)
        return dataclasses.replace(member, components=(component,))

    # the table: <c, J d> by differences, <J^T c, d> by the transpose
    d_y, d_floor = _direction(member, 4)
    jd, error = _richardson(
        lambda h: float(c @ frozen(_moved(member, d_y, d_floor, h))), 1e-4
    )
    cot, explicit = speed_distribution_vjp(result, c, ("beta_c",))
    jtc = float(np.sum(cot.log_df * d_y)) + float(
        np.dot(cot.form["below_floor_coefficients"], d_floor)
    )
    assert abs(jtc - jd) <= 3.0 * error + 1e-10 * abs(jd)
    # beta_c at a fixed table
    jb, error = _richardson(lambda h: float(c @ frozen(with_theta("beta_c", h))), 1e-4)
    assert abs(explicit["beta_c"] - jb) <= 3.0 * error + 1e-10 * abs(jb)
    # r_a on the speeds whose pieces the steps leave as they are: elsewhere a
    # piece appears or leaves with the step, and the difference is no slope

    def pieces(model):
        sm = speed_model(model)
        psi, xi, _ = _speed_point(model, sm, r)
        return [tuple(angle_pieces(sm, psi, xi, float(v))[3]) for v in result.v_nodes]

    h = 1e-3
    keep = np.ones(c.size, dtype=bool)
    here = pieces(member)
    for step in (h, -h, 0.5 * h, -0.5 * h):
        keep &= np.array(
            [a == b for a, b in zip(here, pieces(with_theta("r_a", step)), strict=True)]
        )
    assert keep.sum() > c.size // 2
    cm = np.where(keep, c, 0.0)
    _, explicit = speed_distribution_vjp(result, cm, ("r_a",))
    ja, error = _richardson(lambda s: float(cm @ frozen(with_theta("r_a", s))), h)
    assert abs(explicit["r_a"] - ja) <= 3.0 * error + 1e-10 * abs(ja)


def test_a_distribution_not_of_its_rule_and_com_ii_are_refused(member):
    result = speed_distribution(member, 1.0, QUICK)
    altered = dataclasses.replace(result, pdf=result.pdf * (1.0 + 1e-15))
    with pytest.raises(ResultError, match="not the one given"):
        speed_distribution_vjp(altered, np.ones_like(result.pdf))
    a = Anisotropy(beta_c=-0.25, r_a=100.0, kind="II")
    com_ii = dataclasses.replace(
        member,
        components=(dataclasses.replace(member.components[0], anisotropy=a),),
    )
    with pytest.raises(InputError, match="COM Type II"):
        speed_distribution_vjp(
            dataclasses.replace(result, model=com_ii),
            np.ones_like(result.pdf),
            ("r_a",),
        )


@CPU_ONLY
@pytest.mark.parametrize("family", ["COM_I", "OM_I"])
def test_a_rows_transpose_against_differences(parent, member, family):
    # one row each of the two forms: COM I reads F with its L factor, OM I
    # reads G, the cumulative integral of F
    model = member if family == "COM_I" else _child(parent, r_a=5.0)
    projected = projected_dispersion(model, dispersion_profile(model, QUICK), QUICK)
    kept = kept_ranges(model, projected, QUICK)
    row = losvd_row(model, 1.0, kept, projected, QUICK)
    table = SimpleNamespace(rows=(row,), kept=kept)
    c = np.random.default_rng(5).normal(size=row.losvd[0].shape)
    cot = losvd_rows_vjp(model, table, QUICK, [c])
    rules = ellipse_rules(QUICK)
    base = ellipse_route(model, rules)
    base.vrad_kept, base.vk_kept = kept.v_rad, kept.v_k
    rule, _ = base.rule(1.0)
    _, va, inv, trace = _row_levels(base, 1.0, row.v_nodes, row.v_weights, rule)
    d_y, d_floor = _direction(model, 6)
    d_floor[0] = 0.0  # the edge law's exponent, which the map toward K = 0 reads, held

    def pairing(h):
        route = ellipse_route(_moved(model, d_y, d_floor, h), rules)
        route.vrad_kept, route.vk_kept = kept.v_rad, kept.v_k
        d = _row_by_levels(route, 1.0, va, rule, trace)[inv]
        return float(c @ (d / (2.0 * float(np.dot(row.v_weights, d)))))

    jd, error = _richardson(pairing, 1e-4)
    jtc = float(np.sum(cot.log_df * d_y)) + float(
        np.dot(cot.form["below_floor_coefficients"], d_floor)
    )
    assert abs(jtc - jd) <= 3.0 * error + 1e-10 * abs(jd)


def test_a_row_not_of_its_rule_is_refused(member):
    projected = projected_dispersion(member, dispersion_profile(member, QUICK), QUICK)
    kept = kept_ranges(member, projected, QUICK)
    row = losvd_row(member, 1.0, kept, projected, QUICK)
    altered = dataclasses.replace(row, losvd=row.losvd * (1.0 + 1e-15))
    table = SimpleNamespace(rows=(altered,), kept=kept)
    with pytest.raises(ResultError, match="not the row stored"):
        losvd_rows_vjp(member, table, QUICK, [np.ones(row.losvd[0].shape)])


@pytest.fixture(scope="module")
def om_table(parent):
    """The LosTable of an OM Type I Method II child."""
    from pianistpy import LosTable

    return LosTable(_child(parent, r_a=5.0))


def test_the_reading_transpose_is_the_reading_transposed(om_table):
    # the reading of a table (losvd) is linear in its rows: <c, A d> by the
    # forward on rows moved by d, against <A^T c, d> by the transpose
    from pianistpy.observables.core import _read_losvd

    part = om_table.parts[0]
    rng = np.random.default_rng(8)
    x = rng.uniform(0.05, 10.0, 60)
    u = rng.uniform(-1.2, 1.2, 60)
    c = rng.normal(size=60)
    back = losvd_read_vjp(part, x, u, c)
    deltas = [rng.normal(size=row.losvd[0].shape) for row in part.table.rows]
    rows = tuple(
        dataclasses.replace(row, losvd=row.losvd + d[np.newaxis, :])
        for row, d in zip(part.table.rows, deltas, strict=True)
    )
    moved = SimpleNamespace(
        model=part.model, table=dataclasses.replace(part.table, rows=rows)
    )
    forward = float(c @ (_read_losvd(moved, x, u) - _read_losvd(part, x, u)))
    transpose = sum(float(b @ d) for b, d in zip(back, deltas, strict=True))
    assert transpose == pytest.approx(forward, rel=1e-11)


def test_the_aperture_transpose_is_the_aperture_transposed(om_table):
    # an aperture's values are linear in the rows through the reads its
    # record states (radii, weights and breaks): <c, A d> by those reads on
    # rows moved by d, against <A^T c, d> by the transpose, which holds the
    # recomputed aperture to the stage's own first
    from pianistpy import Aperture
    from pianistpy.observables._aperture_losvd import aperture_losvd_panels
    from pianistpy.observables._losvd_stages import table_stencil
    from pianistpy.observables._projection import surface_density_reader
    from pianistpy.observables.stages import INNER_DISK_TOLERANCE, aperture_losvd

    model, table, projected = om_table.model, om_table.table, om_table.projected
    aperture = Aperture.circle(1.0)
    result = aperture_losvd(model, table, projected, aperture, QUICK)
    rng = np.random.default_rng(9)
    c = rng.normal(size=result.losvd[0].shape)
    back = aperture_losvd_vjp(model, table, projected, aperture, QUICK, result, c)
    route = ellipse_route(model, ellipse_rules(QUICK), with_df=False)
    route.vrad_kept, route.vk_kept = table.kept.v_rad, table.kept.v_k
    surface = surface_density_reader(
        projected.r_2d,
        projected.surface_density[0],
        float(model.profile.r_table_max),
        model.spatially_finite,
    )
    record = []
    v = aperture_losvd_panels(
        surface,
        table_stencil(route, table),
        route,
        aperture,
        int(QUICK.observables["n_aperture_nodes"]),
        int(QUICK.observables["aperture_nodes_per_decade"]),
        INNER_DISK_TOLERANCE,
        record=record,
    )[0]
    deltas = [rng.normal(size=row.losvd[0].shape) for row in table.rows]
    rows = tuple(
        dataclasses.replace(row, losvd=row.losvd + d[np.newaxis, :])
        for row, d in zip(table.rows, deltas, strict=True)
    )

    def read(stencil):
        return np.array(
            [
                float(
                    np.dot(
                        w,
                        stencil.at_pairs(
                            route, r, np.full(len(r), float(v[j])), curves=cv
                        ),
                    )
                )
                for j, (r, w, cv) in enumerate(record)
            ]
        )

    base = table_stencil(route, table)
    moved = table_stencil(route, dataclasses.replace(table, rows=rows))
    assert np.allclose(read(base), result.losvd[0], rtol=0.0, atol=1e-15)
    forward = float(c @ (read(moved) - read(base)))
    transpose = sum(float(b @ d) for b, d in zip(back, deltas, strict=True))
    assert transpose == pytest.approx(forward, rel=1e-11)


@SAME_BITS_IN_THE_POOL
def test_the_rows_transpose_in_the_pool_is_the_serial_one(om_table):
    # each row by the same function in a worker as in one process, joined in
    # row order: the same to the bit
    from pianistpy import runtime

    table = om_table.table
    n = len(table.rows)
    rng = np.random.default_rng(10)
    c = [None] * n
    for i in (n // 4, n // 2, (3 * n) // 4):
        c[i] = rng.normal(size=table.rows[i].losvd[0].shape)
    saved = runtime._workers
    try:
        runtime.set_workers(1)
        serial = losvd_rows_vjp(om_table.model, table, QUICK, c)
        runtime.set_workers(3)
        pooled = losvd_rows_vjp(om_table.model, table, QUICK, c)
    finally:
        runtime.set_workers(saved)
    for name in ("log_df", "log_df_slope", "log_df_below", "log_df_slope_below"):
        assert getattr(pooled, name).tobytes() == getattr(serial, name).tobytes()
    assert pooled.form.keys() == serial.form.keys()
    for name in serial.form:
        a, b = np.asarray(pooled.form[name]), np.asarray(serial.form[name])
        assert a.tobytes() == b.tobytes()


def test_a_row_of_zero_cotangent_is_a_row_of_none(om_table, monkeypatch):
    # a reading's transpose gives every row a cotangent, zero away from the
    # radii it reads: those rows are not transposed, and add nothing, to the bit
    from pianistpy import runtime
    from pianistpy.observables import _table_vjp

    table = om_table.table
    n = len(table.rows)
    rng = np.random.default_rng(11)
    c = [None] * n
    c[n // 2] = rng.normal(size=table.rows[n // 2].losvd[0].shape)
    zeros = [
        np.zeros(row.losvd[0].shape) if x is None else x
        for row, x in zip(table.rows, c, strict=True)
    ]
    calls = []
    transpose = _table_vjp._row_transpose
    monkeypatch.setattr(
        _table_vjp,
        "_row_transpose",
        lambda route, i, *rest: calls.append(i) or transpose(route, i, *rest),
    )
    saved = runtime._workers
    try:
        runtime.set_workers(1)
        none = losvd_rows_vjp(om_table.model, table, QUICK, c)
        zero = losvd_rows_vjp(om_table.model, table, QUICK, zeros)
    finally:
        runtime.set_workers(saved)
    assert calls == [n // 2, n // 2]
    for name in ("log_df", "log_df_slope", "log_df_below", "log_df_slope_below"):
        assert getattr(zero, name).tobytes() == getattr(none, name).tobytes()
    for name in none.form:
        assert (
            np.asarray(zero.form[name]).tobytes()
            == np.asarray(none.form[name]).tobytes()
        )


@pytest.mark.parametrize("family", ["COM_I", "OM_I"])
def test_the_rows_explicit_part_against_differences(parent, member, family):
    # the rows at a fixed DF table, each at the levels its own rule ended on
    # and at its own t-nodes:
    # each parameter by central differences, and both moved at once along a
    # direction t, <c, J t> against <J^T c, t>. The central differences of
    # the frozen forward scatter by about 1e-6 of their value at steps from
    # 6e-5 to 2e-3 (the forward is not smooth there: Q_c's clamp at the
    # supports' ends), more than the extrapolation's error states; the
    # forward differences converge onto the explicit part, halving with h
    from pianistpy.observables._rows_theta import losvd_rows_explicit

    theta = THETA if family == "COM_I" else dict(r_a=5.0)
    model = member if family == "COM_I" else _child(parent, **theta)
    projected = projected_dispersion(model, dispersion_profile(model, QUICK), QUICK)
    kept = kept_ranges(model, projected, QUICK)
    row = losvd_row(model, 1.0, kept, projected, QUICK)
    table = SimpleNamespace(rows=(row,), kept=kept)
    c = np.random.default_rng(11).normal(size=row.losvd[0].shape)
    names = tuple(theta)
    explicit = losvd_rows_explicit(model, table, QUICK, [c], names)
    rules = ellipse_rules(QUICK)
    base = ellipse_route(model, rules)
    base.vrad_kept, base.vk_kept = kept.v_rad, kept.v_k
    rule, _ = base.rule(1.0)
    _, va, inv, trace = _row_levels(base, 1.0, row.v_nodes, row.v_weights, rule)
    # the t-nodes held: the support in t of each speed is the base route's
    support = _row_support(base, 1.0, va)
    # the steps: R_a's ten times beta_c's, as its scale is
    t = {"beta_c": 0.7, "r_a": -6.0}

    def pairing(step):
        a = dataclasses.replace(
            model.components[0].anisotropy,
            **{k: theta[k] + h for k, h in step.items()},
        )
        moved = dataclasses.replace(
            model, components=(dataclasses.replace(model.components[0], anisotropy=a),)
        )
        route = ellipse_route(moved, rules)
        route.vrad_kept, route.vk_kept = kept.v_rad, kept.v_k
        d = _row_by_levels(route, 1.0, va, rule, trace, support=support)[inv]
        return float(c @ (d / (2.0 * float(np.dot(row.v_weights, d)))))

    steps = {"beta_c": 1e-4, "r_a": 1e-3}
    for name in names:
        jd, error = _richardson(lambda h, name=name: pairing({name: h}), steps[name])
        assert abs(explicit[name] - jd) <= 3.0 * error + 1e-6 * abs(jd)
    jd, error = _richardson(lambda h: pairing({k: h * t[k] for k in names}), 1e-4)
    jtc = sum(explicit[k] * t[k] for k in names)
    assert abs(jtc - jd) <= 3.0 * error + 1e-6 * abs(jd)


def test_the_rows_explicit_part_skips_a_row_of_zero_cotangent(parent, monkeypatch):
    # as the rows' transpose: a row whose cotangent is zero is not recomputed,
    # and the explicit part is that of its None, to the bit
    from pianistpy.observables import _rows_theta
    from pianistpy.observables._rows_theta import losvd_rows_explicit

    model = _child(parent, r_a=5.0)
    projected = projected_dispersion(model, dispersion_profile(model, QUICK), QUICK)
    kept = kept_ranges(model, projected, QUICK)
    row = losvd_row(model, 1.0, kept, projected, QUICK)
    table = SimpleNamespace(rows=(row, row), kept=kept)
    c = np.random.default_rng(11).normal(size=row.losvd[0].shape)
    calls = []
    levels = _rows_theta._row_levels
    monkeypatch.setattr(
        _rows_theta,
        "_row_levels",
        lambda *args: calls.append(1) or levels(*args),
    )
    none = losvd_rows_explicit(model, table, QUICK, [c, None], ("r_a",))
    zero = losvd_rows_explicit(model, table, QUICK, [c, np.zeros_like(c)], ("r_a",))
    assert len(calls) == 2
    assert np.float64(zero["r_a"]).tobytes() == np.float64(none["r_a"]).tobytes()


@pytest.mark.parametrize("rule", ["tail", "zero"])
def test_the_rows_explicit_part_refuses_a_moving_support_edge(member, rule):
    # a DF that diverges at zero energy, or ends at its floor with a positive
    # value, moves the rows' edge where F is not zero: refused by name
    from pianistpy.observables._rows_theta import losvd_rows_explicit

    form = dataclasses.replace(member.form, below_floor=(rule,))
    model = dataclasses.replace(member, form=form)
    table = SimpleNamespace(rows=(), kept=SimpleNamespace(v_rad=None, v_k=None))
    with pytest.raises(InputError, match=f"the rule '{rule}'"):
        losvd_rows_explicit(model, table, QUICK, [], ("beta_c",))
