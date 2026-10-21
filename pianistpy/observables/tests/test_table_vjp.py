r"""Tests of the transposes (vector-Jacobian products) of the DF path.

The gradients of `Differentiable` call the transposes in
`pianistpy.observables._table_vjp` and the explicit parts in
`pianistpy.observables._rows_theta`. The tests cover the transposes of the
spline's slopes, of the DF lookup, of the velocity distribution, of the LOSVD
rows, of the reading of a LOSVD table and of the LOSVD in an aperture.

A transpose :math:`J^{\rm T}` of a forward map with Jacobian :math:`J` is
tested by the identity :math:`\langle c, J d\rangle = \langle J^{\rm T} c,
d\rangle`. In the identity, :math:`c` is a random cotangent on the forward's
output and :math:`d` is a random direction in the forward's input. A linear forward
gives :math:`\langle c, J d\rangle` as the difference of two forward calls.
A nonlinear forward gives :math:`\langle c, J d\rangle` as a central
difference along :math:`d`, extrapolated from the steps :math:`h` and
:math:`h/2` (`_richardson`). Every discrete choice of the forward (a
level, a node set, an angle piece) is held at the choice made for the
unmoved input. The two sides of a nonlinear identity agree within three
times the extrapolation's error estimate plus :math:`10^{-10}` of
:math:`|\langle c, J d\rangle|`. The explicit part of the LOSVD rows uses
:math:`10^{-6}` of :math:`|\langle c, J d\rangle|` in place of
:math:`10^{-10}`. Further tests assert that two computations of one
transpose (pooled and serial, checked and unchecked, cached and fresh)
give the same bits. The transposes are derived in Notes Section T:7.
"""

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
from pianistpy.observables._ellipse import EllipseRoute
from pianistpy.observables._ellipse_model import ellipse_route
from pianistpy.observables._losvd_stages import ellipse_rules
from pianistpy.observables._table_vjp import (
    _lookup_reader,
    _read_sums,
    _row_by_levels,
    _row_levels,
    _row_reads,
    _row_support,
    aperture_losvd_vjp,
    lookup_vjp,
    losvd_read_vjp,
    losvd_rows_vjp,
    spline_slopes_vjp,
    velocity_distribution_vjp,
)
from pianistpy.observables._velocity import angle_integral, angle_pieces, velocity_model
from pianistpy.observables.stages import (
    _velocity_point,
    dispersion_profile,
    kept_ranges,
    losvd_row,
    projected_dispersion,
    velocity_distribution,
)

pytestmark = pytest.mark.slow

QUICK = Settings(quick=True)
# CHECKED is the quick settings with check_rows=True, under which the rows'
# transpose and explicit part check each recomputed row against the stored row.
CHECKED = Settings(quick=True, check_rows=True)
THETA = dict(beta_c=0.25, r_a=5.0)
# The tests marked CPU_ONLY require a row recomputed level by level to equal
# the forward's row bit for bit. The CPU gives equal bits. A GPU's reduction
# order does not guarantee equal bits.
CPU_ONLY = pytest.mark.skipif(
    platform() != "cpu",
    reason="the transpose holds its recompute to the forward bit for bit, which "
    "a GPU's reduction order does not promise",
)
# The pool's processes and the main process give the same bits only where
# they share one backend and its reduction order, which holds on the CPU.
SAME_BITS_IN_THE_POOL = pytest.mark.skipif(
    platform() != "cpu",
    reason="the pooled transpose equals the serial one bit for bit only where the "
    "workers and the main process share one backend and its reduction order",
)


@pytest.fixture(scope="module")
def parent():
    """The Hernquist model truncated at ``k_t = 0.05``, under the quick settings."""
    return solve(Component(DoublePowerLaw.hernquist()), Truncation(k_t=0.05), QUICK)


def _child(parent, **anisotropy):
    """Return the Method II model of the profile of `parent` with the anisotropy
    given: the DF computed under the quick settings, then truncated."""
    component = Component(profile=parent.profile, anisotropy=Anisotropy(**anisotropy))
    return truncate(compute_df(component, QUICK))


@pytest.fixture(scope="module")
def member(parent):
    r"""The Cuddeford-Osipkov-Merritt (COM) Type I Method II model of the
    profile of `parent`, with :math:`\beta_{\rm c} = 0.25` and
    :math:`R_{\rm a} = 5`. The model's DF table has no exact slopes:
    ``table.slopes_exact[0]`` is False."""
    return _child(parent, **THETA)


def _direction(model, seed):
    """Return a random direction (d_y, d_floor) drawn with the seed `seed`.

    d_y has the shape of the DF table's ``log_df`` and is nonzero only on the
    nodes ``i_first[0]`` to ``i_last[0]`` of the table. d_floor has one entry
    per coefficient of the edge law, the DF's form below its floor
    (``form.below_floor_coefficients[0]``).
    """
    rng = np.random.default_rng(seed)
    table = model.table
    d_y = np.zeros(table.log_df.shape)
    first, last = table.i_first[0], table.i_last[0]
    d_y[0, first : last + 1] = rng.normal(size=last + 1 - first)
    d_floor = rng.normal(size=len(model.form.below_floor_coefficients[0]))
    return d_y, d_floor


def _moved(model, d_y, d_floor, h):
    """Return `model` with its table's ``log_df`` moved by ``h * d_y`` and its
    edge law's coefficients moved by ``h * d_floor``."""
    table = dataclasses.replace(model.table, log_df=model.table.log_df + h * d_y)
    coefficients = np.asarray(model.form.below_floor_coefficients[0]) + h * d_floor
    form = dataclasses.replace(
        model.form, below_floor_coefficients=(tuple(coefficients),)
    )
    return dataclasses.replace(model, table=table, form=form)


def _richardson(f, h):
    """Return the central difference of f at 0 extrapolated from the steps h
    and h/2, and the extrapolation's error estimate.

    With D(h) = (f(h) - f(-h)) / (2h), the extrapolation is
    (4 D(h/2) - D(h)) / 3 and the error estimate is |D(h/2) - D(h)| / 3.
    """
    coarse = (f(h) - f(-h)) / (2.0 * h)
    fine = (f(0.5 * h) - f(-0.5 * h)) / h
    return (4.0 * fine - coarse) / 3.0, abs(fine - coarse) / 3.0


@pytest.mark.parametrize("n", [4, 5, 9, 300])
def test_the_spline_transpose_is_scipys_slopes_transposed(n):
    r"""Assert that ``spline_slopes_vjp(x, c) @ y`` equals ``c @ s`` for n
    random knots x, values y and cotangent c. s holds the slopes at the knots
    of SciPy's not-a-knot `CubicSpline` through (x, y). The tolerance is a
    relative :math:`10^{-13}` and an absolute :math:`10^{-13}` times
    ``abs(c) @ abs(s)``."""
    rng = np.random.default_rng(n)
    x = np.sort(rng.uniform(-10.0, 3.0, n))
    y, c = rng.normal(size=n), rng.normal(size=n)
    slopes = CubicSpline(x, y, bc_type="not-a-knot")(x, 1)
    lhs = float(c @ slopes)
    assert float(spline_slopes_vjp(x, c) @ y) == pytest.approx(
        lhs, rel=1e-13, abs=1e-13 * float(np.abs(c) @ np.abs(slopes))
    )


def test_the_lookup_transpose_against_differences(member):
    r"""Assert that the DF table of `member` has no exact slopes, and that
    `lookup_vjp` satisfies :math:`\langle J^{\rm T} c, d\rangle = \langle c,
    J d\rangle` for the lookup's F at 452 energies K. :math:`J` is the
    Jacobian of F at the 452 energies in the table's ``log_df`` and the edge
    law's coefficients, and :math:`d` is the direction of `_direction`. The
    central difference takes the step :math:`10^{-4}`. The two sides agree
    within three times the extrapolation's error estimate plus
    :math:`10^{-10}` of :math:`|\langle c, J d\rangle|`."""
    assert not member.table.slopes_exact[0]
    d_y, d_floor = _direction(member, 1)
    rng = np.random.default_rng(2)
    # The energies K are meant to lie below the table's first node (where the
    # lookup reads the edge law), inside the table, and above the table.
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
    r"""Assert that the lookup's transpose stays finite for a table whose F
    exceeds the float64 range, when F is read with a shift inside the
    exponent.

    The test raises the lookup's first array, log10 F, by 400, at energies
    midway between the table's nodes. `_read_sums` with the shift 400 reads
    F as :math:`10^{\log_{10}F + 400 - 400}` and returns finite sums. The
    first two sums equal the sums of the unraised table within an absolute
    :math:`10^{-11}` times the largest magnitude of the unraised sum.
    `_read_sums` of the raised table without the shift returns a first sum
    with a non-finite entry.
    """
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
    # The raised exponent log10 F + 400 rounds at 400 times the float64
    # epsilon, which is about 2e-13 of F. Terms of some sums cancel, so each
    # sum is compared against an absolute bound scaled by the largest
    # magnitude of the unraised sum, not entry by entry.
    for a, b in zip(shifted[:2], base[:2], strict=True):
        np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-11 * np.max(np.abs(b)))
    assert not np.all(np.isfinite(_read_sums(up, read, k, c)[0]))


def test_the_velocity_distribution_transpose_against_differences(member):
    r"""Assert three identities of `velocity_distribution_vjp` for the velocity
    distribution of `member` at r = 1.

    The forward recomputes the distribution at the result's velocity nodes with
    the angle rule that the result records (n_angle_nodes, n_angle_panels).
    Each identity holds within three times the extrapolation's error
    estimate plus :math:`10^{-10}` of the central difference's magnitude.
    The identities are the following:

    - the cotangent on the DF table, paired with the direction of
      `_direction`, equals the central difference along that direction with
      the step :math:`10^{-4}`;
    - the explicit part in :math:`\beta_{\rm c}` at a fixed DF table equals the
      central difference in :math:`\beta_{\rm c}` with the step :math:`10^{-4}`;
    - the explicit part in :math:`R_{\rm a}` at a fixed DF table equals the central
      difference in :math:`R_{\rm a}` with the step :math:`10^{-3}`, with the
      cotangent set to zero at every velocity whose angle pieces change under a
      step in :math:`R_{\rm a}`.

    The test also asserts that more than half of the velocities keep their angle
    pieces under every step in :math:`R_{\rm a}` that the central difference takes.
    """
    r = 1.0
    result = velocity_distribution(member, r, QUICK)
    rng = np.random.default_rng(3)
    c = rng.normal(size=result.pdf.shape)
    n_nodes = int(result.record["n_angle_nodes"])
    n_panels = int(result.record["n_angle_panels"])

    def frozen(model):
        sm = velocity_model(model)
        psi, xi, scale = _velocity_point(model, sm, r)
        v = result.v_nodes
        integral = angle_integral(sm, psi, xi, v, n_nodes, n_panels)
        return scale * v ** (2.0 - 2.0 * sm.b) * integral

    def with_theta(name, h):
        a = dataclasses.replace(
            member.components[0].anisotropy, **{name: THETA[name] + h}
        )
        component = dataclasses.replace(member.components[0], anisotropy=a)
        return dataclasses.replace(member, components=(component,))

    # The DF table: <c, J d> by central differences, <J^T c, d> by the
    # transpose.
    d_y, d_floor = _direction(member, 4)
    jd, error = _richardson(
        lambda h: float(c @ frozen(_moved(member, d_y, d_floor, h))), 1e-4
    )
    cot, explicit = velocity_distribution_vjp(result, c, ("beta_c",))
    jtc = float(np.sum(cot.log_df * d_y)) + float(
        np.dot(cot.form["below_floor_coefficients"], d_floor)
    )
    assert abs(jtc - jd) <= 3.0 * error + 1e-10 * abs(jd)
    # The explicit part in beta_c, at a fixed DF table.
    jb, error = _richardson(lambda h: float(c @ frozen(with_theta("beta_c", h))), 1e-4)
    assert abs(explicit["beta_c"] - jb) <= 3.0 * error + 1e-10 * abs(jb)
    # The explicit part in R_a, at a fixed DF table. At some velocities a step in
    # R_a adds or removes an angle piece, and the central difference across
    # that step is not a derivative. The cotangent is therefore kept only at
    # the velocities whose angle pieces stay the same under every step taken.

    def pieces(model):
        sm = velocity_model(model)
        psi, xi, _ = _velocity_point(model, sm, r)
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
    _, explicit = velocity_distribution_vjp(result, cm, ("r_a",))
    ja, error = _richardson(lambda s: float(cm @ frozen(with_theta("r_a", s))), h)
    assert abs(explicit["r_a"] - ja) <= 3.0 * error + 1e-10 * abs(ja)


def test_a_distribution_not_of_its_rule_and_com_ii_are_refused(member):
    r"""Assert that `velocity_distribution_vjp` raises ResultError for a velocity
    distribution whose pdf is scaled by :math:`1 + 10^{-15}`, which differs from the
    distribution recomputed with the result's angle rule. Assert also that
    `velocity_distribution_vjp` raises InputError for a derivative in
    :math:`R_{\rm a}` of a COM Type II model."""
    result = velocity_distribution(member, 1.0, QUICK)
    altered = dataclasses.replace(result, pdf=result.pdf * (1.0 + 1e-15))
    with pytest.raises(
        ResultError, match="differs from the velocity distribution given"
    ):
        velocity_distribution_vjp(altered, np.ones_like(result.pdf))
    a = Anisotropy(beta_c=-0.25, r_a=100.0, kind="II")
    com_ii = dataclasses.replace(
        member,
        components=(dataclasses.replace(member.components[0], anisotropy=a),),
    )
    with pytest.raises(InputError, match="COM Type II"):
        velocity_distribution_vjp(
            dataclasses.replace(result, model=com_ii),
            np.ones_like(result.pdf),
            ("r_a",),
        )


@CPU_ONLY
@pytest.mark.parametrize("family", ["COM_I", "OM_I"])
def test_a_rows_transpose_against_differences(parent, member, family):
    r"""Assert that `losvd_rows_vjp` satisfies :math:`\langle J^{\rm T} c,
    d\rangle = \langle c, J d\rangle` for one LOSVD row at
    :math:`R_{\rm 2D} = 1`.

    :math:`J` is the Jacobian of the normalized row in the DF table's
    ``log_df`` and the edge law's coefficients, and :math:`d` is the direction
    of `_direction` with its entry for the edge law's exponent set to zero.
    The forward recomputes the row with the velocity levels recorded in the
    unmoved row's trace (`_row_by_levels`) and normalizes the row as
    ``D / (2 w @ D)``. The central difference takes the step
    :math:`10^{-4}`. The two sides agree within three times the
    extrapolation's error estimate plus :math:`10^{-10}` of
    :math:`|\langle c, J d\rangle|`. The test runs on a COM Type I model,
    whose row reads F with its factor in the angular momentum L, and on an
    OM Type I model, whose row reads G, the cumulative integral of F.
    """
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
    # The edge law's exponent, d_floor[0], is held fixed, because the route's
    # map toward K = 0 reads that exponent.
    d_floor[0] = 0.0

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
    """Assert that `losvd_rows_vjp` with check_rows=True raises ResultError
    for a LOSVD row whose values are scaled by :math:`1 + 10^{-15}`. The
    check compares the row recomputed by the row's own rule with the row
    stored, bit for bit."""
    projected = projected_dispersion(member, dispersion_profile(member, QUICK), QUICK)
    kept = kept_ranges(member, projected, QUICK)
    row = losvd_row(member, 1.0, kept, projected, QUICK)
    altered = dataclasses.replace(row, losvd=row.losvd * (1.0 + 1e-15))
    table = SimpleNamespace(rows=(altered,), kept=kept)
    with pytest.raises(ResultError, match="differs from the stored distribution"):
        losvd_rows_vjp(member, table, CHECKED, [np.ones(row.losvd[0].shape)])


@pytest.fixture(scope="module")
def om_table(parent):
    r"""The LosTable of the OM Type I Method II model of the profile of
    `parent`, with :math:`R_{\rm a} = 5`."""
    from pianistpy import LosTable

    return LosTable(_child(parent, r_a=5.0))


def test_the_reading_transpose_is_the_reading_transposed(om_table):
    r"""Assert that `losvd_read_vjp` satisfies :math:`\langle A^{\rm T} c,
    d\rangle = \langle c, A d\rangle` to a relative :math:`10^{-11}`.

    :math:`A` is the reading of the LOSVD table of `om_table` (`losvd`) at
    60 random pairs of projected radius and velocity, a map linear in the rows'
    values. :math:`d` is a random change of every row's values. The forward
    side is the reading of the table with the rows moved by :math:`d` minus
    the reading of the unmoved table.
    """
    from pianistpy.observables.core import _read_losvd, _reader_of

    part = om_table.parts[0]
    rng = np.random.default_rng(8)
    x = rng.uniform(0.05, 10.0, 60)
    u = rng.uniform(-1.2, 1.2, 60)
    c = rng.normal(size=60)
    back = losvd_read_vjp(_reader_of(part), x, u, c)
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
    r"""Assert that `aperture_losvd_vjp` satisfies :math:`\langle A^{\rm T}
    c, d\rangle = \langle c, A d\rangle` to a relative :math:`10^{-11}`.

    :math:`A` maps the LOSVD rows of `om_table` to the LOSVD in a circular
    aperture of radius 1. :math:`A` is linear in the rows' values through the
    reads that the aperture's record lists at each velocity: radii, weights and
    curves. :math:`d` is a random change of every row's values. The forward
    side is the record's reads on the rows moved by :math:`d` minus the
    record's reads on the unmoved rows. The test first asserts that the
    record's reads on the unmoved rows equal the LOSVD of `aperture_losvd`
    to an absolute :math:`10^{-15}`.
    """
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
    """Assert that `losvd_rows_vjp` on 3 processes returns the same bits as
    `losvd_rows_vjp` on 1 process, in every array of the DF table's
    cotangent and every entry of the cotangent's forms. The cotangent is
    nonzero on three rows of the n rows of `om_table`, the rows n // 4,
    n // 2 and 3n // 4. A pool process transposes a row by the same function
    as the single process, and the call sums the rows' arrays in row
    order."""
    from pianistpy import runtime

    table = om_table.table
    n = len(table.rows)
    rng = np.random.default_rng(10)
    c = [None] * n
    for i in (n // 4, n // 2, (3 * n) // 4):
        c[i] = rng.normal(size=table.rows[i].losvd[0].shape)
    saved = runtime._cores
    try:
        runtime.set_cores(1)
        serial = losvd_rows_vjp(om_table.model, table, QUICK, c)
        runtime.set_cores(3)
        pooled = losvd_rows_vjp(om_table.model, table, QUICK, c)
    finally:
        runtime.set_cores(saved)
    for name in ("log_df", "log_df_slope", "log_df_below", "log_df_slope_below"):
        assert getattr(pooled, name).tobytes() == getattr(serial, name).tobytes()
    assert pooled.form.keys() == serial.form.keys()
    for name in serial.form:
        a, b = np.asarray(pooled.form[name]), np.asarray(serial.form[name])
        assert a.tobytes() == b.tobytes()


def test_the_rows_transpose_pool_raises_when_a_worker_ends(om_table, monkeypatch):
    """Assert that `losvd_rows_vjp` on 2 processes raises WorkerError within
    BOUND seconds (`_pools.bounded`) when the first process of the call's
    pool has been killed, as the kernel kills a process when the machine
    runs out of memory (`_pools.end_first_worker`). Assert also that the
    call started exactly one pool."""
    from pianistpy import runtime
    from pianistpy.exceptions import WorkerError
    from pianistpy.observables import _table_vjp

    from ._pools import bounded, end_first_worker

    table = om_table.table
    n = len(table.rows)
    rng = np.random.default_rng(13)
    c = [None] * n
    for i in (n // 4, n // 2, (3 * n) // 4):
        c[i] = rng.normal(size=table.rows[i].losvd[0].shape)
    started = end_first_worker(_table_vjp, monkeypatch)
    saved = runtime._cores
    try:
        runtime.set_cores(2)
        with bounded(), pytest.raises(WorkerError):
            losvd_rows_vjp(om_table.model, table, QUICK, c)
    finally:
        runtime.set_cores(saved)
    assert len(started) == 1


def test_a_row_of_zero_cotangent_is_a_row_of_none(om_table, monkeypatch):
    """Assert that `losvd_rows_vjp` transposes no row whose cotangent is all
    zeros, as for a row whose cotangent is None.

    The cotangent is nonzero on the middle row of `om_table` alone. The call
    with None on every other row and the call with zero arrays on every
    other row each run `_row_transpose` on the middle row only, and the two
    calls return the same bits. A reading's transpose (`losvd_read_vjp`)
    gives every row a cotangent, which is zero on the rows away from the
    radii read.
    """
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
    saved = runtime._cores
    try:
        runtime.set_cores(1)
        none = losvd_rows_vjp(om_table.model, table, QUICK, c)
        zero = losvd_rows_vjp(om_table.model, table, QUICK, zeros)
    finally:
        runtime.set_cores(saved)
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
    r"""Assert that the explicit part `losvd_rows_explicit` of one LOSVD row
    at :math:`R_{\rm 2D} = 1` equals central differences in the anisotropy
    parameters at a fixed DF table.

    The forward recomputes the row with each velocity at the level recorded in
    the unmoved row's trace and with the support in t (the t-nodes) of the
    unmoved model. The explicit part in each parameter equals the central
    difference in that parameter, with the step :math:`10^{-4}` in
    :math:`\beta_{\rm c}` and :math:`10^{-3}` in :math:`R_{\rm a}`. The explicit part
    paired with the direction t = (0.7, -6.0) in :math:`(\beta_{\rm c}, R_{\rm a})`
    equals the central difference along t with the step :math:`10^{-4}`.
    Each comparison holds within three times the extrapolation's error
    estimate plus :math:`10^{-6}` of the central difference's magnitude. The
    test runs on a COM Type I model, in :math:`\beta_{\rm c}` and :math:`R_{\rm a}`, and
    on an OM Type I model, in :math:`R_{\rm a}`.
    """
    # The relative bound 1e-6 is looser than the 1e-10 of the DF-table tests.
    # The frozen forward is not smooth in the anisotropy parameters, because
    # Q_c is clamped at the ends of the velocities' supports. The central
    # differences therefore scatter by more than the extrapolation's error
    # estimate.
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
    # The support in t of each velocity is held at the unmoved model's route, so
    # the t-nodes do not move with a step.
    support = _row_support(base, 1.0, va)
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

    # The step in R_a is ten times the step in beta_c, following the larger
    # scale of R_a.
    steps = {"beta_c": 1e-4, "r_a": 1e-3}
    for name in names:
        jd, error = _richardson(lambda h, name=name: pairing({name: h}), steps[name])
        assert abs(explicit[name] - jd) <= 3.0 * error + 1e-6 * abs(jd)
    jd, error = _richardson(lambda h: pairing({k: h * t[k] for k in names}), 1e-4)
    jtc = sum(explicit[k] * t[k] for k in names)
    assert abs(jtc - jd) <= 3.0 * error + 1e-6 * abs(jd)


def test_the_rows_explicit_part_skips_a_row_of_zero_cotangent(parent, monkeypatch):
    r"""Assert that `losvd_rows_explicit` recomputes no row whose cotangent is
    all zeros, as for a row whose cotangent is None.

    The table holds two copies of one LOSVD row, and the cotangent is
    nonzero on the first copy. The call with None on the second copy and the
    call with zeros on the second copy run `_row_reads` once each, and the
    two explicit parts in :math:`R_{\rm a}` have the same bits.
    """
    from pianistpy.observables import _rows_theta
    from pianistpy.observables._rows_theta import losvd_rows_explicit

    model = _child(parent, r_a=5.0)
    projected = projected_dispersion(model, dispersion_profile(model, QUICK), QUICK)
    kept = kept_ranges(model, projected, QUICK)
    row = losvd_row(model, 1.0, kept, projected, QUICK)
    table = SimpleNamespace(rows=(row, row), kept=kept)
    c = np.random.default_rng(11).normal(size=row.losvd[0].shape)
    calls = []
    reads = _rows_theta._row_reads
    monkeypatch.setattr(
        _rows_theta,
        "_row_reads",
        lambda *args: calls.append(1) or reads(*args),
    )
    none = losvd_rows_explicit(model, table, QUICK, [c, None], ("r_a",))
    zero = losvd_rows_explicit(model, table, QUICK, [c, np.zeros_like(c)], ("r_a",))
    assert len(calls) == 2
    assert np.float64(zero["r_a"]).tobytes() == np.float64(none["r_a"]).tobytes()


def test_the_rows_explicit_part_forms_g_once_to_the_bit(parent, monkeypatch):
    r"""Assert that the branches of G formed once give the same bits as the
    branches of G formed at every read.

    `_g_twin` returns a function that forms the branches of G from the
    lookup's arrays alone, and a function that reads G from the branches.
    At 50 values of Q_c that depend on :math:`R_{\rm a}`, JAX's value and
    derivative in :math:`R_{\rm a}` of the sum of G have the same bits with the
    branches formed once and with the branches formed inside the traced
    function. The explicit part in :math:`R_{\rm a}` of one LOSVD row at
    :math:`R_{\rm 2D} = 1` also has the same bits with
    `_rows_theta._g_twin` replaced by a version that forms the branches at
    every read.
    """
    import jax
    import jax.numpy as jnp

    from pianistpy.observables import _rows_theta
    from pianistpy.observables._rows_theta import losvd_rows_explicit
    from pianistpy.observables._table_vjp import _g_twin, _lookup_parts

    model = _child(parent, r_a=5.0)
    route = ellipse_route(model, ellipse_rules(QUICK))
    edf = route._df()
    tables, read = _g_twin(model, edf, edf.F.shift)
    leaves = _lookup_parts(model)[2]
    branches = tables(*leaves)
    q = np.geomspace(1e-3, 0.99, 50) * edf.Qend

    def every_read(ra):
        return jnp.sum(read(tables(*leaves), q / (1.0 + (2.0 / ra) ** 2)))

    def once(ra):
        return jnp.sum(read(branches, q / (1.0 + (2.0 / ra) ** 2)))

    for a, b in zip(
        jax.value_and_grad(every_read)(5.0), jax.value_and_grad(once)(5.0), strict=True
    ):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
    projected = projected_dispersion(model, dispersion_profile(model, QUICK), QUICK)
    kept = kept_ranges(model, projected, QUICK)
    row = losvd_row(model, 1.0, kept, projected, QUICK)
    table = SimpleNamespace(rows=(row,), kept=kept)
    c = np.random.default_rng(12).normal(size=row.losvd[0].shape)
    held = losvd_rows_explicit(model, table, QUICK, [c], ("r_a",))

    def g_every_read(*args):
        tables, read = _g_twin(*args)
        return (lambda *a: a), (lambda a, q, d=None: read(tables(*a), q, d))

    monkeypatch.setattr(_rows_theta, "_g_twin", g_every_read)
    formed = losvd_rows_explicit(model, table, QUICK, [c], ("r_a",))
    assert np.float64(held["r_a"]).tobytes() == np.float64(formed["r_a"]).tobytes()


@pytest.mark.parametrize("rule", ["tail", "zero"])
def test_the_rows_explicit_part_refuses_a_moving_support_edge(member, rule):
    """Assert that `losvd_rows_explicit` raises InputError, with a message
    that gives the below-floor rule, for `member` with its below-floor rule
    set to 'tail' or to 'zero'.

    Under the rule 'tail' the DF diverges at zero energy. Under the rule
    'zero' the DF of `member` ends at its floor K_0 with a positive value.
    Under either rule the support edge of the rows moves where F is nonzero,
    and `losvd_rows_explicit` does not carry the term of the moving edge.
    """
    from pianistpy.observables._rows_theta import losvd_rows_explicit

    form = dataclasses.replace(member.form, below_floor=(rule,))
    model = dataclasses.replace(member, form=form)
    table = SimpleNamespace(rows=(), kept=SimpleNamespace(v_rad=None, v_k=None))
    with pytest.raises(InputError, match=f"the rule '{rule}'"):
        losvd_rows_explicit(model, table, QUICK, [], ("beta_c",))


def _held_row(model):
    r"""Return the LOSVD row of `model` at :math:`R_{\rm 2D} = 1` under the quick
    settings, and a table that holds that row alone with the model's kept
    ranges."""
    projected = projected_dispersion(model, dispersion_profile(model, QUICK), QUICK)
    kept = kept_ranges(model, projected, QUICK)
    row = losvd_row(model, 1.0, kept, projected, QUICK)
    return row, SimpleNamespace(rows=(row,), kept=kept)


def _assert_same_reads(a, b):
    """Assert that two collections of a row's reads, `a` and `b`, have the
    same bits.

    The reads of F ("F" entries) and the reads of G ("G" entries) are each
    compared after joining in order. The pairs of each level ("P" entries)
    are compared one by one: the velocities, the weights, the radius, the rule,
    the arrays u and v, the groups and the node counts chosen.
    """
    for kind in ("F", "G"):
        for j in (1, 2, 3):
            x = [np.asarray(e[j]).ravel() for e in a if e[0] == kind]
            y = [np.asarray(e[j]).ravel() for e in b if e[0] == kind]
            x = np.concatenate(x) if x else np.zeros(0)
            y = np.concatenate(y) if y else np.zeros(0)
            assert x.tobytes() == y.tobytes()
    pa = [e for e in a if e[0] == "P"]
    pb = [e for e in b if e[0] == "P"]
    assert len(pa) == len(pb)
    for (_, s_a, p_a, w_a), (_, s_b, p_b, w_b) in zip(pa, pb, strict=True):
        assert s_a.tobytes() == s_b.tobytes() and w_a.tobytes() == w_b.tobytes()
        assert p_a["r2d"] == p_b["r2d"] and p_a["rule"] == p_b["rule"]
        assert p_a["u"].tobytes() == p_b["u"].tobytes()
        assert p_a["v"].tobytes() == p_b["v"].tobytes()
        assert [(g.tolist(), k) for g, k in p_a["groups"]] == [
            (g.tolist(), k) for g, k in p_b["groups"]
        ]

        def counts(p):
            return {
                int(i): chosen[n].tobytes()
                for at, chosen in p["counts"]
                for n, i in enumerate(at)
            }

        assert counts(p_a) == counts(p_b)


@CPU_ONLY
@pytest.mark.parametrize("family", ["COM_I", "OM_I"])
def test_a_row_is_run_once_a_gradient_with_the_reads_of_its_levels(
    parent, member, family, monkeypatch
):
    r"""Assert that `losvd_rows_vjp` and `losvd_rows_explicit` each run a row's
    rule once with check_rows=False, and give the same bits as with
    check_rows=True.

    For one LOSVD row at :math:`R_{\rm 2D} = 1`, `_row_reads` runs the row's
    rule once with the reads collected. `_row_reads` returns the same row,
    distinct velocities and inverse map as `_row_levels`, and the same reads as
    `_row_by_levels` run on the levels of `_row_levels`, all bit for bit. With
    check_rows=False, the two calls together run `EllipseRoute.row` twice
    and `_row_by_levels` never. With check_rows=True, each call runs
    `EllipseRoute.row` once and then `_row_by_levels` once. The DF table's
    cotangent and the explicit part have the same bits under the two
    settings. The test runs on a COM Type I model and an OM Type I model.
    """
    from pianistpy import runtime
    from pianistpy.observables import _rows_theta, _table_vjp
    from pianistpy.observables._rows_theta import losvd_rows_explicit

    model = member if family == "COM_I" else _child(parent, r_a=5.0)
    names = ("beta_c", "r_a") if family == "COM_I" else ("r_a",)
    row, table = _held_row(model)
    route = ellipse_route(model, ellipse_rules(QUICK))
    route.vrad_kept, route.vk_kept = table.kept.v_rad, table.kept.v_k
    rule, _ = route.rule(1.0)
    d, va, inv, trace = _row_levels(route, 1.0, row.v_nodes, row.v_weights, rule)
    levels = []
    _row_by_levels(route, 1.0, va, rule, trace, levels)
    d_once, va_once, inv_once, once = _row_reads(
        route, 1.0, row.v_nodes, row.v_weights, rule
    )
    assert d_once.tobytes() == d.tobytes() and va_once.tobytes() == va.tobytes()
    assert inv_once.tobytes() == inv.tobytes()
    assert any(e[0] == "P" for e in once)
    _assert_same_reads(once, levels)
    # Every call of EllipseRoute.row and of _row_by_levels is recorded in
    # `calls`.
    calls = []
    run = EllipseRoute.row
    monkeypatch.setattr(
        EllipseRoute,
        "row",
        lambda self, *a, **k: calls.append("row") or run(self, *a, **k),
    )
    for module in (_table_vjp, _rows_theta):
        monkeypatch.setattr(
            module,
            "_row_by_levels",
            lambda *a, f=module._row_by_levels, **k: (
                calls.append("levels") or f(*a, **k)
            ),
        )
    c = np.random.default_rng(13).normal(size=row.losvd[0].shape)
    out = {}
    saved = runtime._cores
    try:
        runtime.set_cores(1)
        for settings in (QUICK, CHECKED):
            calls.clear()
            cot = losvd_rows_vjp(model, table, settings, [c])
            explicit = losvd_rows_explicit(model, table, settings, [c], names)
            out[settings.check_rows] = (cot, explicit, list(calls))
    finally:
        runtime.set_cores(saved)
    assert out[False][2] == ["row", "row"]
    assert out[True][2] == ["row", "levels", "row", "levels"]
    (once, e_once, _), (checked, e_checked, _) = out[False], out[True]
    for name in ("log_df", "log_df_slope", "log_df_below", "log_df_slope_below"):
        assert getattr(once, name).tobytes() == getattr(checked, name).tobytes()
    assert once.form.keys() == checked.form.keys()
    for name in once.form:
        a, b = np.asarray(once.form[name]), np.asarray(checked.form[name])
        assert a.tobytes() == b.tobytes()
    for name in names:
        assert (
            np.float64(e_once[name]).tobytes() == np.float64(e_checked[name]).tobytes()
        )


def test_the_rows_checks_run_with_check_rows(parent, monkeypatch):
    """Assert that the rows' checks run only with check_rows=True.

    With check_rows=True, `losvd_rows_explicit` raises ResultError for a
    LOSVD row scaled by :math:`1 + 10^{-15}`. `losvd_rows_explicit` also
    raises ResultError when `_rows_theta._plane_twin` is scaled by
    :math:`1 + 10^{-9}`, so that the twin of the plane does not reproduce
    the row. With check_rows=False, which is the default,
    `losvd_rows_explicit` and `losvd_rows_vjp` return for the scaled row,
    and `losvd_rows_explicit` returns with the scaled twin.
    """
    from pianistpy.observables import _rows_theta
    from pianistpy.observables._rows_theta import losvd_rows_explicit

    model = _child(parent, r_a=5.0)
    row, table = _held_row(model)
    c = np.random.default_rng(14).normal(size=row.losvd[0].shape)
    altered = SimpleNamespace(
        rows=(dataclasses.replace(row, losvd=row.losvd * (1.0 + 1e-15)),),
        kept=table.kept,
    )
    with pytest.raises(ResultError, match="differs from the stored distribution"):
        losvd_rows_explicit(model, altered, CHECKED, [c], ("r_a",))
    losvd_rows_explicit(model, altered, QUICK, [c], ("r_a",))
    losvd_rows_vjp(model, altered, QUICK, [c])
    twin = _rows_theta._plane_twin
    monkeypatch.setattr(_rows_theta, "_plane_twin", lambda *a: twin(*a) * (1.0 + 1e-9))
    with pytest.raises(ResultError, match="twin of the plane"):
        losvd_rows_explicit(model, table, CHECKED, [c], ("r_a",))
    losvd_rows_explicit(model, table, QUICK, [c], ("r_a",))


@CPU_ONLY
def test_the_rows_read_one_route_and_lookup_held_once(parent, monkeypatch):
    """Assert that one RowsBasis builds the route with the DF once for two
    calls of `losvd_rows_vjp` and two calls of `losvd_rows_explicit`.

    Every call with the basis returns the same bits as the same call without
    a basis. Each call of `RowsBasis.route` returns a new route object, and
    the new route's cache of extrema (``_fext``) is empty.
    """
    from pianistpy.observables import _ellipse_model
    from pianistpy.observables._rows_theta import losvd_rows_explicit
    from pianistpy.observables._table_vjp import RowsBasis

    model = _child(parent, r_a=5.0)
    projected = projected_dispersion(model, dispersion_profile(model, QUICK), QUICK)
    kept = kept_ranges(model, projected, QUICK)
    row = losvd_row(model, 1.0, kept, projected, QUICK)
    table = SimpleNamespace(rows=(row,), kept=kept)
    c = [np.random.default_rng(12).normal(size=row.losvd[0].shape)]
    alone = losvd_rows_vjp(model, table, QUICK, c)
    alone_explicit = losvd_rows_explicit(model, table, QUICK, c, ("r_a",))
    built = []
    made = _ellipse_model.ellipse_route
    monkeypatch.setattr(
        _ellipse_model,
        "ellipse_route",
        lambda *args, **kwargs: built.append(1) or made(*args, **kwargs),
    )
    basis = RowsBasis(model, QUICK)
    held = [losvd_rows_vjp(model, table, QUICK, c, basis) for _ in range(2)]
    explicit = [
        losvd_rows_explicit(model, table, QUICK, c, ("r_a",), basis) for _ in range(2)
    ]
    assert len(built) == 1
    assert basis.route(table) is not basis.route(table)
    assert basis.route(table)._fext == {}
    for cot in held:
        for name in ("log_df", "log_df_slope", "log_df_below", "log_df_slope_below"):
            assert getattr(cot, name).tobytes() == getattr(alone, name).tobytes()
        for name in alone.form:
            assert (
                np.asarray(cot.form[name]).tobytes()
                == np.asarray(alone.form[name]).tobytes()
            )
    for part in explicit:
        assert (
            np.float64(part["r_a"]).tobytes()
            == np.float64(alone_explicit["r_a"]).tobytes()
        )


def test_a_second_reading_builds_no_route(om_table, monkeypatch):
    """Assert that `losvd` keeps its route and stencil on the LOSVD table, so
    that a second reading and the reading's transpose build no route.

    `aperture_losvd` with a circular aperture of radius 1 first forms
    extrema on the table, and the table's readers are then cleared. Two
    calls of `losvd`, one `_read_losvd` and one `losvd_read_vjp` build one
    route in all. The second `losvd` returns the bits of the first. The
    reading and the reading's transpose return the bits of a fresh route
    and stencil from `_stencil_of`, and the fresh route's cache of extrema
    (``_fext``) is empty. The radii are up to 20 radii at which the aperture
    formed extrema and 20 random radii.
    """
    from pianistpy import Aperture, aperture_losvd, losvd
    from pianistpy.observables import core
    from pianistpy.observables._table_vjp import _stencil_read_vjp

    part = om_table.parts[0]
    aperture_losvd(om_table, Aperture.circle(1.0), np.array([0.0, 0.3]))
    key = (part.model.identity, ellipse_rules(part.model.settings))
    formed = np.array(sorted(part.table.extrema[key]))
    assert formed.size
    rng = np.random.default_rng(13)
    x = np.concatenate(
        [formed[:: max(1, formed.size // 20)][:20], rng.uniform(0.05, 10.0, 20)]
    )
    u = rng.uniform(-1.2, 1.2, x.size)
    c = rng.normal(size=x.size)
    part.table.readers.clear()
    built = []
    made = core.ellipse_route
    monkeypatch.setattr(
        core,
        "ellipse_route",
        lambda *args, **kwargs: built.append(1) or made(*args, **kwargs),
    )
    first = np.asarray(losvd(om_table, x, u))
    second = np.asarray(losvd(om_table, x, u))
    read = core._read_losvd(part, x, u)
    back = losvd_read_vjp(core._reader_of(part), x, u, c)
    assert len(built) == 1
    assert second.tobytes() == first.tobytes()
    route, stencil = core._stencil_of(part)
    assert route._fext == {}
    assert read.tobytes() == stencil.at_pairs(route, x, u).tobytes()
    route, stencil = core._stencil_of(part)
    fresh = _stencil_read_vjp(stencil, route, x, u, c)
    assert len(back) == len(fresh)
    for a, b in zip(back, fresh, strict=True):
        assert a.tobytes() == b.tobytes()
