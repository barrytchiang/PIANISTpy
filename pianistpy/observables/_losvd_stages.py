r"""Helpers shared by the stages of the line-of-sight velocity distribution.

The stages behind `pianistpy.losvd`, `pianistpy.aperture_losvd` and
`pianistpy.LosTable` use this module for five tasks: building the ellipse
route's rules from the settings' ``losvd`` group; placing the projected radii
of the table's rows; starting the process pool; converting the table's rows to
and from the `PanelTable` that reads the table between rows; and the projected
mass that sets where the rows start and stop.

The projected mass inside a projected radius :math:`R` is, up to the constant
factor :math:`2\pi`,

.. math::

    M(R) = \int_0^R \Sigma(R')\, R'\, {\rm d}R',

with :math:`\Sigma` the surface density. The rows start at the radius where
:math:`M` is :math:`10^{-6}` times :math:`M` at the innermost radius the table
is read at, the setting ``losvd.innermost_read_radius``. Notes Section T:6.5
treats the table of rows."""

__all__ = [
    "base_rows",
    "cut_rows",
    "ellipse_rules",
    "panel_table_of",
    "row_start",
    "row_stop",
    "rows_of",
    "served_radius",
    "shared_pool",
    "start_pool",
    "table_stencil",
]

import contextlib
import math
from collections.abc import Iterator
from typing import Any

import numpy as np

from ..df import Model
from ..exceptions import InputError
from ..runtime import get_cores
from ..settings import Settings
from ..typing import Array
from ..validation import TIER_BOUNDS
from ._ellipse import EllipseRoute
from ._ellipse_model import floor_radius
from ._ellipse_rules import EllipseRules
from ._layout import Lattice
from ._losvd_table import new_pool
from ._panel_table import PanelTable
from ._projection import TINY, surface_density_reader

# The last base row lies at EDGE_FRACTION times the radius where the rows end
# (`floor_radius`).
EDGE_FRACTION = 0.9999
# The entry n_u of the row rule of the regular class and of the cusp class.
# No part of the route reads n_u. The entry keeps the rule's six-entry form
# (n_u, k0, ks, kq, m, n_phi).
REGULAR_N_U = 100
CUSP_N_U = 200
# The keys of a row's break velocities in `LosvdRow.v_breaks`.
BREAK_NAMES = ("v_k", "v_rad", "v_edge", "v_c", "v_end")


ROWS_ROUTES = ("kernel", "numpy")


def _rows_route(settings: Settings) -> str:
    """Return the setting ``observables.rows_route``, one of ROWS_ROUTES.

    Raises
    ------
    InputError
        If ``observables.rows_route`` is not in ROWS_ROUTES.
    """
    route = settings.observables["rows_route"]
    if route not in ROWS_ROUTES:
        raise InputError(
            f"The setting observables.rows_route must be one of {list(ROWS_ROUTES)}; "
            f"the value given is {route!r}."
        )
    return str(route)


def ellipse_rules(settings: Settings) -> EllipseRules:
    """Return the ellipse route's rules from the settings' ``losvd`` group.

    The row rule of the regular class and of the cusp class is
    ``(n_u, 0, 0, kq, m, n_phi)``. ``n_u`` is REGULAR_N_U or CUSP_N_U. ``kq``
    is ``losvd.n_floor_layers``. ``m`` is ``losvd.n_energy_nodes``.
    ``n_phi`` is ``losvd.n_angle_nodes``. The least layer counts ``k0`` and
    ``ks`` are 0, so the two classes differ only in ``n_u``, which no part of
    the route reads.

    Raises
    ------
    InputError
        If ``observables.rows_route`` is not in ROWS_ROUTES.
    """
    f = settings.losvd
    rule = (
        0,
        0,
        int(f["n_floor_layers"]),
        int(f["n_energy_nodes"]),
        int(f["n_angle_nodes"]),
    )
    return EllipseRules(
        rule_regular=(REGULAR_N_U,) + rule,
        rule_cusp=(CUSP_N_U,) + rule,
        s_ratio=float(f["energy_layer_ratio"]),
        n_bisect=int(f["n_bisections"]),
        chunk=int(f["n_block_elements"]),
        t_nodes=int(f["n_sightline_nodes"]),
        t_panels=int(f["n_sightline_panels"]),
        t_levels=int(f["sightline_panels_refined"]),
        t_ratio=float(f["sightline_layer_ratio"]),
        t_layers=int(f["n_sightline_layers"]),
        l_scan=int(f["n_crossing_scan"]),
        g_nodes=int(f["n_cumulative_nodes"]),
        vp_ratio=float(f["velocity_layer_ratio"]),
        vp_layers_zero=int(f["n_velocity_layers_zero"]),
        vp_layers_kink=int(f["n_velocity_layers_break"]),
        vp_nodes=int(f["n_velocity_nodes"]),
        brk_scan=int(f["n_break_scan"]),
        brk_zoom=int(f["n_break_zooms"]),
        rb_layers=int(f["n_radius_break_layers"]),
        rb_bisect=int(f["n_radius_break_bisections"]),
        rb_interior=int(f["n_radius_break_samples"]),
        rb_ext=int(f["n_edge_samples"]),
        rt_layers=int(f["n_edge_layers"]),
        row_order=int(f["n_stencil_rows"]),
        row_order_test=int(f["n_stencil_test_rows"]),
        kernel=_rows_route(settings) == "kernel",
    )


def base_rows(model: Model, settings: Settings) -> Array:
    """Return the projected radii of the base rows, in internal units.

    The radii are spaced evenly in log radius, at least
    ``losvd.rows_per_decade`` per decade. The first radius is the profile's
    first radius. The last radius is EDGE_FRACTION times the radius where the
    rows end (`floor_radius`).
    """
    lo = float(model.profile.r[0])
    hi = floor_radius(model) * EDGE_FRACTION
    per_decade = int(settings.losvd["rows_per_decade"])
    n = int(math.ceil(per_decade * math.log10(hi / lo))) + 1
    base = np.logspace(math.log10(lo), math.log10(hi), n)
    base[0] = max(float(base[0]), lo)
    return base


def _projected_mass(model: Model, projected: Any) -> tuple[Array, Array, float]:
    r"""Return (ln R, M(R), M_total): the log of each projected radius of
    `projected`, the projected mass M inside each radius, and the total
    projected mass, each M without the factor :math:`2\pi`.

    M at a radius is the sum of two parts: the mass inside the first radius
    (`SurfaceDensityReading.inner_disk`); and the trapezoid rule in
    :math:`\ln R` of :math:`\Sigma R^2` from the first radius to that radius.
    M_total adds to M at the last radius the mass of the annulus from the
    last radius :math:`R_{\rm last}` to the profile's last radius
    :math:`R_{\rm t}`. Where the surface density reader has an edge, the
    annulus takes
    :math:`\Sigma(R) = \Sigma(R_{\rm last}) [(R_{\rm t}^2 - R^2) /
    (R_{\rm t}^2 - R_{\rm last}^2)]^{1/2}`. Elsewhere the annulus takes
    :math:`\Sigma(R) = \Sigma(R_{\rm last})`. The annulus mass is multiplied
    by the ratio of the rows' trapezoid rule in :math:`\ln R` to the rows'
    trapezoid rule in :math:`R`.
    """
    r = np.asarray(projected.r_2d, dtype=np.float64)
    sigma = np.asarray(projected.surface_density[0], dtype=np.float64)
    r_t = float(model.profile.r_table_max)
    reading = surface_density_reader(r, sigma, r_t, model.spatially_finite)
    f = sigma * r * r
    ln_r = np.log(r)
    steps = np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(ln_r))
    inside = reading.inner_disk + np.concatenate([[0.0], steps])
    v_max = float(np.sqrt(max(r_t**2 - float(r[-1]) ** 2, 0.0)))
    annulus = (
        (float(sigma[-1]) / v_max * v_max**3 / 3.0)
        if (reading.r_t is not None and v_max > 0.0)
        else float(sigma[-1]) * v_max**2 / 2.0
    )
    rows = max(float(np.trapezoid(sigma * r, r)), TINY)
    total = float(inside[-1]) + annulus / rows * float(steps[-1])
    return ln_r, inside, total


def row_start(model: Model, projected: Any, r_read: float) -> float:
    """Return the projected radius where the rows start.

    The radius is the one inside which the projected mass is ``1e-6`` times
    the projected mass inside `r_read` (`_projected_mass`). The function
    returns the profile's first radius instead in two cases: `r_read` is not
    positive; or the radius found lies inside the first radius of
    `projected`.
    """
    lo = float(model.profile.r[0])
    if r_read <= 0.0:
        return lo
    ln_r, inside, _ = _projected_mass(model, projected)
    target = TIER_BOUNDS[0] * float(np.interp(np.log(r_read), ln_r, inside))
    if target <= inside[0]:
        return lo
    return max(lo, float(np.exp(np.interp(target, inside, ln_r))))


def row_stop(model: Model, projected: Any, share: float) -> float:
    """Return the projected radius where the rows stop.

    The radius is the one outside which the fraction `share` of the total
    projected mass lies (`_projected_mass`). The function returns infinity
    in two cases: `share` is not positive; or the radius found lies at or
    beyond the last radius of `projected`.
    """
    if share <= 0.0:
        return float("inf")
    ln_r, inside, total = _projected_mass(model, projected)
    target = (1.0 - share) * total
    if target >= inside[-1]:
        return float("inf")
    return float(np.exp(np.interp(target, inside, ln_r)))


def served_radius(model: Model, projected: Any, r_first: float) -> float:
    """Return the smallest radius of a centered circle whose projected mass
    is at least ``1e6`` times the projected mass inside `r_first`.

    For a table whose rows start at `r_first`, the mass inside `r_first` is
    at most ``1e-6`` of the mass of such a circle. The function returns
    infinity where the radius found lies at or beyond the last radius of
    `projected`.
    """
    ln_r, inside, _ = _projected_mass(model, projected)
    target = float(np.interp(np.log(r_first), ln_r, inside)) / TIER_BOUNDS[0]
    if target >= inside[-1]:
        return float("inf")
    return float(np.exp(np.interp(target, inside, ln_r)))


def cut_rows(rows: Array, r_start: float, r_stop: float) -> Array:
    """Return the radii of `rows` from the last radius at or inside `r_start`
    to the first radius at or beyond `r_stop`.

    The returned radii are a contiguous slice of `rows`, so no radius moves.
    """
    first = max(int(np.searchsorted(rows, r_start, side="right")) - 1, 0)
    last = min(int(np.searchsorted(rows, r_stop, side="left")), len(rows) - 1)
    return rows[first : last + 1]


# The pools of the open `shared_pool` scope, in a dict keyed by `start_pool`'s
# key. The list holds one dict while a scope is open and is empty otherwise.
_SCOPES: list[dict[Any, Any]] = []


@contextlib.contextmanager
def shared_pool() -> Iterator[None]:
    """Open a scope that keeps every keyed pool from `start_pool` open until
    the scope ends.

    The stages of one table that use the same route then start the pool's
    processes once. At the end of the scope, every kept pool is terminated
    and joined. A scope opened inside an open scope adds nothing, and the
    outer scope keeps the pools.
    """
    if _SCOPES:
        yield
        return
    _SCOPES.append({})
    try:
        yield
    finally:
        for pool in _SCOPES.pop().values():
            pool.terminate()
            pool.join()


def start_pool(route: EllipseRoute, key: Any = None) -> Any:
    """Return a pool of `get_cores` processes on the CPU, each process
    holding a copy of `route` (`new_pool`), for use as a context manager.

    Outside a `shared_pool` scope, or where `key` is None, the pool is new
    and the context manager terminates the pool at its exit. Inside a
    `shared_pool` scope, the pool for `key` is kept and returned again for
    the same key, and the context manager leaves the pool open. The callers
    pass as `key` the model's identity and the rules the route was built
    with.
    """
    if not _SCOPES or key is None:
        return new_pool(get_cores(), route)
    pools = _SCOPES[-1]
    if key not in pools:
        pools[key] = new_pool(get_cores(), route)
    return contextlib.nullcontext(pools[key])


def rows_of(
    table: PanelTable,
    norm_rel: Array,
    m2_rel: Array,
    beyond_rel: Array,
    info: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return each row of `table` as a dict of the fields of a `LosvdRow`.

    Each row keeps the half ``V >= 0`` of its nodes, weights and values. The
    record of each row holds the row's norm against the surface density and
    second moment against the squared projected dispersion, each less one
    (`norm_rel`, `m2_rel`), and the row's share beyond the escape velocity
    (`beyond_rel`), together with the entries of `info` for that row.
    """
    rows = []
    for i in range(len(table.R_2D)):
        v, w, d = table.row(i)
        half = len(v) // 2
        breaks = dict(
            zip(
                BREAK_NAMES,
                (
                    float(table.Vk[i]),
                    float(table.Vrad[i]),
                    float(table.Vedge[i]),
                    float(table.Vc[i]),
                    float(table.Vend[i]),
                ),
                strict=True,
            )
        )
        rows.append(
            dict(
                r_2d=float(table.R_2D[i]),
                layout=str(int(table._layout[i])),
                v_breaks=breaks,
                v_escape=float(table.vesc[i]),
                v_nodes=np.asarray(v[half:]),
                v_weights=np.asarray(w[half:]),
                panel_edges=np.asarray(table.row_breaks(i), dtype=np.float64),
                losvd=np.asarray(d[half:])[np.newaxis, :],
                record=dict(
                    norm_against_surface_density=float(norm_rel[i]),
                    m2_against_projected_dispersion=float(m2_rel[i]),
                    beyond_escape=float(beyond_rel[i]),
                    cusp=bool(info["cusp"][i]),
                    t_panels=int(info["t_panels"][i]),
                    t_change=float(info["t_change"][i]),
                    t_converged=bool(info["t_converged"][i]),
                    seconds=float(info["row_seconds"][i]),
                ),
            )
        )
    return rows


def panel_table_of(
    lattice: Lattice,
    r_2d: Array,
    rows: list[tuple[Any, ...]],
    r_breaks: Array,
    r_edge: float,
    order: int,
    uncertified: Array,
    why: list[str],
    r_end: float = float("inf"),
) -> PanelTable:
    """Return the `PanelTable` that reads a table between its rows.

    Each entry of `rows` is
    ``(V, w, D, v_esc, V_k, V_end, V_rad, n_panels, code, V_edge, V_c)`` on
    ``V >= 0``, with D normalized to unit integral over V. The function does
    not normalize D. The returned table reads no radius beyond `r_end`, which
    is the last row of a table cut inside its edge. The arguments
    `uncertified` and `why` are the intervals of radius, shape (n, 2), that
    the table's check left above its tolerance, and the reason for each
    interval.
    """
    table = PanelTable.from_rows(lattice, r_2d, rows, r_breaks=r_breaks, r_edge=r_edge)
    table.R_end = min(table.R_end, float(r_end))
    table.order = int(order)
    table.uncert = np.asarray(uncertified, dtype=np.float64).reshape(-1, 2)
    table.uncert_why = list(why)
    return table


def table_stencil(route: Any, table: Any) -> PanelTable:
    """Return the `PanelTable` that reads the `LosvdTable` `table` between
    its rows (`panel_table_of`).

    The `PanelTable` takes the stored rows of `table`, which are normalized,
    the lattice of `route`, and six items from `table`: the radii where the
    layout changes; the edge ``r_edge``; the last radius read, ``r_end``;
    the number of rows the interpolation between rows uses; the intervals of
    radius that the table's check left above its tolerance; and the reason
    for each such interval.
    """
    return panel_table_of(
        route.lattice,
        np.array([row.r_2d for row in table.rows]),
        [
            (
                row.v_nodes,
                row.v_weights,
                row.losvd[0],
                row.v_escape,
                row.v_breaks["v_k"],
                row.v_breaks["v_end"],
                row.v_breaks["v_rad"],
                len(row.panel_edges) - 1,
                int(row.layout),
                row.v_breaks["v_edge"],
                row.v_breaks["v_c"],
            )
            for row in table.rows
        ],
        table.r_breaks,
        float(table.identity["r_edge"]),
        table.order,
        table.r_uncertified,
        list(table.why_uncertified),
        float(table.identity.get("r_end", float("inf"))),
    )
