"""The pieces the LOSVD stages share: the rules from the settings, the base
rows, the pool, and the conversion between the table of rows and the stencil
that reads it."""

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
from ..runtime import get_workers
from ..settings import Settings
from ..typing import Array
from ..validation import GOAL
from ._ellipse import EllipseRoute
from ._ellipse_model import floor_radius
from ._ellipse_rules import EllipseRules
from ._layout import Lattice
from ._losvd_table import new_pool
from ._panel_table import PanelTable
from ._projection import TINY, surface_density_reader

# The base rows end this far inside the last row of the profile.
EDGE_FRACTION = 0.9999
# N_u of a regular and a cusp row, which no route reads (the t rule has its
# own panels); kept so that the rules read as the working code's.
REGULAR_N_U = 100
CUSP_N_U = 200
# The names of a row's break velocities, as `LosvdRow.v_breaks` holds them.
BREAK_NAMES = ("v_k", "v_rad", "v_edge", "v_c", "v_end")


ROWS_ROUTES = ("kernel", "numpy")


def _rows_route(settings: Settings) -> str:
    """Return the rows' route of the settings, or refuse an unknown one."""
    route = settings.observables["rows_route"]
    if route not in ROWS_ROUTES:
        raise InputError(
            f"observables.rows_route is one of {list(ROWS_ROUTES)}; got {route!r}."
        )
    return str(route)


def ellipse_rules(settings: Settings) -> EllipseRules:
    """Return the ellipse route's rules from the settings' ``losvd`` group.

    The rule of a row is (N_u, K_0, K_s, K_q, m, N_phi): N_u is read by no
    route and K_0, K_s are floors under each node pair's own layers, so the
    regular and the cusp class share the three fields that act."""
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
    """Return the base rows: ``rows_per_decade`` a decade, logarithmic from
    the profile's first row to `EDGE_FRACTION` of where the rows end
    (`floor_radius`: the profile's last row, or the DF's floor inside it)."""
    lo = float(model.profile.r[0])
    hi = floor_radius(model) * EDGE_FRACTION
    per_decade = int(settings.losvd["rows_per_decade"])
    n = int(math.ceil(per_decade * math.log10(hi / lo))) + 1
    base = np.logspace(math.log10(lo), math.log10(hi), n)
    base[0] = max(float(base[0]), lo)
    return base


def _projected_mass(model: Model, projected: Any) -> tuple[Array, Array, float]:
    r"""Return (ln R, :math:`\int_0^R \Sigma R'\,{\rm d}R'` at the rows of
    `projected`, the integral over all R): the surface density reader's disk
    inside the first row (`SurfaceDensityReading.inner_disk`), the trapezoid
    in :math:`\ln R` over the rows, and the annulus beyond the last row with
    the edge-divided surface density frozen there, relative to the rows' own
    trapezoid in R."""
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
    """Return the projected radius whose disk holds `GOAL` times the
    projected mass inside `r_read`, from the model's surface density
    (`_projected_mass`); the profile's first row where `r_read` is not
    positive or the radius lies inside it."""
    lo = float(model.profile.r[0])
    if r_read <= 0.0:
        return lo
    ln_r, inside, _ = _projected_mass(model, projected)
    target = GOAL * float(np.interp(np.log(r_read), ln_r, inside))
    if target <= inside[0]:
        return lo
    return max(lo, float(np.exp(np.interp(target, inside, ln_r))))


def row_stop(model: Model, projected: Any, share: float) -> float:
    """Return the projected radius outside which `share` of the total
    projected mass lies, from the model's surface density
    (`_projected_mass`); infinity where `share` is not positive or the
    radius lies beyond the rows of `projected`."""
    if share <= 0.0:
        return float("inf")
    ln_r, inside, total = _projected_mass(model, projected)
    target = (1.0 - share) * total
    if target >= inside[-1]:
        return float("inf")
    return float(np.exp(np.interp(target, inside, ln_r)))


def served_radius(model: Model, projected: Any, r_first: float) -> float:
    """Return the smallest radius of a centered circle a table whose rows
    start at `r_first` serves at the goal: the radius whose disk holds the
    projected mass inside `r_first` over `GOAL` (`_projected_mass`)."""
    ln_r, inside, _ = _projected_mass(model, projected)
    target = float(np.interp(np.log(r_first), ln_r, inside)) / GOAL
    if target >= inside[-1]:
        return float("inf")
    return float(np.exp(np.interp(target, inside, ln_r)))


def cut_rows(rows: Array, r_start: float, r_stop: float) -> Array:
    """Return the rows from the last one at or inside `r_start` to the first
    one at or beyond `r_stop`: the rows kept are the full grid's own."""
    first = max(int(np.searchsorted(rows, r_start, side="right")) - 1, 0)
    last = min(int(np.searchsorted(rows, r_stop, side="left")), len(rows) - 1)
    return rows[first : last + 1]


# The pools of the open `shared_pool` scope, by the key of their route: one
# dict while a scope is open, none outside.
_SCOPES: list[dict[Any, Any]] = []


@contextlib.contextmanager
def shared_pool() -> Iterator[None]:
    """Keep the pools started inside the scope open to its end, so that the
    stages of one table that read the same route start their workers once."""
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
    """Return a pool of `get_workers` processes, each holding a copy of the
    route (`init_worker`), on the CPU, to be used as a context manager.

    Inside a `shared_pool` scope, a pool started for `key` (the model's
    identity and the rules the route was built with) is kept and returned
    again for the same key; the scope closes it.
    """
    if not _SCOPES or key is None:
        return new_pool(get_workers(), route)
    pools = _SCOPES[-1]
    if key not in pools:
        pools[key] = new_pool(get_workers(), route)
    return contextlib.nullcontext(pools[key])


def rows_of(
    table: PanelTable,
    norm_rel: Array,
    m2_rel: Array,
    beyond_rel: Array,
    info: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return each row of a table as the fields of a `LosvdRow`."""
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
    """Return the stencil of a table from its rows ``(V, w, D, v_esc, V_k,
    V_end, V_rad, n_panels, code, V_edge, V_c)``, the D of each normalized;
    read to `r_end` at most (the last row of a table cut inside its edge)."""
    table = PanelTable.from_rows(lattice, r_2d, rows, r_breaks=r_breaks, r_edge=r_edge)
    table.R_end = min(table.R_end, float(r_end))
    table.order = int(order)
    table.uncert = np.asarray(uncertified, dtype=np.float64).reshape(-1, 2)
    table.uncert_why = list(why)
    return table


def table_stencil(route: Any, table: Any) -> PanelTable:
    """Return the stencil that reads a `LosvdTable` between its rows: its
    stored rows, each normalized, on the route's lattice, with the table's
    breaks, edge, order and unchecked intervals."""
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
