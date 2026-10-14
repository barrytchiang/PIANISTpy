"""The pieces the LOSVD stages share: the rules from the settings, the base
rows, the pool, and the conversion between the table of rows and the stencil
that reads it."""

__all__ = [
    "GPU_FOOTPRINT_MARGIN",
    "GPU_RESERVE",
    "base_rows",
    "ellipse_rules",
    "gpu_pool_size",
    "panel_table_of",
    "rows_of",
    "shared_pool",
    "start_pool",
]

import contextlib
import math
from collections.abc import Iterator
from typing import Any

import numpy as np

from ..df import Model
from ..exceptions import InputError, ResultError
from ..runtime import card_memory, get_workers, log, workers_explicit
from ..settings import Settings
from ..typing import Array
from ._ellipse import EllipseRoute
from ._ellipse_rules import EllipseRules
from ._layout import Lattice
from ._losvd_table import new_pool, row_worker
from ._panel_table import PanelTable

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
    the profile's first row to `EDGE_FRACTION` of its last."""
    lo = float(model.profile.r[0])
    hi = float(model.profile.r_table_max) * EDGE_FRACTION
    per_decade = int(settings.losvd["rows_per_decade"])
    n = int(math.ceil(per_decade * math.log10(hi / lo))) + 1
    base = np.logspace(math.log10(lo), math.log10(hi), n)
    base[0] = max(float(base[0]), lo)
    return base


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


# On a GPU every worker is a process with its own CUDA context and its own
# allocations on the one card, so the card's memory bounds the pool, not the
# cores [3-182]. A worker's footprint is read from the card at the pool's
# start: the drop in the card's free memory across one worker's start and one
# row of the live table. Both bounds are measured (2026-10-12, the 3-182
# battery, 45 pools on the card): the margin is the largest ratio of a worker's
# peak over the table to its pool's probe footprint, 328 / 327 MB = 1.003 in 23
# losvd_table pools (first FV_nfw10_COMI_b+0.25_Ra10_Zt0p05), taken to 1.01 for
# the 1 MB reading (329 / 326); the reserve is the main process's growth on
# the card after the probe, 0 MB in every pool.
GPU_FOOTPRINT_MARGIN = 1.01
GPU_RESERVE = 0.0


def gpu_pool_size(free: int, total: int, footprint: int, n_max: int) -> int:
    """Return the workers a card holds: the probe's worker and as many more
    as `free` (after the probe) holds at `GPU_FOOTPRINT_MARGIN` times the
    `footprint`, keeping `GPU_RESERVE` of `total` free; at least one, at most
    `n_max`."""
    room = free - GPU_RESERVE * total
    more = math.floor(room / (GPU_FOOTPRINT_MARGIN * max(footprint, 1)))
    return max(1, min(n_max, 1 + max(more, 0)))


def _workers_for(route: EllipseRoute, probe_r2d: float | None) -> int:
    """`get_workers`, or on a GPU, unless the workers were set to a number,
    what the card holds (`gpu_pool_size`), measured on a row at
    `probe_r2d`."""
    n_max = get_workers()
    if probe_r2d is None or workers_explicit() or n_max == 1:
        return n_max
    before = card_memory()
    if before is None:
        return n_max
    none = np.zeros((0, 2))
    try:
        with new_pool(1, route) as probe:
            probe.apply(row_worker, ((0, probe_r2d, (none, none)),))
            after = card_memory()
    except RuntimeError as e:
        # a second process on a card in Exclusive_Process mode, with no MPS
        # server of this user's to share it through (measured 2026-10-12)
        if "Unable to initialize backend" not in str(e):
            raise
        raise ResultError(
            "The GPU pool's probe worker could not reach the card, which the "
            "main process holds; a card in Exclusive_Process mode takes a "
            f"second process only through an MPS server ({e})."
        ) from e
    if after is None:
        return n_max
    n = gpu_pool_size(after[0], after[1], before[0] - after[0], n_max)
    log.info(
        "GPU pool: %d workers (a worker %.0f MB on the card, %.0f MB free of "
        "%.0f MB; %d asked).",
        n,
        (before[0] - after[0]) / 2**20,
        after[0] / 2**20,
        after[1] / 2**20,
        n_max,
    )
    return n


def start_pool(
    route: EllipseRoute, key: Any = None, probe_r2d: float | None = None
) -> Any:
    """Return a pool of `get_workers` processes, each holding a copy of the
    route (`init_worker`), to be used as a context manager.

    On a GPU, given `probe_r2d` (a radius of the live table) and unless the
    workers were set to a number, the pool takes what the card's memory
    holds instead (`gpu_pool_size`) [3-182]. Inside a `shared_pool` scope, a
    pool started for `key` (the model's identity and the rules the route was
    built with) is kept and returned again for the same key; the scope
    closes it.
    """
    if not _SCOPES or key is None:
        return new_pool(_workers_for(route, probe_r2d), route)
    pools = _SCOPES[-1]
    if key not in pools:
        pools[key] = new_pool(_workers_for(route, probe_r2d), route)
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
) -> PanelTable:
    """Return the stencil of a table from its rows ``(V, w, D, v_esc, V_k,
    V_end, V_rad, n_panels, code, V_edge, V_c)``, the D of each normalized."""
    table = PanelTable.from_rows(lattice, r_2d, rows, r_breaks=r_breaks, r_edge=r_edge)
    table.order = int(order)
    table.uncert = np.asarray(uncertified, dtype=np.float64).reshape(-1, 2)
    table.uncert_why = list(why)
    return table
