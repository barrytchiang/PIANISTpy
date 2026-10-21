"""Pairs of projected radius and line-of-sight velocity drawn from a model.

A user builds a `ProjectedSampler` from a model or the model's `LosTable`,
an annulus of projected radius and an optional user-specified range of
line-of-sight velocity. `ProjectedSampler.sample` then returns the drawn
pairs as `ProjectedPairs`.

The pairs follow the probability density
``p(R, v) proportional to 2 pi R Sigma(R) D(v | R)`` on the annulus and in
the velocity range. In ``p(R, v)``, ``R`` is the projected radius, ``v`` is
the line-of-sight velocity, ``Sigma(R)`` is the surface density, and
``D(v | R)`` is the line-of-sight velocity distribution at ``R`` that the
`LosTable` holds.

The sampler builds its tables of the inverse cumulative distributions once,
at construction. The sampler refines each table until the table's error in
the cumulative distribution is at most :math:`10^{-6}`, or until the table
reaches its largest size. The validation report of the sampler records the
error reached. The line-of-sight velocity distribution is described in
Notes Section T:6.4.
"""

__all__ = ["ProjectedPairs", "ProjectedSampler"]

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..df import Model
from ..exceptions import AccuracyError, InputError
from ..particles import stages as particle_stages
from ..typing import Array, ArrayLike
from ..validation import (
    TIER_BOUNDS,
    Check,
    ValidationReport,
    refuse_unless_asked,
)
from ._ellipse_rules import gl01
from ._panel_table import PanelTable
from ._projection import surface_density_reader
from .core import (
    LosTable,
    _announce,
    _internal_length,
    _internal_velocity,
    _stencil_of,
    _weights,
)
from .stages import INNER_DISK_TOLERANCE

PROJECTED_KEYS = ("v_table", "window", "radius_quadrature", "row_norm", "annulus_cut")

# The mechanics of the sampler. The LosTable's stencil (PanelTable)
# interpolates the velocity distributions between the table's rows of
# projected radius. The stencil uses a Lagrange polynomial, in the stencil's
# own variable of a stretch (PanelTable._coord), through a set of
# PanelTable.order successive rows. The set of rows shifts by one row at
# fixed radii, so the interpolant is one smooth function between two
# successive shifts. _Part cuts the annulus into stretches at the table's
# radius breaks (R_breaks). _Stretch cuts each stretch into pieces at the
# radii where the set of rows shifts. Every table of the sampler is
# interpolated inside one piece only.

# The nodes in lambda_u = ln(u / (1 - u)) cover [-LAMBDA_END, LAMBDA_END].
# Beyond LAMBDA_END, u or 1 - u is below 1e-12, and the table of s continues
# linearly with the slope at the end node.
LAMBDA_END = 28.0
# The node count in lambda_u starts at 33 and doubles (n to 2 n - 1) up to
# 1025, until the Hermite interpolant of s meets 1e-6 at the midpoints.
N_LAMBDA_START, N_LAMBDA_MAX = 33, 1025
# The node count of a piece starts at 4 and triples up to 108, until the
# barycentric interpolant across the piece meets 1e-6 at the new nodes.
N_PIECE_START, N_PIECE_MAX = 4, 108
# The Gauss-Legendre node count of the mass integral on each piece, and the
# node count of the finer rule that checks the mass integral against 1e-6.
N_RADIUS_GAUSS, N_RADIUS_CHECK = 8, 16
# The largest number of bracketed Newton steps of one root solve.
NEWTON_STEPS = 60
# The number of radii whose velocity breaks are scanned in one array. Each
# radius's scan holds the rules' brk_scan points (4001 by default), so
# CURVE_CHUNK bounds the memory of one scan.
CURVE_CHUNK = 1024
TINY = np.finfo(np.float64).tiny
EPS = np.finfo(np.float64).eps


@dataclass(frozen=True)
class ProjectedPairs:
    """Pairs of projected radius and line-of-sight velocity.

    `ProjectedSampler.sample` returns a `ProjectedPairs`. Each array
    attribute holds one value per pair.

    Attributes
    ----------
    r_2d : array, shape (n,)
        Projected radius of each pair. The unit depends on the source's unit
        system (`Model.to`): units of :math:`r_0` in the internal and the
        model systems, parsecs in the physical system.
        ``units["r_2d"]`` gives the unit.
    v_los : array, shape (n,)
        Line-of-sight velocity of each pair. The unit depends on the
        source's unit system: internal units, units of
        :math:`(G M_0/r_0)^{1/2}` in the model system, km/s in the physical
        system. ``units["v_los"]`` gives the unit.
    mass : array, shape (n,)
        Mass of each pair. Every pair carries the mass
        `ProjectedSampler.mass` divided by the number of pairs. The unit
        depends on the source's unit system: internal units, units of
        :math:`M_0` in the model system, solar masses in the physical
        system. ``units["mass"]`` gives the unit.
    component : array of int, shape (n,)
        Index of the model component each pair was drawn from, in the order
        of the model's components.
    units : dict
        The unit of each of ``"r_2d"``, ``"v_los"`` and ``"mass"``. In the
        physical system, the unit is the Astropy unit's name, such as
        ``"pc"``. In the internal and the model systems, the unit is the
        system's name, ``"internal"`` or ``"model"``.
    record : dict
        The draw's record, with the keys ``"source"`` (the names of the
        model's components), ``"R_range"`` and ``"v_range"`` (the annulus
        and the velocity range, in internal units), ``"seed"``, ``"mass"``
        (`ProjectedSampler.mass`, in internal units), ``"shares"`` (each
        component's share of ``"mass"``) and ``"grids"`` (the sizes of the
        sampler's tables).
    validation : ValidationReport
        The validation report of the `ProjectedSampler` that drew the pairs.

    See Also
    --------
    ProjectedSampler : The sampler that draws the pairs.
    LosTable : The tables of the model that the sampler reads.

    Examples
    --------
    >>> import pianistpy as pp
    >>> from pianistpy.observables import ProjectedSampler
    >>> model = pp.solve(
    ...     pp.Component(pp.DoublePowerLaw.hernquist()),
    ...     settings=pp.Settings(quick=True),
    ... )
    >>> pairs = ProjectedSampler(pp.LosTable(model), (0.5, 2.0)).sample(10)
    >>> sorted(pairs.units)
    ['mass', 'r_2d', 'v_los']
    """

    r_2d: Array
    v_los: Array
    mass: Array
    component: Array
    units: dict[str, str]
    record: dict[str, Any] = field(default_factory=dict)
    validation: ValidationReport | None = None


def _error(check: Check) -> float:
    return check.error


def _expit(x: Array) -> Array:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def _chebyshev(n: int) -> tuple[Array, Array]:
    """Return the n Chebyshev nodes of the first kind on (0, 1), increasing,
    and the nodes' barycentric weights.

    The n nodes are a subset of the 3 n nodes, so tripling the node count
    keeps every existing node.
    """
    j = np.arange(n)
    angle = (2 * j + 1) * np.pi / (2 * n)
    return 0.5 * (1.0 - np.cos(angle)), (-1.0) ** j * np.sin(angle)


def _barycentric(x: Array, nodes: Array, weights: Array) -> Array:
    """Return the coefficients of the barycentric interpolant at points `x`.

    The coefficients have the shape (len(x), len(nodes)). The interpolated
    value at ``x[i]`` is the sum over ``j`` of coefficient ``[i, j]`` times
    the value at ``nodes[j]``. Where ``x[i]`` equals ``nodes[j]``, the
    coefficient ``[i, j]`` is 1 and every other coefficient of ``x[i]`` is 0.
    """
    d = x[:, None] - nodes[None, :]
    hit = d == 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        c = weights[None, :] / d
        c = c / c.sum(axis=1, keepdims=True)
    on = hit.any(axis=1)
    if on.any():
        c[on] = hit[on].astype(np.float64)
    return c


def _hermite(lam: Array, nodes: Array, s: Array, ds: Array) -> Array:
    """Return the cubic Hermite interpolant of one row's ``s`` at `lam`.

    The interpolant runs through the values `s` and the slopes `ds` at the
    uniform `nodes`. Beyond the first and the last node, the interpolant
    continues linearly with the slope at the end node.
    """
    h = nodes[1] - nodes[0]
    j = np.clip(np.floor((lam - nodes[0]) / h).astype(np.int64), 0, len(nodes) - 2)
    t = (lam - nodes[j]) / h
    t2, t3 = t * t, t * t * t
    val = (
        (2 * t3 - 3 * t2 + 1) * s[j]
        + (t3 - 2 * t2 + t) * ds[j] * h
        + (-2 * t3 + 3 * t2) * s[j + 1]
        + (t3 - t2) * ds[j + 1] * h
    )
    val = np.where(lam < nodes[0], s[0] + ds[0] * (lam - nodes[0]), val)
    return np.where(lam > nodes[-1], s[-1] + ds[-1] * (lam - nodes[-1]), val)


class _Row:
    """The line-of-sight velocity distribution at one projected radius, as the
    stencil interpolates it, with its exact cumulative distribution.

    `PanelTable.predict` returns the distribution of the velocity ``|v|`` on
    ``|v| >= 0`` as values at m Gauss-Legendre nodes on each sub-panel. On a
    sub-panel, the distribution is the polynomial of degree m - 1 through
    the m values. An m-point Gauss-Legendre rule integrates the sub-panel's polynomial
    exactly over a whole sub-panel or over part of a sub-panel. So the
    cumulative distribution ``H(|v|)`` is exact for the interpolated
    distribution. ``total`` is the integral of the distribution over
    ``|v| >= 0``, which is 1/2 for a distribution of ``v`` normalized to 1.
    ``v_end`` is the largest velocity of the distribution.
    """

    def __init__(self, stencil: PanelTable, route: Any, r: float, curves: dict) -> None:
        vm, wm, dm = stencil.predict(route, r, stencil.order, curves=curves)
        self.r = r
        m = stencil.lattice.rules.vp_nodes
        xg, wg = gl01(m)
        self.xg, self.wg = np.asarray(xg), np.asarray(wg)
        self.bw = np.array([1.0 / np.prod(xg[j] - np.delete(xg, j)) for j in range(m)])
        v = np.asarray(vm).reshape(-1, m)
        self.width = (v[:, -1] - v[:, 0]) / (xg[-1] - xg[0])
        self.lo = v[:, 0] - xg[0] * self.width
        self.hi = self.lo + self.width
        self.d = np.asarray(dm).reshape(-1, m)
        sums = (np.asarray(wm).reshape(-1, m) * self.d).sum(axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(sums)])
        self.total = float(self.cum[-1])
        self.v_end = float(self.hi[-1])
        self._tables: dict[int, tuple[Array, Array]] = {}

    def _poly(self, k: Array, t: Array) -> Array:
        """Return the polynomial of sub-panel ``k[i]`` at the unit positions
        ``t[i]`` on sub-panel ``k[i]``, with the shape (n, q) of `t`.

        A unit position is 0 at the sub-panel's lower end and 1 at the
        sub-panel's upper end.
        """
        dd = t[..., None] - self.xg
        hit = dd == 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            c = self.bw / dd
            val = np.sum(c * self.d[k][:, None, :], axis=-1) / np.sum(c, axis=-1)
        if hit.any():
            rows, cols, j = np.nonzero(hit)
            val[rows, cols] = self.d[k][rows, j]
        return val

    def cdf(self, a: ArrayLike) -> Array:
        """Return the cumulative distribution ``H(|a|)`` of the line-of-sight
        velocity, from 0 at ``|a| = 0`` to 1 at ``|a| = v_end`` and beyond."""
        a = np.abs(np.asarray(a, dtype=np.float64))
        out = np.ones(a.shape)
        live = a < self.v_end
        k = np.clip(
            np.searchsorted(self.lo, a[live], side="right") - 1, 0, len(self.lo) - 1
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(
                self.width[k] > 0.0, (a[live] - self.lo[k]) / self.width[k], 0.0
            )
        t = np.clip(t, 0.0, 1.0)
        part = self._poly(k, t[:, None] * self.xg[None, :]) @ self.wg
        out[live] = (self.cum[k] + part * t * self.width[k]) / self.total
        return out

    def table(self, lam: Array) -> tuple[Array, Array]:
        """Return ``s`` and ``ds / d lambda_u`` at the nodes `lam` of
        ``lambda_u``, on the exact cumulative distribution ``H``.

        At a node, ``u = 1 / (1 + exp(-lambda_u))`` is the target value of
        ``H``, and ``v`` is the velocity where ``H(v) = u``. The table holds
        ``s = ln(v / (v_end - v))``. A bracketed Newton solve finds ``v``
        inside one sub-panel. For ``lambda_u > 0``, the solve measures the
        mass from ``v_end`` downward, so that the solve keeps the relative
        precision of ``1 - u``. The slope ``ds / d lambda_u`` is computed from the
        distribution's value at ``v``. The result is cached by the number of
        nodes.

        Raises
        ------
        AccuracyError
            If the slope ``ds / d lambda_u`` at a node is not positive and
            finite.
        """
        key = len(lam)
        if key in self._tables:
            return self._tables[key]
        u, u_c = _expit(lam), _expit(-lam)
        top = lam > 0.0
        mass = np.where(top, np.asarray(u_c), np.asarray(u)) * self.total
        above = self.total - self.cum
        n_sub = len(self.lo)
        k_lo = np.searchsorted(self.cum, mass, side="right") - 1
        k_hi = np.searchsorted(-above, -mass, side="right") - 1
        k = np.clip(np.where(top, k_hi, k_lo), 0, n_sub - 1)
        rest = np.where(top, mass - above[k + 1], mass - self.cum[k])
        w = self.width[k]
        # tau is the unit position inside sub-panel k, measured from the end
        # of sub-panel k nearer to the end of the distribution the solve
        # starts from.
        tau, a, b = np.full(len(lam), 0.5), np.zeros(len(lam)), np.ones(len(lam))
        done = np.zeros(len(lam), dtype=bool)
        for _ in range(NEWTON_STEPS):
            x = tau[:, None] * self.xg[None, :]
            pos = np.where(top[:, None], 1.0 - x, x)
            f = w * tau * (self._poly(k, pos) @ self.wg) - rest
            a, b = np.where(f < 0.0, tau, a), np.where(f < 0.0, b, tau)
            here = np.where(top, 1.0 - tau, tau)
            slope = w * self._poly(k, here[:, None])[:, 0]
            step = tau - f / np.maximum(slope, TINY)
            new = np.where((step > a) & (step < b), step, 0.5 * (a + b))
            new = np.where(done, tau, new)
            done |= np.abs(new - tau) <= 4.0 * EPS * np.abs(new)
            tau = new
            if np.all(done):
                break
        v = np.where(top, self.hi[k] - w * tau, self.lo[k] + w * tau)
        gap = np.where(top, (self.v_end - self.hi[k]) + w * tau, self.v_end - v)
        s = np.log(v) - np.log(gap)
        here = np.where(top, 1.0 - tau, tau)
        density = self._poly(k, here[:, None])[:, 0] / self.total
        with np.errstate(divide="ignore", invalid="ignore"):
            ds = (u * u_c) * self.v_end / (density * v * gap)
        bad = ~np.isfinite(ds) | (ds <= 0.0)
        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            raise AccuracyError(
                f"The inverse of the cumulative distribution of the line-of-sight "
                f"velocity has no positive finite slope at lambda_u = "
                f"{float(lam[i]):.6e} (v = {float(v[i]):.6e}, density "
                f"{float(density[i]):.3e}). The slope of the inverse is computed from "
                f"the density at lambda_u = {float(lam[i]):.6e} and is not replaced "
                f"by a substitute value. The nodes of the inverse end at lambda_u = "
                f"+-{LAMBDA_END:g}, a constant of the method that no setting moves."
            )
        self._tables[key] = (s, ds)
        return s, ds


class _Piece:
    """An interval of a stretch on which the stencil uses one fixed set of
    rows.

    The piece spans ``[lo, hi]`` in the stencil's variable of the stretch.
    The piece holds ``n`` Chebyshev nodes in ``[lo, hi]`` and one `_Row` at
    the radius of each node. ``err_v`` and ``err_window`` hold the largest
    errors of the piece's last check, for the velocity table and for the
    cumulative probabilities at the ends of the velocity range.
    """

    def __init__(self, lo: float, hi: float) -> None:
        self.lo, self.hi = lo, hi
        self.n = N_PIECE_START
        self.rows: list[_Row] = []
        self.done = False
        self.err_v = Check(0.0)
        self.err_window = Check(0.0)

    def nodes(self, n: int) -> tuple[Array, Array]:
        x, w = _chebyshev(n)
        return self.lo + (self.hi - self.lo) * x, w


class _Stretch:
    """One stretch of the annulus between two successive radius breaks of
    the table, cut into pieces at the radii where the stencil's set of rows
    shifts.

    On each piece, the stretch tabulates the velocity table ``s`` at the
    piece's Chebyshev nodes in the stencil's variable ``theta``. Across a
    piece, ``s``, ``ln v_end`` and the cumulative probabilities at the ends
    of the velocity range are read by the barycentric interpolant through
    the nodes. Between the nodes in ``lambda_u``, ``s`` is read by the cubic
    Hermite interpolant of `_hermite`. Chebyshev nodes of the first kind
    exclude the piece's ends, so no node lies on a radius break.
    """

    def __init__(self, part: "_Part", st: int, r_a: float, r_b: float) -> None:
        self.part, self.st = part, st
        sten = part.stencil
        sel = np.flatnonzero(sten.stretch == st)
        first, last = int(sel[0]), int(sel[-1])
        k = min(int(sten.order), last - first + 1)
        moves = sten.R_2D[first + k // 2 : last - k + k // 2 + 1]
        moves = moves[(moves > r_a) & (moves < r_b)]
        edges = np.concatenate(
            [[self.coord(r_a)], [self.coord(r) for r in moves], [self.coord(r_b)]]
        )
        self.edges = edges
        self.pieces = [_Piece(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True)]
        self.lam = np.linspace(-LAMBDA_END, LAMBDA_END, N_LAMBDA_START)
        self._refine()

    def coord(self, r: float) -> float:
        sten = self.part.stencil
        return float(sten._coord(np.array([r]), np.array([self.st]))[0])

    def radius(self, theta: ArrayLike) -> Array:
        sten, st = self.part.stencil, self.st
        lt, ht = bool(sten._st_lt[st]), bool(sten._st_ht[st])
        th = np.asarray(theta, dtype=np.float64)
        if lt and ht:
            xi = 0.5 * (1.0 - np.cos(th))
        elif lt:
            xi = th * th
        elif ht:
            xi = 1.0 - (1.0 - th) ** 2
        else:
            xi = th
        lo, hi = sten._st_lo[st], sten._st_hi[st]
        return np.exp(lo + xi * (hi - lo))

    def dr_dtheta(self, theta: Array) -> Array:
        sten, st = self.part.stencil, self.st
        lt, ht = bool(sten._st_lt[st]), bool(sten._st_ht[st])
        th = np.asarray(theta, dtype=np.float64)
        if lt and ht:
            dxi = 0.5 * np.sin(th)
        elif lt:
            dxi = 2.0 * th
        elif ht:
            dxi = 2.0 * (1.0 - th)
        else:
            dxi = np.ones_like(th)
        return self.radius(th) * (sten._st_hi[st] - sten._st_lo[st]) * dxi

    def all_rows(self) -> list[_Row]:
        return [row for piece in self.pieces for row in piece.rows]

    def sizes(self) -> tuple[int, int, int]:
        """Return the number of pieces, the largest node count of one piece,
        and the node count in ``lambda_u``."""
        return len(self.pieces), max(p.n for p in self.pieces), len(self.lam)

    # The interpolants of one piece, built from the rows at the piece's nodes.

    def _values(self, piece: _Piece, theta: Array, lam: Array) -> Array:
        """Return the velocity ``|v|`` at the pairs ``(theta[i], lam[i])``, all
        inside one piece."""
        nodes, w = piece.nodes(piece.n)
        c = _barycentric(theta, nodes, w)
        s = np.zeros(len(theta))
        ln_v_end = np.zeros(len(theta))
        for j, row in enumerate(piece.rows):
            sj, dsj = row.table(self.lam)
            s += c[:, j] * _hermite(lam, self.lam, sj, dsj)
            ln_v_end += c[:, j] * np.log(row.v_end)
        return np.exp(ln_v_end) * _expit(s)

    def _window(self, piece: _Piece, theta: Array) -> tuple[Array, Array]:
        nodes, w = piece.nodes(piece.n)
        c = _barycentric(theta, nodes, w)
        ends = np.array([self.part.window_ends(row) for row in piece.rows])
        return c @ ends[:, 0], c @ ends[:, 1]

    def piece_of(self, theta: Array) -> Array:
        return np.clip(
            np.searchsorted(self.edges, theta, side="right") - 1,
            0,
            len(self.pieces) - 1,
        )

    def velocity(self, theta: Array, lam: Array) -> Array:
        out = np.empty(len(theta))
        idx = self.piece_of(theta)
        for i in np.unique(idx):
            sel = np.flatnonzero(idx == i)
            out[sel] = self._values(self.pieces[i], theta[sel], lam[sel])
        return out

    def window(self, theta: Array) -> tuple[Array, Array]:
        lo, hi = np.empty(len(theta)), np.empty(len(theta))
        idx = self.piece_of(theta)
        for i in np.unique(idx):
            sel = np.flatnonzero(idx == i)
            lo[sel], hi[sel] = self._window(self.pieces[i], theta[sel])
        return lo, hi

    def _refine(self) -> None:
        """Refine the nodes of every piece and the nodes in ``lambda_u``.

        A piece's node count triples, keeping every existing node, until
        one of two conditions holds: the piece's interpolants are within
        :math:`10^{-6}` of the exact rows at the new nodes; or a further
        tripling would exceed ``N_PIECE_MAX``. Then the node count in
        ``lambda_u`` doubles (n to 2 n - 1) until one of two conditions
        holds: the Hermite interpolant of ``s`` is within :math:`10^{-6}` at
        the midpoints between nodes, for every row of the stretch; or the
        node count reaches ``N_LAMBDA_MAX``. After each doubling in
        ``lambda_u``, the pieces are checked again.
        """
        first = [p.nodes(p.n)[0] for p in self.pieces]
        rows = self.part.rows(self.radius(np.concatenate(first)), self.st)
        at = 0
        for piece, nodes in zip(self.pieces, first, strict=True):
            piece.rows, at = rows[at : at + len(nodes)], at + len(nodes)
        while True:
            pending = [p for p in self.pieces if not p.done]
            while pending:
                news = []
                for p in pending:
                    nodes, _ = p.nodes(3 * p.n)
                    news.append(np.delete(nodes, np.arange(1, 3 * p.n, 3)))
                rows = self.part.rows(self.radius(np.concatenate(news)), self.st)
                at, still = 0, []
                u = _expit(self.lam)
                for p, check in zip(pending, news, strict=True):
                    check_rows, at = rows[at : at + len(check)], at + len(check)
                    err_v, err_w = Check(0.0), Check(0.0)
                    lo, hi = self._window(p, check)
                    for j, (th, row) in enumerate(zip(check, check_rows, strict=True)):
                        v = self._values(p, np.full(len(self.lam), th), self.lam)
                        e = float(np.max(np.abs(row.cdf(v) - u)))
                        if e > err_v.error:
                            err_v = Check(e, float(row.r))
                        g_lo, g_hi = self.part.window_ends(row)
                        e = max(abs(float(lo[j]) - g_lo), abs(float(hi[j]) - g_hi))
                        if e > err_w.error:
                            err_w = Check(e, float(row.r))
                    p.err_v, p.err_window = err_v, err_w
                    if (
                        max(err_v.error, err_w.error) <= TIER_BOUNDS[0]
                        or 3 * p.n > N_PIECE_MAX
                    ):
                        p.done = True
                        continue
                    merged: list[_Row] = []
                    old = iter(p.rows)
                    new = iter(check_rows)
                    for i in range(3 * p.n):
                        merged.append(next(old) if i % 3 == 1 else next(new))
                    p.rows, p.n = merged, 3 * p.n
                    still.append(p)
                pending = still
            self.err_lambda = Check(0.0)
            mids = 0.5 * (self.lam[1:] + self.lam[:-1])
            u_mid = _expit(mids)
            for row in self.all_rows():
                s, ds = row.table(self.lam)
                v = row.v_end * _expit(_hermite(mids, self.lam, s, ds))
                e = float(np.max(np.abs(row.cdf(v) - u_mid)))
                if e > self.err_lambda.error:
                    self.err_lambda = Check(e, float(row.r))
            if self.err_lambda.error <= TIER_BOUNDS[0] or len(self.lam) >= N_LAMBDA_MAX:
                return
            lam = np.empty(2 * len(self.lam) - 1)
            lam[0::2], lam[1::2] = self.lam, mids
            self.lam = lam
            for p in self.pieces:
                p.done = False

    def errors(self) -> tuple[Check, Check]:
        v = max([p.err_v for p in self.pieces] + [self.err_lambda], key=_error)
        return v, max((p.err_window for p in self.pieces), key=_error)


class _Part:
    """The tables of one component on the annulus.

    The part holds the component's stencil, the interpolant of the
    component's surface density, the stretches of the annulus, and the
    component's mass on the annulus in the velocity range.

    Raises
    ------
    InputError
        If the annulus ends at or inside the table's first row.
    InputError
        If the annulus starts at or beyond the table's last row.
    InputError
        If the share of the annulus's mass outside the table's rows exceeds
        ``INNER_DISK_TOLERANCE``.
    InputError
        If the annulus overlaps an interval that the table's check left
        above the table's tolerance.
    AccuracyError
        If the annulus starts inside the table's first row and the surface
        density's slope between the first two rows is not below 2
        (`SurfaceDensityReading.inner_disk`).
    AccuracyError
        If the velocity table has no positive finite slope at a node
        (`_Row.table`).
    """

    def __init__(
        self, part: LosTable, r_lo: float, r_hi: float, v_lo: float, v_hi: float
    ) -> None:
        self.route, self.stencil = _stencil_of(part)
        model = part.model
        projected = part.projected
        self.surface = surface_density_reader(
            projected.r_2d,
            projected.surface_density[0],
            float(model.profile.r_table_max),
            model.spatially_finite,
        )
        self.v_lo, self.v_hi = v_lo, v_hi
        sten = self.stencil
        first, last = float(sten.R_2D[0]), float(sten.R_max)
        if not (r_hi > first):
            raise InputError(
                f"The annulus [{r_lo:.6e}, {r_hi:.6e}] lies inside the table's "
                f"innermost tabulated radius {first:.6e}, which the setting "
                f"losvd.innermost_read_radius sets; the table is not extrapolated."
            )
        if not (r_lo < last):
            raise InputError(
                f"The annulus [{r_lo:.6e}, {r_hi:.6e}] lies beyond the table's "
                f"outermost tabulated radius {last:.6e}, which the setting "
                f"losvd.outer_mass_share sets; the table is not extrapolated."
            )
        self.r_lo, self.r_hi = max(r_lo, first), min(r_hi, last)
        # The annulus's mass outside the table's rows, which the sampler cuts
        # off. Inside the first row, the estimate adds the inner disk of the
        # surface density, as the aperture calls do.
        cut_in = (
            self.surface.inner_disk + self._sigma_mass(self.surface.r_min, first)
            if r_lo < first
            else 0.0
        )
        cut_out = self._sigma_mass(last, r_hi) if r_hi > last else 0.0
        inside = self._sigma_mass(self.r_lo, self.r_hi)
        self.cut = (cut_in + cut_out) / max(inside + cut_in + cut_out, TINY)
        if self.cut > INNER_DISK_TOLERANCE:
            raise InputError(
                f"The annulus [{r_lo:.6e}, {r_hi:.6e}] reaches beyond the table's "
                f"tabulated radii [{first:.6e}, {last:.6e}], and an estimated "
                f"fraction {self.cut:.2e} of the annulus's mass lies outside the "
                f"tabulated radii, more than the tolerance {INNER_DISK_TOLERANCE:g}; "
                f"the table is not extrapolated."
            )
        for (ua, ub), why in zip(sten.uncert, sten.uncert_why, strict=True):
            if ua < self.r_hi and ub > self.r_lo:
                raise InputError(
                    f"The annulus [{self.r_lo:.9e}, {self.r_hi:.9e}] overlaps the "
                    f"range [{ua:.9e}, {ub:.9e}], where the table's error estimate is "
                    f"above the tolerance; the range is "
                    + (why or "an end gap of a stretch of tabulated radii")
                    + "."
                )
        edges = [self.r_lo] + [
            float(b) for b in sten.R_breaks if self.r_lo < b < self.r_hi
        ]
        edges.append(self.r_hi)
        sts = np.searchsorted(
            sten.R_breaks, 0.5 * (np.array(edges[:-1]) + np.array(edges[1:]))
        )
        self.stretches = [
            _Stretch(self, int(st), a, b)
            for st, a, b in zip(sts, edges[:-1], edges[1:], strict=True)
        ]
        errors = [s.errors() for s in self.stretches]
        self.err_v = max((e[0] for e in errors), key=_error)
        self.err_window = max((e[1] for e in errors), key=_error)
        ends = np.array(
            [self.window_ends(row) for s in self.stretches for row in s.all_rows()]
        )
        self.full = bool(np.all(ends[:, 0] == 0.0) and np.all(ends[:, 1] == 1.0))
        self._mass_tables()

    def _sigma_mass(self, a: float, b: float) -> float:
        """Return the integral of ``Sigma R dR`` on ``[a, b]``, with ``[a, b]``
        clipped to the rows of the surface density's table."""
        a, b = max(a, self.surface.r_min), min(b, self.surface.r_max)
        if not b > a:
            return 0.0
        t, w = np.polynomial.legendre.leggauss(64)
        t = 0.5 * (t + 1.0)
        # The map r = a + (b - a) (1 - (1 - t)^2) clusters the nodes at the
        # upper end b, where Sigma can go as a square root of R_t - R.
        r = a + (b - a) * (1.0 - (1.0 - t) ** 2)
        jac = (b - a) * 2.0 * (1.0 - t)
        return float(np.sum(0.5 * w * jac * r * self.surface(r)))

    def rows(self, r: Array, st: int) -> list[_Row]:
        """Return one `_Row` at each radius in `r` of stretch `st`.

        The velocity breaks of the radii are scanned together, in chunks of
        ``CURVE_CHUNK`` radii.
        """
        out: list[_Row] = []
        for c in range(0, len(r), CURVE_CHUNK):
            rc = r[c : c + CURVE_CHUNK]
            curves = PanelTable.mid_curves(self.route, rc)
            out.extend(
                _Row(
                    self.stencil,
                    self.route,
                    float(ri),
                    {k: np.atleast_1d(v[i]) for k, v in curves.items()},
                )
                for i, ri in enumerate(rc)
            )
        return out

    def window_ends(self, row: _Row) -> tuple[float, float]:
        def g(v: float) -> float:
            if not np.isfinite(v):
                return 0.0 if v < 0.0 else 1.0
            return float(0.5 + 0.5 * np.sign(v) * row.cdf(np.array([abs(v)]))[0])

        return g(self.v_lo), g(self.v_hi)

    def _integrand(self, sx: _Stretch, theta: Array) -> Array:
        r = sx.radius(theta)
        f = np.ones(len(theta))
        if not self.full:
            lo, hi = sx.window(theta)
            f = np.clip(hi - lo, 0.0, 1.0)
        return 2.0 * np.pi * r * self.surface(r) * f * sx.dr_dtheta(theta)

    def _gauss(self, sx: _Stretch, a: Array, b: Array, n: int) -> Array:
        t, w = gl01(n)
        th = a[:, None] + (b - a)[:, None] * np.asarray(t)[None, :]
        val = self._integrand(sx, th.ravel()).reshape(th.shape)
        return (val * np.asarray(w)[None, :]).sum(axis=1) * (b - a)

    def _mass_tables(self) -> None:
        """Tabulate the cumulative mass at the piece ends of every stretch,
        and the relative error of the mass integral.

        The mass integral uses ``N_RADIUS_GAUSS`` Gauss-Legendre nodes on
        each piece. The error is the summed difference from the rule with
        ``N_RADIUS_CHECK`` nodes, relative to the total mass.
        """
        cums, err = [], 0.0
        for sx in self.stretches:
            a, b = sx.edges[:-1], sx.edges[1:]
            m = self._gauss(sx, a, b, N_RADIUS_GAUSS)
            m2 = self._gauss(sx, a, b, N_RADIUS_CHECK)
            cums.append(np.concatenate([[0.0], np.cumsum(m)]))
            err = max(err, float(np.sum(np.abs(m - m2))))
        offsets = np.concatenate([[0.0], np.cumsum([c[-1] for c in cums])])
        self.cums, self.offsets = cums, offsets
        total = float(offsets[-1])
        # The internal unit of density carries a factor 4 pi, so the mass in
        # internal units is the integral divided by 4 pi. Sampler's
        # particles carry their mass in the same unit.
        self.mass = total / (4.0 * np.pi)
        self.err_radius = err / total if total > 0.0 else 0.0

    def draw(self, q: Array, w: Array) -> tuple[Array, Array]:
        """Return the projected radii at the mass fractions `q`, and the
        line-of-sight velocities at the uniform numbers `w`.

        A bracketed Newton solve finds each pair's radius inside one piece.
        The uniform number ``w`` maps to the cumulative probability
        ``lo + w (hi - lo)``, with ``lo`` and ``hi`` the cumulative
        probabilities of ``v`` at the ends of the velocity range at the
        pair's radius.
        """
        target = q * self.offsets[-1]
        k = np.clip(
            np.searchsorted(self.offsets, target, side="right") - 1,
            0,
            len(self.stretches) - 1,
        )
        r, v = np.empty(len(q)), np.empty(len(q))
        for j, sx in enumerate(self.stretches):
            sel = np.flatnonzero(k == j)
            if not sel.size:
                continue
            local = target[sel] - self.offsets[j]
            cum = self.cums[j]
            i = np.clip(np.searchsorted(cum, local, side="right") - 1, 0, len(cum) - 2)
            start = sx.edges[i]
            a, b = start.copy(), sx.edges[i + 1].copy()
            rest = local - cum[i]
            th = 0.5 * (a + b)
            # Each pair stops at the pair's own convergence, so a pair's radius
            # does not depend on the other pairs of the draw.
            done = np.zeros(sel.size, dtype=bool)
            for _ in range(NEWTON_STEPS):
                g = self._gauss(sx, start, th, N_RADIUS_GAUSS) - rest
                a, b = np.where(g < 0.0, th, a), np.where(g < 0.0, b, th)
                step = th - g / np.maximum(self._integrand(sx, th), TINY)
                new = np.where((step > a) & (step < b), step, 0.5 * (a + b))
                new = np.where(done, th, new)
                done |= np.abs(new - th) <= 4.0 * EPS * np.maximum(np.abs(th), 1.0)
                th = new
                if np.all(done):
                    break
            r[sel] = sx.radius(th)
            if self.full:
                # The velocity range holds the whole distribution, so the
                # cumulative probabilities at the range's ends are 0 and 1.
                lo, hi = np.zeros(sel.size), np.ones(sel.size)
            else:
                lo, hi = sx.window(th)
            uu = lo + w[sel] * (hi - lo)
            half = np.abs(2.0 * uu - 1.0)
            with np.errstate(divide="ignore"):
                lam = np.log(half) - np.log1p(-half)
            v[sel] = np.sign(2.0 * uu - 1.0) * sx.velocity(th, lam)
        return r, v


class ProjectedSampler:
    r"""Sampler of pairs of projected radius and line-of-sight velocity.

    The sampler draws pairs :math:`(R, v)` of projected radius :math:`R`
    and line-of-sight velocity :math:`v` from the probability density
    :math:`p(R, v) \propto 2\pi R\,\Sigma(R)\,D(v\,|\,R)` on the annulus
    `R_range` and in the user-specified range `v_range` of line-of-sight
    velocity. :math:`\Sigma(R)` is the surface density, and
    :math:`D(v\,|\,R)` is the line-of-sight velocity distribution at
    :math:`R`. The distribution of :math:`R` is weighted at each radius by
    the share of :math:`D(v\,|\,R)` inside `v_range`. :math:`D(v\,|\,R)`
    is the distribution that the model's `LosTable` holds, for every
    anisotropy model, including Cuddeford-Osipkov-Merritt (COM) models.
    Call `ProjectedSampler` rather than `Sampler` to draw a projected sample
    on an annulus without drawing 3D particles.

    Parameters
    ----------
    source : Model or LosTable
        The model, of one component or a superposition of components, or the
        model's `LosTable`. Given a `Model`, the sampler builds the model's
        `LosTable` first. For several samplers of one model, pass
        ``pp.LosTable(model)``.
    R_range : (float, float)
        The annulus :math:`R_{\rm lo} \leq R_{\rm 2D} \leq R_{\rm hi}`, with
        :math:`0 \leq R_{\rm lo} < R_{\rm hi}`. The unit depends on the
        model's unit system (`Model.to`): units of the scale radius
        :math:`r_0` in the internal and the model systems, parsecs in the
        physical system. In the physical system, Astropy quantities of any
        length unit are also accepted. An end of the annulus inside the
        table's innermost projected radius moves to the innermost projected
        radius. An end beyond the table's outermost projected radius moves to
        the outermost projected radius. The sampler moves an end only if the
        annulus's mass cut off is at most :math:`10^{-6}` of the annulus's
        mass.
    v_range : (float, float), optional
        The user-specified range :math:`v_{\rm lo} < v < v_{\rm hi}` of
        signed line-of-sight velocity. The unit depends on the model's unit
        system: internal units, units of :math:`(G M_0/r_0)^{1/2}` in the
        model system, km/s in the physical system. Default: ``None``, which
        means the whole velocity distribution.

    Attributes
    ----------
    validation : ValidationReport
        The sampler's validation report. Each check is a dimensionless
        error. ``v_table`` is the error of the cumulative distribution of
        :math:`v` that the draw interpolates. ``window`` is the error of the
        cumulative probabilities at the ends of `v_range`.
        ``radius_quadrature`` is the difference of the annulus's mass from
        the mass by a finer quadrature, relative to the annulus's mass.
        ``row_norm`` is the largest difference of a tabulated velocity
        distribution's integral from 1. ``annulus_cut`` is the share of the
        annulus's mass cut off outside the table's projected radii.
    mass : float
        The annulus's mass in the velocity range, in internal units.

    Raises
    ------
    InputError
        If `source` is neither a `Model` nor a `LosTable`.
    InputError, AccuracyError, ResultError, WorkerError
        Given a `Model`, if `LosTable` raises the exception.
    AccuracyError
        If the `LosTable`'s validation report is above :math:`10^{-2}` and
        the settings do not set ``write_anyway=True``.
    InputError
        If `R_range` is not :math:`0 \leq R_{\rm lo} < R_{\rm hi}`.
    InputError
        If `v_range` is not :math:`v_{\rm lo} < v_{\rm hi}`.
    InputError
        If the annulus lies wholly inside the table's innermost projected
        radius or wholly beyond the table's outermost projected radius.
    InputError
        If the annulus's mass outside the table's projected radii is more
        than :math:`10^{-6}` of the annulus's mass.
    AccuracyError
        If `R_range` starts inside the table's innermost projected radius
        and the surface density's logarithmic slope between the table's
        two innermost projected radii is 2 or steeper, so that the mass
        inside the innermost projected radius diverges.
    InputError
        If the annulus overlaps an interval where the table did not meet the
        table's tolerance (``table.table.r_uncertified``).
    AccuracyError
        If the inverse cumulative distribution of :math:`|v|` at a tabulated
        radius has no positive finite slope.
    InputError
        If the velocity range holds none of the distribution on the
        annulus.
    AccuracyError
        If the sampler's validation report is above :math:`10^{-2}` and the
        settings do not set ``write_anyway=True``.

    Warns
    -----
    AccuracyWarning
        If the sampler's validation report, or the `LosTable`'s, is above
        :math:`10^{-4}` (``report.tier`` is ``"warned"``).

    See Also
    --------
    ProjectedPairs : The pairs that `ProjectedSampler.sample` returns.
    LosTable : The tables of a model, built once for many calls.
    losvd : The line-of-sight velocity distribution at projected radii.
    Sampler : The sampler of 3D particles.

    Examples
    --------
    >>> import pianistpy as pp
    >>> from pianistpy.observables import ProjectedSampler
    >>> model = pp.solve(
    ...     pp.Component(pp.DoublePowerLaw.hernquist()),
    ...     settings=pp.Settings(quick=True),
    ... )
    >>> sampler = ProjectedSampler(pp.LosTable(model), (0.5, 2.0))
    >>> pairs = sampler.sample(100, seed=1)
    >>> pairs.r_2d.shape, pairs.v_los.shape
    ((100,), (100,))
    >>> bool(((pairs.r_2d >= 0.5) & (pairs.r_2d <= 2.0)).all())
    True
    """

    def __init__(
        self,
        source: Model | LosTable,
        R_range: tuple[Any, Any],
        v_range: tuple[Any, Any] | None = None,
    ) -> None:
        if isinstance(source, LosTable):
            table = source
        elif isinstance(source, Model):
            table = LosTable(source)
        else:
            raise InputError(
                "The source of ProjectedSampler must be a Model or a LosTable; the "
                "source given is neither."
            )
        model = table.model
        refuse_unless_asked("The LosTable", table.validation, model.settings)
        r_lo, r_hi = (float(_internal_length(model, x)) for x in R_range)
        if not (0.0 <= r_lo < r_hi):
            raise InputError(
                "R_range must satisfy 0 <= R_lo < R_hi; the range given, in internal "
                f"units, is ({r_lo}, {r_hi})."
            )
        if v_range is None:
            v_lo, v_hi = -np.inf, np.inf
        else:
            v_lo, v_hi = (float(_internal_velocity(model, x)) for x in v_range)
            if not v_lo < v_hi:
                raise InputError(
                    "v_range must satisfy v_lo < v_hi; the range given, in internal "
                    f"units, is ({v_lo}, {v_hi})."
                )
        self.table, self.model = table, model
        self.R_range, self.v_range = (r_lo, r_hi), (v_lo, v_hi)
        self.parts = [_Part(p, r_lo, r_hi, v_lo, v_hi) for p in table.parts]
        self.weights = _weights(model) if len(self.parts) > 1 else np.ones(1)
        shares = self.weights * np.array([p.mass for p in self.parts])
        self.mass = float(shares.sum())
        if not self.mass > 0.0:
            raise InputError(
                f"The user-specified range [{v_lo:.6e}, {v_hi:.6e}] of line-of-sight "
                f"velocity holds none of the line-of-sight velocity distribution on "
                f"the annulus [{r_lo:.6e}, {r_hi:.6e}]."
            )
        self.shares = shares / self.mass
        norms = [
            Check(abs(2.0 * row.total - 1.0), row.r)
            for p in self.parts
            for sx in p.stretches
            for row in sx.all_rows()
        ]
        self.validation = ValidationReport(
            keys=PROJECTED_KEYS,
            checks={
                "v_table": max((p.err_v for p in self.parts), key=_error),
                "window": max((p.err_window for p in self.parts), key=_error),
                "radius_quadrature": Check(max(p.err_radius for p in self.parts)),
                "row_norm": max(norms, key=_error),
                "annulus_cut": Check(max(p.cut for p in self.parts)),
            },
            parents_level=table.validation.level,
        )
        refuse_unless_asked("The ProjectedSampler", self.validation, model.settings)
        _announce("The ProjectedSampler", self.validation)

    def sample(self, n: int, seed: int = 0) -> ProjectedPairs:
        """Return `n` pairs drawn from the annulus and the velocity range.

        Pair :math:`i` is drawn from random numbers of pair :math:`i` alone,
        as `Sampler` draws its particles. So at one `seed`, pair :math:`i` is
        the same for every `n` above :math:`i`. For a superposition of
        components, the component of each pair is drawn with probability
        equal to the component's share of the annulus's mass in the velocity
        range.

        Parameters
        ----------
        n : int
            Number of pairs.
        seed : int, optional
            Seed of the random numbers. Default: ``0``, which means that
            every call with ``seed=0`` draws the same pairs.

        Returns
        -------
        pairs : ProjectedPairs
            The pairs, in the source's unit system. Each pair carries the
            mass `mass` divided by `n`.

        Raises
        ------
        InputError
            If the source's unit system is not the internal system and the
            unit system's scales do not form conversion factors.

        See Also
        --------
        ProjectedPairs : The class of the returned pairs.
        """
        n = int(n)
        numbers = particle_stages.random_numbers_block(seed, 0, 0, n)
        q, w = np.asarray(numbers.radius), np.asarray(numbers.velocity_magnitude)
        comp = np.zeros(n, dtype=np.int64)
        if len(self.parts) > 1:
            u = particle_stages.component_numbers(seed, 0, n)
            comp = np.searchsorted(
                np.cumsum(self.shares)[:-1], np.asarray(u), side="right"
            )
        r, v = np.empty(n), np.empty(n)
        for k, part in enumerate(self.parts):
            sel = np.flatnonzero(comp == k)
            if sel.size:
                r[sel], v[sel] = part.draw(q[sel], w[sel])
        units = self.model.units
        profile = self.model.components[0].unit_profile
        out, names = {}, {}
        for name, kind, values in (
            ("r_2d", "length", r),
            ("v_los", "velocity", v),
            ("mass", "mass", np.full(n, self.mass / n)),
        ):
            converted = units.convert(kind, values, profile)
            names[name] = str(getattr(converted, "unit", units.system))
            out[name] = np.asarray(getattr(converted, "value", converted))
        return ProjectedPairs(
            r_2d=out["r_2d"],
            v_los=out["v_los"],
            mass=out["mass"],
            component=comp,
            units=names,
            record={
                "source": [c.name for c in self.model.components],
                "R_range": self.R_range,
                "v_range": self.v_range,
                "seed": int(seed),
                "mass": self.mass,
                "shares": self.shares.tolist(),
                "grids": [[sx.sizes() for sx in p.stretches] for p in self.parts],
            },
            validation=self.validation,
        )
