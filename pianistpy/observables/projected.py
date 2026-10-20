"""Pairs of projected radius and line-of-sight velocity drawn from a model.

The pairs follow p(R, v) ~ 2 pi R Sigma(R) D(v | R) on an annulus of
projected radius and in a window of line-of-sight velocity, D the
line-of-sight velocity distribution the model's `LosTable` holds. The draw
reads tables built at construction; no stencil is set up per pair.

The pieces. The table's stencil reads between its rows by a Lagrange
polynomial in its own variable of the stretch (`PanelTable._coord`) through
a window of rows that moves at the rows; between two moves the reading is
one smooth piece. The annulus is cut at the table's R-breaks and at the rows
where the window moves, and every table below is read inside one piece only.

The rows. At a radius the stencil's row (`PanelTable.predict`, the reading
between rows the table checked) is a polynomial through m Gauss nodes on
each sub-panel, so its cumulative is exact by an m-point rule on each
sub-panel and on the part of one, and its inverse is found inside one
sub-panel by a bracketed Newton from the nearer end.

The v table. On the radii of a piece (Chebyshev nodes of the first kind in
the stencil's variable, so that no radius lies on an R-break) the inverse of
the cumulative H of |v| is tabulated as the speed table tabulates its rows:
s = ln(v / (V_end - v)) at nodes uniform in lambda_u = ln(u / (1 - u)), with
its exact slope; s is read between nodes by cubic Hermite and across the
piece by its barycentric interpolant. The window's ends, G(v) = 1/2 +
sign(v) H(|v|) / 2, and ln V_end are read the same way.

The radius. The mass of the annulus in the window is integrated in the
stencil's variable, Gauss-Legendre on each piece, on the checked reading of
the surface density; a pair's radius solves its mass fraction by a bracketed
Newton inside its piece.

A piece's nodes are tripled (every node kept) until the reading meets the
goal against the exact cumulative at the new nodes; the nodes in lambda_u are
doubled (n to 2 n - 1) until the reading between them does at every radius.
"""

__all__ = ["ProjectedPairs", "ProjectedSampler"]

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..df import Model
from ..exceptions import InputError
from ..particles import stages as particle_stages
from ..typing import Array, ArrayLike
from ..validation import (
    GOAL,
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

# lambda_u's nodes cover [-LAMBDA_END, LAMBDA_END]: beyond, u or 1 - u is
# below 1e-12 and s is continued by its end slope.
LAMBDA_END = 28.0
N_LAMBDA_START, N_LAMBDA_MAX = 33, 1025
# a piece's nodes: from 4, tripled up to 108
N_PIECE_START, N_PIECE_MAX = 4, 108
# Gauss-Legendre nodes of the radius's integral on each piece, and the rule it
# is checked against.
N_RADIUS_GAUSS, N_RADIUS_CHECK = 8, 16
NEWTON_STEPS = 60
# radii whose breaks are scanned together (the scan holds 4001 points each)
CURVE_CHUNK = 1024
TINY = np.finfo(np.float64).tiny
EPS = np.finfo(np.float64).eps


@dataclass(frozen=True)
class ProjectedPairs:
    """Pairs drawn by `ProjectedSampler.sample`.

    Attributes
    ----------
    r_2d, v_los, mass : array, shape (n,)
        Projected radius, line-of-sight velocity and mass of each pair, in
        the source's unit system (named in `units`).
    component : array of int, shape (n,)
        The component each pair was drawn from.
    units : dict
        The unit of each quantity.
    record : dict
        The source, the ranges in internal units, the seed, the annulus's
        mass in the window and the components' shares, the tables' sizes.
    validation : ValidationReport
        The sampler's report.
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
    """Chebyshev nodes of the first kind on (0, 1), increasing, and their
    barycentric weights; the nodes of n are among those of 3 n."""
    j = np.arange(n)
    angle = (2 * j + 1) * np.pi / (2 * n)
    return 0.5 * (1.0 - np.cos(angle)), (-1.0) ** j * np.sin(angle)


def _barycentric(x: Array, nodes: Array, weights: Array) -> Array:
    """The weights, shape (len(x), len(nodes)), of the interpolant at x."""
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
    """Cubic Hermite of one row's s on uniform nodes, at lam; continued by the
    end slope beyond the nodes."""
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
    """The stencil's row at one radius, its cumulative exact per sub-panel."""

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
        """Sub-panel k's polynomial at its unit positions t, shape (n, q)."""
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
        """H(|v|): the cumulative of the speed along the line, 0 to 1."""
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
        """s and ds / d lambda_u at the nodes lam, on the exact H; each found
        inside its sub-panel, from the nearer end of the distribution."""
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
        # tau: the position inside sub-panel k measured from its near end
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
            ds = np.where(bad, np.gradient(s, lam), ds)
        self._tables[key] = (s, ds)
        return s, ds


class _Piece:
    """A piece of a stretch where the stencil's window of rows is fixed: its
    nodes in the stencil's variable and their rows."""

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
    """One stretch of the annulus between R-breaks, cut into pieces at the
    rows where the stencil's window moves."""

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
        """The pieces, the most nodes of one, and the nodes in lambda_u."""
        return len(self.pieces), max(p.n for p in self.pieces), len(self.lam)

    # the reading of one piece at its own nodes' rows

    def _values(self, piece: _Piece, theta: Array, lam: Array) -> Array:
        """|v| at (theta, lam) pairs inside one piece."""
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

    def speed(self, theta: Array, lam: Array) -> Array:
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
        """Triple each piece's nodes until its reading meets the goal at the new
        nodes; double the nodes in lambda_u until the reading between them
        does at every row; repeat while either moved."""
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
                    if max(err_v.error, err_w.error) <= GOAL or 3 * p.n > N_PIECE_MAX:
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
            if self.err_lambda.error <= GOAL or len(self.lam) >= N_LAMBDA_MAX:
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
    """One component: its stencil, surface density, stretches and mass."""

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
                f"The annulus [{r_lo:.6e}, {r_hi:.6e}] lies inside the table's first "
                f"row {first:.6e}, which losvd.innermost_read_radius sets; the table "
                f"is not extrapolated."
            )
        if not (r_lo < last):
            raise InputError(
                f"The annulus [{r_lo:.6e}, {r_hi:.6e}] lies beyond the table's last "
                f"row {last:.6e}, which losvd.outer_mass_share sets; the table is not "
                f"extrapolated."
            )
        self.r_lo, self.r_hi = max(r_lo, first), min(r_hi, last)
        # the parts cut at the table's ends, as the aperture's inner disk
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
                f"The annulus [{r_lo:.6e}, {r_hi:.6e}] reaches beyond the table's rows "
                f"[{first:.6e}, {last:.6e}] where an estimated {self.cut:.2e} of its "
                f"surface density lies, more than {INNER_DISK_TOLERANCE:g}; the table "
                f"is not extrapolated."
            )
        for (ua, ub), why in zip(sten.uncert, sten.uncert_why, strict=True):
            if ua < self.r_hi and ub > self.r_lo:
                raise InputError(
                    f"The annulus [{self.r_lo:.9e}, {self.r_hi:.9e}] overlaps "
                    f"[{ua:.9e}, {ub:.9e}], which the table's check left "
                    f"above the tolerance: " + (why or "an end gap of a stretch") + "."
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
        """The integral of Sigma R dR on [a, b], within the surface's rows."""
        a, b = max(a, self.surface.r_min), min(b, self.surface.r_max)
        if not b > a:
            return 0.0
        t, w = np.polynomial.legendre.leggauss(64)
        t = 0.5 * (t + 1.0)
        # a square root at the upper end, where Sigma has one at R_t
        r = a + (b - a) * (1.0 - (1.0 - t) ** 2)
        jac = (b - a) * 2.0 * (1.0 - t)
        return float(np.sum(0.5 * w * jac * r * self.surface(r)))

    def rows(self, r: Array, st: int) -> list[_Row]:
        """The stencil's rows at radii of stretch st, their breaks scanned
        together in chunks."""
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
        """The cumulative mass at the pieces' ends of every stretch, and the
        quadrature's check against the finer rule."""
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
        # the internal mass is the profile's (4 pi is in the density's unit),
        # as the particles of `Sampler` carry it
        self.mass = total / (4.0 * np.pi)
        self.err_radius = err / total if total > 0.0 else 0.0

    def draw(self, q: Array, w: Array) -> tuple[Array, Array]:
        """Radii at mass fractions q and the speeds at the window's numbers w."""
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
            # a pair stops at its own convergence, so its radius is the same
            # whatever pairs it is drawn with
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
                # the whole distribution: the window's numbers are 0 and 1
                lo, hi = np.zeros(sel.size), np.ones(sel.size)
            else:
                lo, hi = sx.window(th)
            uu = lo + w[sel] * (hi - lo)
            half = np.abs(2.0 * uu - 1.0)
            with np.errstate(divide="ignore"):
                lam = np.log(half) - np.log1p(-half)
            v[sel] = np.sign(2.0 * uu - 1.0) * sx.speed(th, lam)
        return r, v


class ProjectedSampler:
    r"""Draw pairs of projected radius and line-of-sight velocity.

    Pairs follow :math:`p(R, v) \propto 2\pi R\,\Sigma(R)\,D(v\,|\,R)` on
    the annulus `R_range` and in the window `v_range`, exactly conditioned:
    the radius's distribution carries the window's share at each radius.
    Exact for every anisotropy (the distribution is the line of sight's
    marginal, the angles integrated in the table).

    Parameters
    ----------
    source : Model or LosTable
        The model (its `LosTable` is built first) or its prepared table; of
        one component or a superposition.
    R_range : (float, float)
        The annulus, in the source's unit system (`Model.to`): internal units
        by default, parsecs (floats or Astropy quantities) in the physical
        system. An end beyond the table's rows is taken at the row where the
        surface density beyond it is within `INNER_DISK_TOLERANCE` of the
        annulus's, and refused otherwise.
    v_range : (float, float), optional
        The window of line-of-sight velocity, signed, in the source's unit
        system (km/s in the physical system); the whole distribution where
        None.

    Attributes
    ----------
    validation : ValidationReport
        ``v_table``: the reading's error in the cumulative, between grid
        radii and between nodes; ``window``: the window ends' reading;
        ``radius_quadrature``: the mass integral against a finer rule,
        relative; ``row_norm``: the rows' distance from unit integral;
        ``annulus_cut``: the share cut at the table's ends.
    mass : float
        The annulus's mass in the window, internal units.

    Raises
    ------
    InputError
        Where the annulus lies outside the table, overlaps an interval the
        table left uncertified, or the window holds none of the
        distribution.
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
            raise InputError("ProjectedSampler reads a Model or a LosTable.")
        model = table.model
        refuse_unless_asked("The LosTable", table.validation, model.settings)
        r_lo, r_hi = (float(_internal_length(model, x)) for x in R_range)
        if not (0.0 <= r_lo < r_hi):
            raise InputError(f"R_range needs 0 <= R_lo < R_hi; got ({r_lo}, {r_hi}).")
        if v_range is None:
            v_lo, v_hi = -np.inf, np.inf
        else:
            v_lo, v_hi = (float(_internal_velocity(model, x)) for x in v_range)
            if not v_lo < v_hi:
                raise InputError(f"v_range needs v_lo < v_hi; got ({v_lo}, {v_hi}).")
        self.table, self.model = table, model
        self.R_range, self.v_range = (r_lo, r_hi), (v_lo, v_hi)
        self.parts = [_Part(p, r_lo, r_hi, v_lo, v_hi) for p in table.parts]
        self.weights = _weights(model) if len(self.parts) > 1 else np.ones(1)
        shares = self.weights * np.array([p.mass for p in self.parts])
        self.mass = float(shares.sum())
        if not self.mass > 0.0:
            raise InputError(
                f"The window [{v_lo:.6e}, {v_hi:.6e}] holds none of the distribution "
                f"on the annulus [{r_lo:.6e}, {r_hi:.6e}]."
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
        """Draw `n` pairs.

        Pair i draws from its own key as `Sampler` does: the radius from the
        channel ``radius``, the velocity from ``speed`` and, of a
        superposition, the component (by its share of the annulus's mass in
        the window) from ``component``; so a draw is the same pair by pair
        for any n.

        Returns
        -------
        pairs : ProjectedPairs
        """
        n = int(n)
        numbers = particle_stages.random_numbers_block(seed, 0, 0, n)
        q, w = np.asarray(numbers.radius), np.asarray(numbers.speed)
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
