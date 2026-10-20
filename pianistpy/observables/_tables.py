"""The intrinsic and projected second moments of a model, from the model alone."""

__all__ = [
    "boundary_term",
    "edge_limit",
    "projected_table",
    "radial_table",
]

import math
from typing import Any

import numpy as np

from ..df import Model
from ..df.profile_checks import density_from_df
from ..runtime import log
from ..settings import Settings
from ._dispersion import radial_dispersion, radial_integral_reader
from ._projection import certify_projection, project

# The projected radii end this far inside the edge, where the surface density
# of a spatially finite model is not yet zero.
_EDGE_FRACTION = 0.9999
# The boundary term's integral starts this far below its upper limit; the
# auxiliary density vanishes as a positive power there, and the piece below
# is far under rounding.
_LOWER_FRACTION = 1e-20
# Gauss-Legendre nodes on each decade of the boundary term's integral.
_NODES_PER_DECADE = 64
# The radial dispersion's rows stall after this many rounds running without
# progress (the worst not halved and no fewer intervals missing): near the edge
# of a spatially finite model the worst holds for several rounds while the
# misses fall, and where the rule's own error is near the tolerance the misses
# double while the worst holds.
_PROFILE_IDLE_ROUNDS = 3
# The rounds of rows the radial dispersion is given at most.
_PROFILE_ROUNDS = 40
# The Gauss-Legendre nodes of each of the Jeans rule's panels it starts from,
# and their doublings before the rule is reported as not converged (512).
_PANEL_NODES = 8
_RULE_DOUBLINGS = 6


def _kernel_beta(model: Model) -> float:
    r"""The L-factor exponent b of the auxiliary density: :math:`\beta_{\rm c}`
    for the constant and COM classes, 0 for the isotropic and OM classes."""
    anisotropy = model.components[0].anisotropy
    return float(anisotropy.beta_c) if anisotropy.model in ("constant", "com") else 0.0


def boundary_term(model: Model, p_outer: float) -> float:
    r"""Return :math:`\int_0^{P_{\rm outer}}\varrho_{\rm aux}\,{\rm d}K` over the
    model's DF.

    Parameters
    ----------
    model : Model
        The model, of one component.
    p_outer : float
        The upper limit, the lowest potential of the profile's table, in
        internal units.

    Returns
    -------
    term : float
        The integral; zero on a spatially finite model, whose table ends where
        the density does.

    Notes
    -----
    :math:`\varrho_{\rm aux}(K)` is the density the DF gives at the potential
    ``K`` (`density_from_df`) divided by the class factor :math:`g` at the
    same radius. The integral runs in ln K by Gauss-Legendre on panels of a
    decade from :math:`10^{-20}P_{\rm outer}`, below which the auxiliary
    density's power law leaves nothing at rounding.
    """
    if model.spatially_finite or not p_outer > 0.0:
        return 0.0
    anisotropy = model.components[0].anisotropy
    # One radius where every class factor is finite: inside R_a for Type II.
    r_any = 0.5 * min(1.0, float(anisotropy.r_a))
    k_lo = _LOWER_FRACTION * p_outer
    n_panels = int(math.ceil(math.log10(p_outer / k_lo)))
    edges = np.linspace(math.log(k_lo), math.log(p_outer), n_panels + 1)
    x, w = np.polynomial.legendre.leggauss(_NODES_PER_DECADE)
    half = 0.5 * (edges[1:] - edges[:-1])
    ln_k = (0.5 * (edges[1:] + edges[:-1]))[:, None] + half[:, None] * x[None, :]
    k = np.exp(ln_k).ravel()
    radii = np.full(k.shape, r_any)
    rho_aux = density_from_df(radii, k, model.table, model.form, anisotropy) / (
        anisotropy.factor(radii)
    )
    return float(np.sum((half[:, None] * w[None, :]).ravel() * rho_aux * k))


def edge_limit(model: Model) -> float:
    r"""Return the limit of :math:`\sigma_r^2 / P` at the edge of a spatially
    finite model, or NaN where the DF states no power there.

    Notes
    -----
    With :math:`F \propto \mathcal{K}^p` at the edge, the auxiliary density goes
    as :math:`P^{n}`, ``n = p + 3/2 - b``, and the Jeans integral gives
    :math:`\sigma_r^2 \to P / (n + 1)`. ``p`` is the form's
    `exponent_at_zero` where it states one, and for the Taylor rule 1 where
    its first coefficient is non-zero and 2 where it is zero.
    """
    if not model.spatially_finite:
        return float("nan")
    form = model.form
    p = form.exponent_at_zero(0)
    if p is None and form.below_floor[0] == "taylor":
        p = 1.0 if form.below_floor_coefficients[0][0] != 0.0 else 2.0
    if p is None:
        return float("nan")
    n = p + 1.5 - _kernel_beta(model)
    return 1.0 / (n + 1.0)


def radial_table(model: Model, settings: Settings) -> dict[str, Any]:
    r"""Return the radial dispersion of a model on its rows, with what it read.

    Returns
    -------
    table : dict
        ``r``, ``sigma_r2`` (on the rows placed from the profile's first row to
        its last), ``boundary_term``, ``edge_limit``, ``p_outer``, ``r_t`` (the
        radius of the last row) and ``certificate``: ``tolerance``,
        ``n_jeans_nodes`` (the rule's nodes a panel the rows are placed
        against),
        ``rule_end`` (``"tolerance"`` or ``"stalled"``) and ``rule_change`` (the
        rule's last change between n and 2n nodes), ``end`` (``"tolerance"`` or
        ``"stalled"``), ``rounds``, ``rows``, ``worst_first``, ``worst_last``
        and ``uncertified`` (the intervals still above the tolerance when the
        rows stalled, as pairs of radii).

    Notes
    -----
    The rule (`radial_dispersion` with ``knots``) integrates once over panels
    between the profile's own table potentials, summed from ``p_outer``, and
    each row adds its part panel to the next one, so a row's value does
    not depend on the radii asked for with it. Its nodes a panel come first:
    from eight, doubled until :math:`\sigma_r` at the worst of the start's
    midpoints moves by less than ``profile_tolerance`` between n and 2n, at
    most six times; a rule that has not converged is logged and stated in the
    ``certificate`` record. The rows start on ``n_profile_start`` logarithmic
    radii. At every
    interval's midpoint in ln r, the reading (`radial_integral_reader`) is
    compared with the rule (`radial_dispersion`) in :math:`\sigma_r`; where it misses
    ``profile_tolerance`` the rule's value there becomes a row. Round by round
    until none misses, or three rounds running neither halve the worst, nor
    leave fewer intervals missing, nor halve the last interval's error (the
    march to a finite edge), or 40 rounds have run (stalled); the
    reading's slopes are the rule's own, so a row added changes the reading on
    its own two intervals only. A stall is logged with its intervals.
    """
    observables = settings.observables
    profile = model.profile
    anisotropy = model.components[0].anisotropy

    def density(r: Any) -> Any:
        return profile.density(r)[0]

    def rho_aux(r: Any) -> Any:
        return density(r) / anisotropy.factor(r)

    p_outer = float(profile.p_trunc[-1])
    r_t = float(profile.r_table_max)
    term = boundary_term(model, p_outer)
    limit = edge_limit(model)
    tolerance = float(observables["profile_tolerance"])

    panel_cache: dict[int, Any] = {}

    def rule_of(n_nodes: int, r: Any) -> Any:
        return radial_dispersion(
            r,
            profile.potential,
            density,
            profile.radius,
            anisotropy,
            p_outer,
            term,
            n_nodes,
            knots=profile.p_trunc,
            panel_cache=panel_cache,
        )[0]

    r = np.logspace(
        np.log10(float(profile.r[0])),
        np.log10(r_t),
        int(observables["n_profile_start"]),
    )

    # The rule's nodes a panel first: doubled until sigma_r at the worst of the
    # start's midpoints moves by less than the tolerance between n and 2n.
    n_nodes = _PANEL_NODES
    mid = np.exp(0.5 * (np.log(r[:-1]) + np.log(r[1:])))
    sigma_n = rule_of(n_nodes, mid)
    rule_end, doublings = "tolerance", 0
    while True:
        sigma_2n = rule_of(2 * n_nodes, mid)
        alive = (sigma_n > 0.0) & (sigma_2n > 0.0)
        rule_change = float(
            np.max(np.abs(np.sqrt(sigma_n[alive] / sigma_2n[alive]) - 1.0), initial=0.0)
        )
        if rule_change <= tolerance:
            break
        if doublings >= _RULE_DOUBLINGS:
            rule_end = "stalled"
            log.warning(
                "The radial dispersion's rule moved by %.2e of sigma_r between %d "
                "and %d Jeans nodes against the tolerance %.1e, after %d "
                "doublings; its rows are placed against %d nodes.",
                rule_change,
                n_nodes,
                2 * n_nodes,
                tolerance,
                doublings,
                n_nodes,
            )
            break
        n_nodes, sigma_n, doublings = 2 * n_nodes, sigma_2n, doublings + 1

    def rule(r: Any) -> Any:
        return rule_of(n_nodes, r)

    sigma_r2 = rule(r)
    worst: list[float] = []
    last: list[float] = []
    misses: list[int] = []
    end, rounds, idle, uncertified = "tolerance", 0, 0, []
    while True:
        read = radial_integral_reader(
            r, sigma_r2, profile.potential, rho_aux, model.spatially_finite, limit
        )
        mid = np.exp(0.5 * (np.log(r[:-1]) + np.log(r[1:])))
        sigma_mid = rule(mid)
        with np.errstate(divide="ignore", invalid="ignore"):
            error = np.where(
                sigma_mid > 0.0,
                np.abs(np.sqrt(np.maximum(read(mid), 0.0) / sigma_mid) - 1.0),
                0.0,
            )
        error = np.where(np.isfinite(error), error, np.inf)
        worst.append(float(np.max(error)))
        last.append(float(error[-1]))
        miss = error > tolerance
        misses.append(int(np.sum(miss)))
        if not np.any(miss):
            break
        if len(worst) >= 2:
            # the march to a finite edge holds the worst while the last
            # interval's error falls round by round: that is progress too
            progress = (
                worst[-1] <= 0.5 * worst[-2]
                or misses[-1] < misses[-2]
                or last[-1] <= 0.5 * last[-2]
            )
            idle = 0 if progress else idle + 1
        if idle >= _PROFILE_IDLE_ROUNDS or rounds >= _PROFILE_ROUNDS:
            end = "stalled"
            uncertified = [
                (float(a), float(b))
                for a, b in zip(r[:-1][miss], r[1:][miss], strict=True)
            ]
            log.warning(
                "The radial dispersion's rows stalled at %.2e of sigma_r against "
                "the tolerance %.1e after %d rounds; %d intervals uncertified, "
                "the first [%.6e, %.6e].",
                worst[-1],
                tolerance,
                rounds,
                len(uncertified),
                *uncertified[0],
            )
            break
        r = np.concatenate([r, mid[miss]])
        sigma_r2 = np.concatenate([sigma_r2, sigma_mid[miss]])
        order = np.argsort(r)
        r, sigma_r2 = r[order], sigma_r2[order]
        rounds += 1
    return dict(
        r=r,
        sigma_r2=sigma_r2,
        boundary_term=term,
        edge_limit=limit,
        p_outer=p_outer,
        r_t=r_t,
        certificate=dict(
            tolerance=tolerance,
            n_jeans_nodes=n_nodes,
            rule_end=rule_end,
            rule_change=rule_change,
            end=end,
            rounds=rounds,
            rows=int(r.size),
            worst_first=worst[0],
            worst_last=worst[-1],
            uncertified=uncertified,
        ),
    )


def projected_table(
    model: Model, read_sigma_r2: Any, settings: Settings, tolerance: float
) -> dict[str, Any]:
    r"""Return the projected table of a model, checked between rows.

    Parameters
    ----------
    model : Model
        The model, of one component.
    read_sigma_r2 : callable
        :math:`\sigma_r^2` at radii.
    settings : Settings
        Numerical settings.
    tolerance : float
        The check's relative tolerance between rows.

    Returns
    -------
    table : dict
        ``r_2d``, ``surface_density``, ``sigma_los2`` and ``certificate``.
    """
    observables = settings.observables
    profile = model.profile
    anisotropy = model.components[0].anisotropy
    r_t = float(profile.r_table_max)
    n_nodes = int(observables["n_projection_nodes"])

    def density(r: Any) -> Any:
        return profile.density(r)[0]

    def rule(radii: Any) -> Any:
        return project(radii, density, read_sigma_r2, r_t, anisotropy, n_nodes)

    r_2d = np.logspace(
        np.log10(float(profile.r[0])),
        np.log10(r_t * _EDGE_FRACTION),
        int(observables["n_radii"]),
    )
    surface_density, sigma_los2 = rule(r_2d)
    r_2d, surface_density, sigma_los2, certificate = certify_projection(
        r_2d,
        surface_density,
        sigma_los2,
        r_t,
        model.spatially_finite,
        rule,
        tolerance,
    )
    return dict(
        r_2d=r_2d,
        surface_density=surface_density,
        sigma_los2=sigma_los2,
        certificate=certificate,
    )
