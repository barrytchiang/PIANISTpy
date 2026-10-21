r"""The tables of the intrinsic and projected second moments of a model.

`sigma_r`, `sigma_t`, `sigma_los`, `sigma_los_aperture` and `LosTable` read
these tables through `stages.dispersion_profile` (which calls
`radial_table`) and `stages.projected_dispersion` (which calls
`projected_table`). `radial_table` tabulates the radial velocity dispersion
:math:`\sigma_r` from the Jeans integral

.. math::

    \varrho_{\rm aux}(R)\,\sigma_r^2(R) = \int_0^{P(R)}
    \varrho_{\rm aux}(K)\,{\rm d}K,

with :math:`R` the 3D radius, :math:`P(R)` the relative potential,
:math:`K` the energy variable of the DF and
:math:`\varrho_{\rm aux} = \varrho/g` the auxiliary density, where
:math:`\varrho` is the density and :math:`g(R)` is the separability factor
of the anisotropy model. Rows are added to the radial table until the
table's reading at every interval's midpoint differs from the Jeans
integral by at most ``profile_tolerance`` in :math:`\sigma_r`
(:math:`10^{-7}` in the default settings, a tenth of :math:`10^{-6}`).
`projected_table` tabulates the surface density and the squared
line-of-sight dispersion, and rows are added to the projected table until
the table reads itself between rows to the relative tolerance that the
caller passes (:math:`10^{-8}` from `stages.projected_dispersion`).
Notes Eq. (T:6.1) states the Jeans integral, and Notes Section T:6.2 states
the projection.
"""

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

# The projected table's starting radii end at _EDGE_FRACTION times the radius
# r_t of the profile's last row. The surface density of a spatially finite model
# is zero at r_t and positive at 0.9999 r_t.
_EDGE_FRACTION = 0.9999
# The lower limit of the boundary term's integral, as a fraction of its upper
# limit P_outer. The auxiliary density falls as a positive power of K toward
# K = 0, so the integral below 1e-20 P_outer is below 1e-20 of the integral
# for a pure power law, below the bound 1e-6.
_LOWER_FRACTION = 1e-20
# Gauss-Legendre nodes on each decade in K of the boundary term's integral.
_NODES_PER_DECADE = 64
# The radial table stalls after _PROFILE_IDLE_ROUNDS successive rounds without
# progress.
# A round makes progress when the round halves the largest midpoint error,
# leaves fewer intervals above profile_tolerance, or halves the error of the
# last interval. Near the edge of a spatially finite model, the largest error
# can hold for several rounds while the last interval's error falls.
_PROFILE_IDLE_ROUNDS = 3
# The largest number of rounds in which the radial table adds rows.
_PROFILE_ROUNDS = 40
# The Gauss-Legendre node count of each panel of the Jeans integral at the
# start, 8, and the largest number of doublings of that count, 6. After 6
# doublings (512 nodes) the node count is logged as not converged.
_PANEL_NODES = 8
_RULE_DOUBLINGS = 6


def _kernel_beta(model: Model) -> float:
    r"""Return the exponent :math:`b` of the factor :math:`(L^2)^{-b}` of a
    model's DF.

    :math:`b` is the central anisotropy :math:`\beta_{\rm c}` of the model's
    first component for the constant-anisotropy and COM models, and 0 for
    the isotropic and Osipkov-Merritt models."""
    anisotropy = model.components[0].anisotropy
    return float(anisotropy.beta_c) if anisotropy.model in ("constant", "com") else 0.0


def boundary_term(model: Model, p_outer: float) -> float:
    r"""Return the boundary term
    :math:`\int_0^{P_{\rm outer}}\varrho_{\rm aux}(K)\,{\rm d}K`, computed
    from the model's DF.

    Parameters
    ----------
    model : Model
        The model, of one component.
    p_outer : float
        The upper limit :math:`P_{\rm outer}`, the lowest relative potential
        of the profile's table, in internal units.

    Returns
    -------
    term : float
        The boundary term, in internal units. The boundary term is zero on a
        spatially finite model, whose profile table ends where the density
        ends, and zero where `p_outer` is not positive.

    Notes
    -----
    :math:`\varrho_{\rm aux}(K)` is the density that the DF gives at the
    relative potential :math:`K` (`density_from_df`), divided by the
    separability factor :math:`g` at the same radius. The function evaluates
    both at one radius, half the smaller of 1 and :math:`r_{\rm a}`, where
    :math:`g` is finite for every anisotropy model. The integral runs in
    :math:`\ln K` from :math:`10^{-20}P_{\rm outer}` to
    :math:`P_{\rm outer}`, on panels of one decade in :math:`K`, with 64
    Gauss-Legendre nodes on each panel.
    """
    if model.spatially_finite or not p_outer > 0.0:
        return 0.0
    anisotropy = model.components[0].anisotropy
    # The radius r_any lies inside R_a, so the separability factor of a Type II
    # model, like that of every other model, is finite at r_any.
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
    finite model.

    The result is NaN for a model that is not spatially finite, and NaN
    where the DF's form states no power :math:`p` of :math:`F` at
    :math:`K = 0`.

    Notes
    -----
    Near the edge, :math:`F \propto K^p`, so the auxiliary density goes as
    :math:`\varrho_{\rm aux} \propto P^n` with :math:`n = p + 3/2 - b`, where
    :math:`b` is the exponent of `_kernel_beta`. The Jeans integral then gives
    :math:`\sigma_r^2 \to P/(n + 1)` as :math:`P \to 0`. The power :math:`p`
    is the form's `exponent_at_zero` where the form states one. For a form
    whose rule below the floor is ``"taylor"``, :math:`p = 1` where the
    first Taylor coefficient is nonzero, and :math:`p = 2` where that
    coefficient is zero.
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
    r"""Return the table of the radial dispersion of a model, with its record.

    Parameters
    ----------
    model : Model
        The model, of one component.
    settings : Settings
        Numerical settings. The function reads ``profile_tolerance`` and
        ``n_profile_start`` from the ``observables`` group.

    Returns
    -------
    table : dict
        A dict with the following keys.

        - ``r``: the table's 3D radii, from the profile's first row to the
          profile's last row, in internal units.
        - ``sigma_r2``: :math:`\sigma_r^2` at ``r``, in internal units.
        - ``boundary_term``: the result of `boundary_term`.
        - ``edge_limit``: the result of `edge_limit`.
        - ``p_outer``: the lowest relative potential of the profile's table.
        - ``r_t``: the radius of the profile's last row.
        - ``certificate``: the record of the table, a dict described below.

        The record holds ``tolerance`` (``profile_tolerance``),
        ``n_jeans_nodes`` (the Gauss-Legendre node count of each panel of the
        Jeans integral), ``rule_end`` (``"tolerance"`` where the node count
        converged and ``"stalled"`` where the node count did not converge),
        ``rule_change`` (the last relative change of :math:`\sigma_r` between
        n and 2n nodes), ``end`` (``"tolerance"`` where every interval met
        ``profile_tolerance`` and ``"stalled"`` otherwise), ``rounds`` (the
        rounds that added rows), ``rows`` (the final number of rows),
        ``worst_first`` and ``worst_last`` (the largest midpoint error in the
        first round and in the last round), and ``uncertified`` (the
        intervals whose midpoint error is above ``profile_tolerance`` when the
        table stalls, as pairs of radii).

    Notes
    -----
    The Jeans integral (`radial_dispersion` with ``knots``) integrates over
    panels between the potentials of the profile's own table, summed from
    ``p_outer``. Each row adds the partial panel from its own potential to the
    next knot. A row's value therefore does not depend on the other radii of
    the call.

    The node count of each panel is set first. The count starts at 8 and
    doubles until :math:`\sigma_r` changes by at most ``profile_tolerance``
    between n and 2n nodes, at every midpoint of the starting radii. The count
    doubles at most 6 times. A node count that has not converged after 6
    doublings is logged and recorded in ``rule_end``.

    The rows start on ``n_profile_start`` logarithmically spaced radii. In
    each round, the function compares the table's reading
    (`radial_integral_reader`) with the Jeans integral in :math:`\sigma_r` at
    the midpoint in :math:`\ln r` of every interval. Where the relative
    difference exceeds ``profile_tolerance``, the Jeans integral's value at
    that midpoint becomes a new row. The reading is a cubic Hermite
    interpolant with exact slopes at the rows, so a new row changes the
    reading only on the two intervals next to the new row.

    The rounds end in one of three ways. The table meets the tolerance when
    no midpoint exceeds ``profile_tolerance``. The table stalls when 3
    successive rounds make no progress (`_PROFILE_IDLE_ROUNDS`). The table
    also stalls when 40 rounds have added rows. A stalled table is logged
    with the intervals above ``profile_tolerance``.
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

    # The node count of each panel is set first. The count doubles until the
    # largest relative change of sigma_r between n and 2n nodes, over the
    # midpoints of the starting radii, is at most profile_tolerance.
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
                "The radial dispersion sigma_r changes by the relative amount %.2e "
                "between %d and %d nodes of the Jeans integral, more than the "
                "tolerance %.1e, after %d doublings of the node count; the radii of "
                "the radial dispersion profile are placed with %d nodes.",
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
            # Near the edge of a spatially finite model, the largest error can
            # hold while the last interval's error falls from round to round.
            # A halving of the last interval's error therefore counts as
            # progress.
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
                "The interpolation of the radial dispersion sigma_r stopped "
                "improving at the relative error %.2e, above the tolerance %.1e, "
                "after %d rounds of added radii. The relative error is above the "
                "tolerance in %d intervals between tabulated radii; the first such "
                "interval is [%.6e, %.6e].",
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
    r"""Return the table of the surface density and the squared line-of-sight
    dispersion of a model.

    The table starts on ``n_radii`` logarithmically spaced projected radii,
    from the profile's first row to 0.9999 of the radius of the profile's
    last row. `certify_projection` then adds rows until the table, read at
    every interval's midpoint, differs from the line-of-sight integral
    (`project`) by at most `tolerance`.

    Parameters
    ----------
    model : Model
        The model, of one component.
    read_sigma_r2 : callable
        The function that returns :math:`\sigma_r^2` at 3D radii, in internal
        units.
    settings : Settings
        Numerical settings. The function reads ``n_projection_nodes`` and
        ``n_radii`` from the ``observables`` group.
    tolerance : float
        The largest relative difference allowed between the table's reading
        at an interval's midpoint and the line-of-sight integral at that
        midpoint.

    Returns
    -------
    table : dict
        A dict with the keys ``r_2d`` (the projected radii),
        ``surface_density``, ``sigma_los2`` (the squared line-of-sight
        dispersion), all in internal units, and ``certificate`` (the record
        of `certify_projection`).
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
