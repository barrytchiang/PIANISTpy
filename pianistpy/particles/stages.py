r"""Stages of the particle sampler and the stages' results.

`Sampler` (in `pianistpy.particles.core`) builds three tables once per model::

    radius_table, velocity_table, angle_table

and each draw of particles runs the stages::

    random_numbers -> draw_radius -> draw_angle -> draw_velocity
        -> draw_directions -> validate_particles

At each radius the phase-space density factorizes into the scaled velocity x and
the angle variate y = sin^2 phi, where phi is the angle between the velocity
and the radius vector (Notes Eq. (T:5.3)). The radius r follows the enclosed
mass fraction M(r) / M(r_e) inside the sampling's edge r_e; the scaled velocity
x = v / (2 Psi)^{1/2} follows p(x | Psi) proportional to
:math:`x^{2 - 2\beta_{\rm c}} F(\Psi (1 - x^2))`, where Psi is the relative
potential at r, :math:`\beta_{\rm c}` is the component's constant anisotropy
and F is the distribution function; and y follows
:math:`{\rm Beta}(1 - \beta_{\rm c}, 1/2)`. Every variable is drawn by
inversion of the variable's cumulative distribution, and no draw is rejected.
A particle's random numbers depend only on the seed, the component index and
the particle's index. A drawn particle's velocity must satisfy the phase-space
bound within BOUND_SAFETY times the bound's rounding floor (Notes Section
T:5.3).
"""

__all__ = [
    "AngleTable",
    "RandomNumbers",
    "RadiusTable",
    "VelocityTable",
    "angle_table",
    "component_numbers",
    "draw_angle",
    "draw_directions",
    "draw_radius",
    "draw_velocity",
    "radius_bounds",
    "radius_table",
    "random_numbers",
    "random_numbers_block",
    "velocity_bounds",
    "velocity_draw_tables",
    "velocity_table",
    "validate_particles",
]

import math
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy import special

from ..df import Model
from ..exceptions import AccuracyError, InputError, ResultError
from ..numerics import hermite_inverse
from ..settings import Settings
from ..typing import Array
from ..validation import Check, ValidationReport
from .velocity import build_velocity_table

# Relative tolerance within which a user-specified r_sample counts as the
# profile's last radius: an r_sample within a factor of 1 +/- 1e-12 of the last
# radius is sampled to the last radius, so an r_sample equal to the last radius
# up to rounding is accepted and is not interpolated.
PROFILE_END_SLACK = 1e-12
# The channel of a particle's random key from which each random variable is
# drawn. Each variable has a channel of its own, so the numbers of one variable
# do not depend on which other variables a draw uses.
RNG_CHANNELS = dict(
    radius=0,
    velocity_magnitude=1,
    angle=2,
    radial_sign=3,
    position=4,
    velocity=5,
    component=6,
)
# Smallest argument of a logarithm, and smallest norm of a direction, in the
# draws. The floor 1e-300 keeps the logarithm and the division finite where a
# uniform number is 0 or 1 and in the branch of jnp.where that a draw discards;
# at every other uniform number the floor changes no value.
LOG_ARGUMENT_MIN = 1e-300
# The velocity rebuilt from a particle's position and velocity arrays must
# satisfy the draw's bound v^2 + sigma xi v_t^2 <= 2 Psi within BOUND_SAFETY
# times the bound's rounding floor eps (1 + xi), where eps is the spacing of
# doubles at 1 and xi = (r / r_a)^2. BOUND_SAFETY = 1 accepts an excess up to
# the rounding floor itself.
BOUND_SAFETY = 1.0


# Number of bisection halvings that find the logit of the mass fraction at a
# radius bound. The logit spans about 83, twice ln(1e18) for
# sampling["tail_mass_min"] = 1e-18, and 100 halvings shrink an interval of
# width 83 below the spacing of doubles at every point of the logit's span.
RADIUS_BOUND_HALVINGS = 100
# Number of bisection halvings that find the logit of the velocity number at a
# velocity bound. The logit spans the velocity table's lambda_u range, also twice
# ln(1e18), so 100 halvings also shrink the interval below the spacing of
# doubles.
VELOCITY_BOUND_HALVINGS = 100


@dataclass(frozen=True)
class RadiusTable:
    """The inverse of the enclosed-mass fraction of each component, the table
    from which `draw_radius` draws radii.

    Parameters
    ----------
    lambda_u : array, shape (n,)
        Table nodes, uniform in lambda = ln(u / (1 - u)), the logit of the
        enclosed-mass fraction u.
    s : array, shape (n_components, n)
        s = ln(r / (r_e - r)), the logit of the radius r relative to the
        sampling's outer radius r_e, at each table node.
    ds_dlambda : array, shape (n_components, n)
        Exact slope ds / dlambda of `s` at the table nodes.
    r_sample : tuple of float
        Outer radius r_e of the sampling of each component, in internal units:
        the truncation radius of a spatially finite model, otherwise the
        user-specified radius.
    mass_sample : tuple of float
        Mass of each component inside `r_sample`, in internal units.
    """

    lambda_u: Array
    s: Array
    ds_dlambda: Array
    r_sample: tuple[float, ...]
    mass_sample: tuple[float, ...]


@dataclass(frozen=True)
class VelocityTable:
    r"""The inverse distribution of the scaled velocity on rows of constant
    potential, the table from which `draw_velocity` draws velocities.

    Parameters
    ----------
    lambda_p : array, shape (n_rows,)
        The rows, uniform in lambda_p = ln(Psi / (p_0 - Psi)), the logit of
        the relative potential Psi relative to the potential's central value
        p_0. The row count starts at ``sampling["n_rows"]`` and is refined
        from n rows to 2 n - 3 rows, every existing row kept, until the
        table's own error across rows is below ``sampling["row_error_goal"]``
        for every component, or until the next refinement would pass
        ``sampling["n_rows_max"]``. The check "row_error" in `validation` and
        the field `row_cap` record where the refinement stopped. The top row
        is the profile's largest potential strictly below p_0, at the depth
        p_0 - Psi that the profile holds for the top row's potential. Potentials that
        round to p_0 do not give distinct rows, whatever depth the profile
        holds for them.
    dlambda_p : float
        Spacing of the rows in `lambda_p`. The rows are formed from
        `dlambda_p`, and the draw reads `dlambda_p` directly, because the
        difference of two rows does not reproduce `dlambda_p` to the last bit.
    lambda_u : array, shape (n,)
        Table nodes of each row, uniform in lambda_u = ln(u / (1 - u)), the
        logit of the cumulative probability u of the scaled velocity.
    s : array, shape (n_components, n_rows, n)
        s = ln(x / (x_e - x)), the logit of the scaled velocity x relative to the
        support's edge x_e, at each component, row and table node.
    ds_dlambda_u : array, shape (n_components, n_rows, n)
        Exact slope ds / dlambda_u of `s` along each row.
    ds_dlambda_p : array, shape (n_components, n_rows, n)
        Exact slope ds / dlambda_p of `s` across rows at fixed cumulative
        probability.
    x_edge : array, shape (n_components, n_rows)
        Edge x_e of the support of the scaled velocity on each row.
    p_0 : float
        Central value p_0 of the relative potential, the upper end of the
        potentials that `lambda_p` maps.
    e_edge : tuple of float
        Energy at which each component's DF ends: 0 where the DF continues
        below the DF's table, else the DF table's lowest energy. A draw forms
        the support's edge as x_e = (1 - e_edge / Psi)^{1/2} (`draw_velocity`).
    validation : ValidationReport
        The table's two checks, listed in `VELOCITY_TABLE_KEYS`, each located at
        the potential Psi of the check's worst row. "row_error" is the
        table's own error across rows: every interior row is predicted from
        the row's two neighbors at twice the row spacing by the draw's
        interpolant, and the error is the larger of the miss in the
        cumulative probability and the miss relative to the scaled velocity.
        "polish_residual" is the largest absolute residual of a table node in
        lambda_u against the exact cumulative probability. A table with
        either check above :math:`10^{-2}` is refused unless the settings set
        ``write_anyway=True``.
    row_cap : str, optional
        Where the rows stopped at ``sampling["n_rows_max"]`` with the error
        across rows above ``sampling["row_error_goal"]``, the cause: the row
        count and the factor by which the error fell over the last doubling
        of the rows. The cause is not a check; the checks are in
        `validation`. Default: ``""``, which means the refinement stopped at
        ``sampling["row_error_goal"]``.
    """

    lambda_p: Array
    dlambda_p: float
    lambda_u: Array
    s: Array
    ds_dlambda_u: Array
    ds_dlambda_p: Array
    x_edge: Array
    p_0: float
    e_edge: tuple[float, ...]
    validation: ValidationReport
    row_cap: str = ""


@dataclass(frozen=True)
class AngleTable:
    r"""The inverse distribution of the velocity angle of each component, the
    table from which `draw_angle` draws angles.

    The angle phi between the velocity and the radius vector follows from the
    variate y = sin^2 phi, which is :math:`{\rm Beta}(1 - \beta_{\rm c}, 1/2)`
    distributed at every radius, where :math:`\beta_{\rm c}` is the component's
    constant anisotropy. The
    table holds the inverse of the Beta distribution from each end, where both
    ends are power laws. An isotropic component uses Beta(1, 1/2).

    Parameters
    ----------
    ln_u : array, shape (n,)
        Table nodes, uniform in the logarithm of the cumulative probability
        from each end, from ``sampling["tail_mass_min"]`` to 1/2.
    ln_y : array, shape (n_components, 2, n)
        Logarithm of the variate from each end: ``ln_y[:, 0]`` is ln y as a
        function of ln u, and ``ln_y[:, 1]`` is ln(1 - y) as a function of
        ln(1 - u).
    dln_y_dln_u : array, shape (n_components, 2, n)
        Exact slopes of `ln_y` at the table nodes, from the Beta density.
    """

    ln_u: Array
    ln_y: Array
    dln_y_dln_u: Array


@dataclass(frozen=True)
class RandomNumbers:
    """The random numbers of a set of particles.

    Parameters
    ----------
    radius : array, shape (n,)
        One uniform random number per particle for the radius.
    velocity_magnitude : array, shape (n,)
        One uniform random number per particle for the velocity magnitude.
    angle : array, shape (n,)
        One uniform random number per particle for the velocity angle.
    radial_sign : array, shape (n,)
        One uniform random number per particle for the sign of the radial
        velocity.
    position : array of uint32, shape (n, 2)
        The random key of each particle's position direction.
    velocity : array of uint32, shape (n, 2)
        The random key of each particle's tangential velocity direction. The
        three normal numbers of each direction are drawn from the direction's
        key inside `draw_directions`, in the compiled function that forms the
        direction, because the compiler rounds a direction formed from numbers
        drawn in a separate compiled function differently, by one unit in the
        last place.
    """

    radius: Array
    velocity_magnitude: Array
    angle: Array
    radial_sign: Array
    position: Array
    velocity: Array


def radius_table(
    model: Model, settings: Settings, r_sample: float | None = None
) -> RadiusTable:
    """Return the inverse of the enclosed-mass fraction of each component.

    Parameters
    ----------
    model : Model
        The model to sample.
    settings : Settings
        Numerical settings. The call reads ``sampling["radius_edge_refined"]``,
        ``sampling["tail_mass_min"]`` and ``sampling["n_radius_nodes"]``.
    r_sample : float, optional
        Outer radius of the sampling, in internal units. Required for a model
        that is not spatially finite; not accepted for a spatially finite
        model. Default: ``None``, which means the model's truncation radius.

    Returns
    -------
    table : RadiusTable
        The inverse of the enclosed-mass fraction of the model's one
        component.

    Raises
    ------
    InputError
        If the model has more than one component.
    InputError
        If `r_sample` is missing for a model that is not spatially finite,
        given for a spatially finite model, or outside the profile's radii.
    ResultError
        If the enclosed mass does not increase in the table's variables s and
        lambda, or the table's inverse is not finite and increasing.

    Notes
    -----
    With the mass fraction u = M(r) / M(r_e) inside the edge r_e = `r_sample`,
    the variables s = ln(r / (r_e - r)) and lambda = ln(u / (1 - u)) follow
    power laws in each other at both ends, the center and the edge, whether
    the edge is smooth (the truncation radius, where the density vanishes) or
    sharp (a user-specified radius, where the density does not vanish). The
    mass fraction is exact at the profile's radii, and the slope
    du/dr = 4 pi r^2 rho / M(r_e) is exact everywhere. The inverse s(lambda)
    is tabulated at nodes uniform in lambda with the inverse's exact slope.
    Profile radii where 1 - u is below ``sampling["radius_edge_refined"]``
    (:math:`10^{-10}`) are left out, because 1 - u there is the difference of
    two nearly equal numbers. At a sharp edge, the last interval is refined at
    halved distances from the edge by the cubic Hermite interpolant of 1 - u
    in r_e - r, whose two end values and two end slopes are exact.
    """
    if len(model.components) != 1:
        raise InputError(
            "The radius table of a model of several components needs each "
            "component's enclosed mass, which the profile does not hold."
        )
    sampling = settings.sampling
    r, _, rho, mass, r_edge, smooth = _sampled_profile(model, r_sample)
    mass_edge = float(mass[-1])
    u = mass / mass_edge
    du_dr = 4.0 * np.pi * r**2 * rho / mass_edge
    q = 1.0 - u
    q_min = sampling["radius_edge_refined"]
    kept = (
        (u > 0.0)
        & (du_dr > 0.0)
        & np.concatenate([[True], np.diff(u) > 0.0])
        & (q >= q_min)
        & (r < r_edge)
    )
    r_n, u_n, q_n, g_n = r[kept], u[kept], q[kept], du_dr[kept]
    d_n = r_edge - r_n
    if (not smooth) and du_dr[-1] > 0.0 and q_n[-1] > q_min:
        d_0, q_0, g_0, g_e = (
            float(d_n[-1]),
            float(q_n[-1]),
            float(g_n[-1]),
            float(du_dr[-1]),
        )
        t = 0.5 ** np.arange(1, int(math.ceil(math.log2(q_0 / q_min))) + 2)
        t2, t3 = t * t, t * t * t
        # The last interval's cubic Hermite interpolant of 1 - u in r_e - r,
        # with 1 - u = 0 at r_e.
        q_k = (
            (t3 - 2 * t2 + t) * d_0 * g_e
            + (-2 * t3 + 3 * t2) * q_0
            + (t3 - t2) * d_0 * g_0
        )
        g_k = (
            (3 * t2 - 4 * t + 1) * g_e
            + (-6 * t2 + 6 * t) * q_0 / d_0
            + (3 * t2 - 2 * t) * g_0
        )
        k = (q_k >= q_min) & (g_k > 0.0)
        d_n = np.concatenate([d_n, t[k] * d_0])
        q_n = np.concatenate([q_n, q_k[k]])
        g_n = np.concatenate([g_n, g_k[k]])
        r_n = np.concatenate([r_n, r_edge - t[k] * d_0])
        u_n = np.concatenate([u_n, 1.0 - q_k[k]])
    s_n = np.log(r_n) - np.log(d_n)
    lambda_n = np.log(u_n) - np.log(q_n)
    dlambda_ds_n = g_n * r_n * d_n / (r_edge * u_n * q_n)
    if not (
        np.all(np.diff(s_n) > 0.0)
        and np.all(np.diff(lambda_n) > 0.0)
        and np.all(dlambda_ds_n > 0.0)
    ):
        raise ResultError(
            "The enclosed mass is not increasing in (s, lambda) on the profile's rows."
        )
    lambda_end = -math.log(sampling["tail_mass_min"])
    lambda_u = np.linspace(-lambda_end, lambda_end, sampling["n_radius_nodes"])
    s, _ = hermite_inverse(lambda_u, s_n, lambda_n, dlambda_ds_n)
    # The exact slope of the inverse: du/dr between the profile's radii comes
    # from linear interpolation of ln(du/dr) in s, and beyond the first and the
    # last profile radius the slope is the end radius's slope.
    u_t, q_t = 1.0 / (1.0 + np.exp(-lambda_u)), 1.0 / (1.0 + np.exp(lambda_u))
    r_t, d_t = r_edge / (1.0 + np.exp(-s)), r_edge / (1.0 + np.exp(s))
    g_t = np.exp(np.interp(s, s_n, np.log(g_n)))
    ds_dlambda = r_edge * u_t * q_t / (r_t * d_t * g_t)
    ds_dlambda = np.where(
        s < s_n[0],
        1.0 / dlambda_ds_n[0],
        np.where(s > s_n[-1], 1.0 / dlambda_ds_n[-1], ds_dlambda),
    )
    if not (
        np.all(np.isfinite(s))
        and np.all(np.diff(s) > 0.0)
        and np.all(np.isfinite(ds_dlambda))
        and np.all(ds_dlambda > 0.0)
    ):
        raise ResultError(
            "The inverse of the enclosed mass is not finite and increasing on the "
            "radius table's nodes."
        )
    return RadiusTable(
        lambda_u=lambda_u,
        s=s[None, :],
        ds_dlambda=ds_dlambda[None, :],
        r_sample=(r_edge,),
        mass_sample=(mass_edge / (4.0 * np.pi),),
    )


def _sampled_profile(
    model: Model, r_sample: float | None
) -> tuple[Array, Array, Array, Array, float, bool]:
    """Return the profile's columns out to the sampling's edge as
    (r, psi, rho, mass, r_edge, smooth).

    The mass column is 4 pi times the profile's enclosed mass. A model that is
    not spatially finite ends at `r_sample`, whose column values come from
    linear interpolation of the columns' logarithms in ln r, and the model's
    edge is sharp (smooth is False). A spatially finite model ends at the
    model's truncation radius, a smooth edge (smooth is True).
    """
    profile = model.profile
    r, psi, rho = profile.r, profile.p_trunc, profile.rho_trunc[0]
    mass = 4.0 * np.pi * profile.mass_trunc
    if model.spatially_finite:
        if r_sample is not None:
            raise InputError(
                "A spatially finite model is sampled to the model's edge, so "
                "r_sample must not be given."
            )
        return r, psi, rho, mass, float(profile.r_t), True
    if r_sample is None:
        raise InputError(
            "A model that is not spatially finite needs r_sample, the outer "
            "radius of the sampling."
        )
    if not (r[0] < r_sample <= r[-1] * (1.0 + PROFILE_END_SLACK)):
        raise InputError(
            f"r_sample = {r_sample!r} lies outside the profile's radii, the "
            f"interval ({r[0]!r}, {r[-1]!r}]."
        )
    if r_sample < r[-1] * (1.0 - PROFILE_END_SLACK):
        log_r, log_cut = np.log(r), np.log(r_sample)
        keep = r < r_sample
        r, psi, rho, mass = (
            np.concatenate(
                [
                    column[keep],
                    [float(np.exp(np.interp(log_cut, log_r, np.log(column))))],
                ]
            )
            for column in (r, psi, rho, mass)
        )
        r[-1] = r_sample
    return r, psi, rho, mass, float(r[-1]), False


def velocity_table(model: Model, settings: Settings) -> VelocityTable:
    """Return the inverse distribution of the scaled velocity on rows of constant
    potential.

    The density of the scaled velocity depends on the radius only through the
    potential, so one table serves every radius and every angle.

    Parameters
    ----------
    model : Model
        The model to sample.
    settings : Settings
        Numerical settings. The call reads the group ``sampling``.

    Returns
    -------
    table : VelocityTable
        The velocity table of the model's one component.

    Raises
    ------
    InputError
        If the model has more than one component.
    AccuracyError
        If a row cannot be resolved, more than :math:`10^{-6}` of the model's
        mass lies at potentials below the lowest row, or a check of the table
        is above :math:`10^{-2}` and the settings do not set
        ``write_anyway=True``.
    ResultError
        If a row is not finite and increasing, or the profile's potentials do
        not span the rows.
    UnphysicalModelError
        If the DF below its table has a mass that is not finite at zero
        energy.

    References
    ----------
    Notes Eq. (T:5.2), Notes Eq. (T:5.3), Notes Section T:5.2.1.

    Notes
    -----
    `pianistpy.particles.velocity.build_velocity_table` builds the table.
    """
    if len(model.components) != 1:
        raise InputError(
            "The velocity table is built for models of one component; the model "
            "given has several components."
        )
    return VelocityTable(**build_velocity_table(model, settings))


def angle_table(model: Model, settings: Settings) -> AngleTable:
    """Return the inverse distribution of the velocity angle.

    For each component, y = sin^2 phi, where phi is the angle between the
    velocity and the radius vector, is Beta(a, 1/2) distributed with
    ``a = 1 - beta_c`` (Notes Eq. (T:5.1)), and 1 - y is Beta(1/2, a)
    distributed. Each half of the inverse is tabulated against the logarithm
    of the half's own end probability, where both ends are power laws. The
    slope d ln y / d ln u of the inverse is u / (y p(y)) in closed form, where
    p(y) is the Beta density.

    Parameters
    ----------
    model : Model
        The model to sample.
    settings : Settings
        Numerical settings. The call reads ``sampling["n_angle_nodes"]`` and
        ``sampling["tail_mass_min"]``.

    Returns
    -------
    table : AngleTable
        The inverse Beta distributions of every component.

    Raises
    ------
    ResultError
        If the inverse of a component's Beta distribution is not finite and
        increasing on the table nodes.
    """
    sampling = settings.sampling
    ln_u = np.linspace(
        math.log(sampling["tail_mass_min"]), math.log(0.5), sampling["n_angle_nodes"]
    )
    u = np.exp(ln_u)
    ln_y = np.empty((len(model.components), 2, ln_u.size))
    dln_y_dln_u = np.empty_like(ln_y)
    for i, component in enumerate(model.components):
        a = 1.0 - component.anisotropy.beta_c
        ln_beta = special.betaln(a, 0.5)
        y = special.betaincinv(a, 0.5, u)
        z = special.betaincinv(0.5, a, u)
        with np.errstate(divide="ignore", invalid="ignore"):
            ln_y_low, ln_z = np.log(y), np.log(z)
        if not (
            np.all(np.isfinite(ln_y_low))
            and np.all(np.isfinite(ln_z))
            and np.all(np.diff(y) > 0)
            and np.all(np.diff(z) > 0)
        ):
            raise ResultError(
                f"The inverse of Beta({a}, 1/2) of component {i} is not finite and "
                "increasing on the angle table's nodes."
            )
        ln_y[i, 0], ln_y[i, 1] = ln_y_low, ln_z
        dln_y_dln_u[i, 0] = np.exp(ln_u - (a * ln_y_low - 0.5 * np.log1p(-y) - ln_beta))
        dln_y_dln_u[i, 1] = np.exp(
            ln_u - (0.5 * ln_z + (a - 1.0) * np.log1p(-z) - ln_beta)
        )
    return AngleTable(ln_u=ln_u, ln_y=ln_y, dln_y_dln_u=dln_y_dln_u)


def random_numbers(seed: int, component: int, n: int) -> RandomNumbers:
    """Return the random numbers of the particles of one component.

    The random numbers of particle `i` depend only on the seed, the component
    index and `i`. The numbers are equal to the bit on every machine and
    backend, for any block size and any number of cores, and the first `n`
    particles of a larger draw are the particles of a draw of `n` particles.
    The model, the settings and the package version of a draw are recorded in
    the draw's result, not in the random keys.

    Parameters
    ----------
    seed : int
        Seed of the draw.
    component : int
        Index of the component.
    n : int
        Number of particles.

    Returns
    -------
    numbers : RandomNumbers
        The uniform numbers and the direction keys of particles 0 to n - 1.

    Notes
    -----
    Particle `i` draws from the key k_i = fold_in(base, i), with
    base = fold_in(PRNGKey(seed), component), and each random variable draws
    from the variable's own channel of k_i (``RNG_CHANNELS``); the two
    directions are stored as keys. The keys are formed by integer arithmetic,
    so the keys and the uniform numbers are equal to the bit on every backend;
    quantities computed from the uniform numbers round differently on
    different backends.
    """
    return random_numbers_block(seed, component, 0, n)


def random_numbers_block(
    seed: int, component: int, start: int, n: int
) -> RandomNumbers:
    """Return the random numbers of particles `start` to `start` + `n` - 1.

    Each particle's key is formed from the particle's own index, so the
    numbers equal the numbers that particles `start` to `start` + `n` - 1
    carry in `random_numbers` of any larger draw.
    """
    return _random_numbers_from(
        jax.random.fold_in(jax.random.PRNGKey(seed), component), n, start
    )


def component_numbers(seed: int, start: int, n: int) -> Array:
    """Return, for each of particles `start` to `start` + `n` - 1, the uniform
    number that chooses the particle's component in a superposed model. The
    number comes from the channel ``component`` of the particle's own key,
    beside the channels of the particle's draw, so the particle's other
    numbers equal the numbers of the particle in a draw of component index
    0."""
    base = jax.random.fold_in(jax.random.PRNGKey(seed), 0)
    keys = jax.vmap(lambda i: jax.random.fold_in(base, i))(
        jnp.arange(start, start + n, dtype=jnp.int32)
    )
    return np.asarray(_channel_uniform(keys, RNG_CHANNELS["component"]))


def _random_numbers_from(base: Array, n: int, start: int = 0) -> RandomNumbers:
    """Return the random numbers of particles `start` to `start` + `n` - 1,
    drawn from the base key `base`."""
    keys = jax.vmap(lambda i: jax.random.fold_in(base, i))(
        jnp.arange(start, start + n, dtype=jnp.int32)
    )

    def uniform(channel):
        return np.asarray(_channel_uniform(keys, channel))

    def channel_keys(channel):
        return np.asarray(jax.vmap(lambda key: jax.random.fold_in(key, channel))(keys))

    return RandomNumbers(
        radius=uniform(RNG_CHANNELS["radius"]),
        velocity_magnitude=uniform(RNG_CHANNELS["velocity_magnitude"]),
        angle=uniform(RNG_CHANNELS["angle"]),
        radial_sign=uniform(RNG_CHANNELS["radial_sign"]),
        position=channel_keys(RNG_CHANNELS["position"]),
        velocity=channel_keys(RNG_CHANNELS["velocity"]),
    )


def _channel_uniform(keys: Array, channel: int) -> Array:
    """Return one uniform number per key of `keys`, drawn from channel
    `channel` of each key."""
    return jax.vmap(
        lambda key: jax.random.uniform(
            jax.random.fold_in(key, channel), dtype=jnp.float64
        )
    )(keys)


def _hermite_uniform(x, x_0, x_1, f, m):
    """Return at `x` the cubic Hermite interpolant through values `f` and
    slopes `m` on nodes uniform over [x_0, x_1].

    The interval that holds `x` is found by index arithmetic, without a
    search. An `x` outside [x_0, x_1] is clamped to the nearer end, so the end
    node's value holds beyond each end.
    """
    n = f.shape[0]
    h = (x_1 - x_0) / (n - 1)
    sx = (jnp.clip(x, x_0, x_1) - x_0) / h
    i = jnp.clip(jnp.floor(sx).astype(jnp.int32), 0, n - 2)
    t = sx - i
    t2, t3 = t * t, t * t * t
    return (
        (2.0 * t3 - 3.0 * t2 + 1.0) * f[i]
        + (t3 - 2.0 * t2 + t) * h * m[i]
        + (-2.0 * t3 + 3.0 * t2) * f[i + 1]
        + (t3 - t2) * h * m[i + 1]
    )


def _logit(u):
    """Return ln(u / (1 - u)) of a uniform number `u`; the difference 1 - u
    is exact for ``u >= 1/2``."""
    return jnp.log(jnp.maximum(u, LOG_ARGUMENT_MIN)) - jnp.log(
        jnp.maximum(1.0 - u, LOG_ARGUMENT_MIN)
    )


@jax.jit
def _radius_kernel(u, lambda_0, lambda_1, s, ds_dlambda, r_edge):
    def one(u_i):
        s_i = _hermite_uniform(_logit(u_i), lambda_0, lambda_1, s, ds_dlambda)
        return r_edge / (1.0 + jnp.exp(-s_i))

    return jax.vmap(one)(u)


@jax.jit
def _radius_bounds_kernel(s_target, lambda_0, lambda_1, s, ds_dlambda):
    """Return the logit of the mass fraction at which the table's radius
    logit reaches each target `s_target`, by bisection on the table's own
    cubic Hermite interpolant."""

    def halve(_, ends):
        lo, hi = ends
        mid = 0.5 * (lo + hi)
        below = _hermite_uniform(mid, lambda_0, lambda_1, s, ds_dlambda) < s_target
        return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

    lo, hi = jax.lax.fori_loop(
        0,
        RADIUS_BOUND_HALVINGS,
        halve,
        (jnp.full_like(s_target, lambda_0), jnp.full_like(s_target, lambda_1)),
    )
    return 0.5 * (lo + hi)


def radius_bounds(
    table: RadiusTable, component: int, r_range: tuple[float, float]
) -> tuple[float, float]:
    """Return the uniform numbers that bound the radius draw to a
    user-specified range.

    Parameters
    ----------
    table : RadiusTable
        The radius table.
    component : int
        Index of the component.
    r_range : tuple of float
        The user-specified radii (r_lo, r_hi), in internal units, with
        ``0 <= r_lo < r_hi`` and ``r_hi`` at most the table's ``r_sample``.

    Returns
    -------
    u_lo : float
        The mass fraction at which the table's own inverse reaches r_lo; 0
        where r_lo lies at or below the table's first radius.
    u_hi : float
        The mass fraction at which the table's own inverse reaches r_hi; 1
        where r_hi lies at or beyond the table's last radius. The radius
        increases with the uniform number, so a uniform number in
        [u_lo, u_hi] draws a radius in [r_lo, r_hi], and u_hi - u_lo is the
        fraction of the sampled mass between r_lo and r_hi.

    Raises
    ------
    InputError
        If r_hi lies at or below the table's first radius, or r_lo lies at or
        above the table's last radius.
    """
    r_edge = float(table.r_sample[component])
    s_c = np.asarray(table.s[component])
    r_first = r_edge / (1.0 + math.exp(-float(s_c[0])))
    r_last = r_edge / (1.0 + math.exp(-float(s_c[-1])))
    r_lo, r_hi = r_range
    if r_hi <= r_first or r_lo >= r_last:
        raise InputError(
            f"The radius range r_range = ({r_lo!r}, {r_hi!r}) holds none of the "
            f"radius table's radii, which run from {r_first:.6e} to {r_last:.6e}."
        )

    def variate(r_k):
        s_k = math.log(r_k) - math.log(r_edge - r_k)
        lam = _radius_bounds_kernel(
            jnp.asarray(s_k),
            float(table.lambda_u[0]),
            float(table.lambda_u[-1]),
            jnp.asarray(s_c),
            jnp.asarray(table.ds_dlambda[component]),
        )
        return float(special.expit(float(lam)))

    u_lo = 0.0 if r_lo <= r_first else variate(r_lo)
    u_hi = 1.0 if r_hi >= r_last else variate(r_hi)
    return u_lo, u_hi


def draw_radius(
    table: RadiusTable,
    component: int,
    numbers: RandomNumbers,
    bounds: tuple[float, float] = (0.0, 1.0),
) -> Array:
    """Return the radius of each particle, in internal units.

    The logit s of the radius relative to the sampling's edge is the table's
    cubic Hermite interpolant at the logit of the particle's uniform number u.
    With `bounds` = (u_lo, u_hi) from `radius_bounds`, the uniform number is
    first mapped into [u_lo, u_hi] as u_lo + (u_hi - u_lo) u; the default
    bounds (0, 1) leave u unchanged.
    """
    u_lo, u_hi = bounds
    return np.asarray(
        _radius_kernel(
            jnp.asarray(u_lo + (u_hi - u_lo) * np.asarray(numbers.radius)),
            float(table.lambda_u[0]),
            float(table.lambda_u[-1]),
            jnp.asarray(table.s[component]),
            jnp.asarray(table.ds_dlambda[component]),
            float(table.r_sample[component]),
        )
    )


def draw_angle(
    table: AngleTable, model: Model, component: int, r: Array, numbers: RandomNumbers
) -> tuple[Array, Array, Array]:
    """Return the velocity angle of each particle and the particle's velocity
    magnitude factor.

    Returns
    -------
    cos_phi : array, shape (n,)
        Cosine of the angle phi between the velocity and the radius vector,
        non-negative; `draw_directions` draws the sign of the radial velocity
        separately.
    sin_phi : array, shape (n,)
        Sine of the angle phi, non-negative.
    velocity_factor : array, shape (n,)
        The factor of the anisotropy model that turns the scaled velocity into
        v / v_esc: 1 for a constant anisotropy, and
        ``1 / sqrt(1 + sigma (r / r_a)^2 sin^2 phi)`` for an Osipkov-Merritt
        model, with sigma = +1 for Type I and sigma = -1 for Type II.

    References
    ----------
    Notes Eq. (T:5.1).

    Notes
    -----
    The variate y = sin^2 is read from the table's lower half for u < 1/2,
    and 1 - y from the table's upper half otherwise, so neither y nor 1 - y
    loses digits. With ``xi = sigma (r / r_a)^2`` (0 for a constant
    anisotropy) and den = 1 + xi (1 - y): sin phi = sqrt(y / den),
    cos phi = sqrt((1 - y) (1 + xi) / den), and the velocity factor is
    sqrt(den / (1 + xi)).
    """
    anisotropy = model.components[component].anisotropy
    r = jnp.asarray(r)
    if math.isinf(anisotropy.r_a):
        xi = jnp.zeros_like(r)
    else:
        sign = 1.0 if anisotropy.kind == "I" else -1.0
        xi = sign * (r / float(anisotropy.r_a)) ** 2
    sin_phi, cos_phi, factor = _angle_kernel(
        jnp.asarray(numbers.angle),
        xi,
        float(table.ln_u[0]),
        float(table.ln_u[-1]),
        jnp.asarray(table.ln_y[component, 0]),
        jnp.asarray(table.dln_y_dln_u[component, 0]),
        jnp.asarray(table.ln_y[component, 1]),
        jnp.asarray(table.dln_y_dln_u[component, 1]),
    )
    return np.asarray(cos_phi), np.asarray(sin_phi), np.asarray(factor)


@jax.jit
def _angle_kernel(u, xi, ln_u_0, ln_u_1, ln_y, m_y, ln_z, m_z):
    y_low = jnp.exp(
        _hermite_uniform(
            jnp.log(jnp.maximum(u, LOG_ARGUMENT_MIN)), ln_u_0, ln_u_1, ln_y, m_y
        )
    )
    z_high = jnp.exp(
        _hermite_uniform(
            jnp.log(jnp.maximum(1.0 - u, LOG_ARGUMENT_MIN)), ln_u_0, ln_u_1, ln_z, m_z
        )
    )
    low = u < 0.5
    y = jnp.where(low, y_low, 1.0 - z_high)
    z = jnp.where(low, 1.0 - y_low, z_high)
    den = 1.0 + xi * z
    return (
        jnp.sqrt(y / den),
        jnp.sqrt(z * (1.0 + xi) / den),
        jnp.sqrt(den / (1.0 + xi)),
    )


def draw_velocity(
    table: VelocityTable,
    model: Model,
    component: int,
    r: Array,
    velocity_factor: Array,
    numbers: RandomNumbers,
    prepared: tuple | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Return the velocity of each particle, in internal units.

    `prepared` is the output of `velocity_draw_tables` for the `table`, `model`
    and `component` passed to `draw_velocity`. A draw made in blocks forms
    `prepared` once; `draw_velocity` forms `prepared` itself where `prepared` is
    None.

    The scaled velocity x is read from the table at the particle's potential Psi,
    and the velocity is v = (2 Psi)^{1/2} x times the anisotropy model's velocity
    factor.

    Returns
    -------
    v : array, shape (n,)
        Velocity of each particle.
    x : array, shape (n,)
        Scaled velocity of each particle.
    d : array, shape (n,)
        Distance x_e - x of each particle's scaled velocity from the support's
        edge x_e. The scaled velocity `x` and the distance `d` are the two halves
        of one logistic function of s, and `validate_particles` checks both.
    psi : array, shape (n,)
        The relative potential at each particle.

    References
    ----------
    Notes Eq. (T:5.2).

    Notes
    -----
    The potential is the profile's own (`TruncatedProfile.potential`), read
    from the profile's `interpolation_tables`: the cubic Hermite interpolant
    of ln Psi in ln r through the interior radii with the radii's exact
    slopes, a straight line with the end radius's slope beyond the interior
    radii, and on the last interval of a profile with an edge row the Kepler
    form ``M_t (1/r - 1/r_t)``. The table row is found by index arithmetic in
    lambda_p and clamped to the rows that have a neighbor on each side.
    Between the two neighboring rows, s is the cubic Hermite interpolant in
    lambda_p with the table's exact across-row slope; along each row, s is the
    cubic Hermite interpolant in lambda_u, and the across-row slope is the
    Catmull-Rom interpolant in lambda_u. The support's edge is formed at the
    particle's own potential where the particle's potential lies outside the
    rows, and otherwise at the potential that the clamped row position
    represents. Where s > 0, past the middle of the row, the scaled velocity is
    formed as x_e - d from the particle's distance d to the edge, so that
    ``x <= x_e`` holds in floating-point arithmetic.
    """
    if prepared is None:
        prepared = velocity_draw_tables(table, model, component)
    potential, (s, ds_dlambda_u, ds_dlambda_p) = prepared
    psi = _psi_kernel(jnp.asarray(r), *potential)
    n_rows = int(table.lambda_p.shape[0])
    x, d = _velocity_kernel(
        jnp.asarray(numbers.velocity_magnitude),
        psi,
        jnp.float64(table.p_0),
        jnp.float64(table.lambda_p[0]),
        jnp.float64(table.dlambda_p),
        jnp.int32(n_rows),
        jnp.float64(table.lambda_u[0]),
        jnp.float64(table.lambda_u[-1]),
        jnp.float64(table.e_edge[component]),
        s,
        ds_dlambda_u,
        ds_dlambda_p,
    )
    v = jnp.sqrt(2.0 * psi) * (x * jnp.asarray(velocity_factor))
    return np.asarray(v), np.asarray(x), np.asarray(d), np.asarray(psi)


@jax.jit
def _velocity_bounds_kernel(x_target, psi, *tables):
    """Return the logit of the velocity number at which the draw's scaled velocity
    at each potential `psi` reaches each target `x_target`, by bisection on
    `_velocity_kernel` itself."""

    def halve(_, ends):
        lo, hi = ends
        mid = 0.5 * (lo + hi)
        below = _velocity_kernel(jax.nn.sigmoid(mid), psi, *tables)[0] < x_target
        return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

    lo, hi = jax.lax.fori_loop(
        0,
        VELOCITY_BOUND_HALVINGS,
        halve,
        (jnp.full_like(x_target, tables[4]), jnp.full_like(x_target, tables[5])),
    )
    return 0.5 * (lo + hi)


def velocity_bounds(
    table: VelocityTable,
    model: Model,
    component: int,
    r: Array,
    v_range: tuple[Any, Any],
    prepared: tuple | None = None,
) -> tuple[Array, Array]:
    """Return the uniform numbers that bound the velocity draw at each radius to a
    user-specified range, for a constant anisotropy.

    Parameters
    ----------
    table : VelocityTable
        The velocity table, as for `draw_velocity`.
    model : Model
        The sampled model, as for `draw_velocity`.
    component : int
        Index of the component, as for `draw_velocity`.
    r : array, shape (n,)
        User-specified 3D radii, in internal units.
    v_range : tuple of float or of array, shape (n,)
        The user-specified velocities (v_lo, v_hi), in internal units, with
        ``0 <= v_lo < v_hi``: one range for every radius, or one range at each
        radius.
    prepared : tuple, optional
        The output of `velocity_draw_tables`, as for `draw_velocity`. Default:
        ``None``, which means `velocity_bounds` forms the tables itself.

    Returns
    -------
    u_lo : array, shape (n,)
        The velocity number at which the draw's own velocity at each radius reaches
        v_lo; 0 where v_lo lies at or below the table's first velocity at the
        radius.
    u_hi : array, shape (n,)
        The velocity number at which the draw's own velocity at each radius reaches
        v_hi; 1 where v_hi lies at or beyond the table's last velocity at the
        radius. The velocity increases with the velocity number, so a velocity number
        in [u_lo, u_hi] draws a velocity between v_lo and v_hi, and u_hi - u_lo is
        the fraction of the velocity distribution at the radius between v_lo and
        v_hi. Where the range holds none of the velocities at a radius, u_lo and
        u_hi are both 0.

    Notes
    -----
    With a constant anisotropy the velocity is v = (2 Psi)^{1/2} x, where the
    scaled velocity x is read from the table at the radius's potential Psi
    (`draw_velocity` with the velocity factor 1), so the user-specified range of
    velocities at a radius is one interval of x.
    """
    if prepared is None:
        prepared = velocity_draw_tables(table, model, component)
    potential, (s, ds_dlambda_u, ds_dlambda_p) = prepared
    psi = _psi_kernel(jnp.asarray(r), *potential)
    tables = (
        jnp.float64(table.p_0),
        jnp.float64(table.lambda_p[0]),
        jnp.float64(table.dlambda_p),
        jnp.int32(table.lambda_p.shape[0]),
        jnp.float64(table.lambda_u[0]),
        jnp.float64(table.lambda_u[-1]),
        jnp.float64(table.e_edge[component]),
        s,
        ds_dlambda_u,
        ds_dlambda_p,
    )
    scale = np.sqrt(2.0 * np.asarray(psi))
    v_lo, v_hi = v_range
    with np.errstate(divide="ignore"):
        x_lo, x_hi = v_lo / scale, v_hi / scale
    ends = [
        np.asarray(_velocity_kernel(jnp.full(scale.shape, u), psi, *tables)[0])
        for u in (special.expit(float(table.lambda_u[0])),
                  special.expit(float(table.lambda_u[-1])))
    ]  # fmt: skip

    def variate(x_k):
        lam = _velocity_bounds_kernel(
            jnp.asarray(np.minimum(x_k, ends[1])), psi, *tables
        )
        return special.expit(np.asarray(lam))

    # Each bound's bisection runs only where some radius needs the bound; the
    # whole distribution, v_lo = 0 and v_hi infinite, needs neither bisection.
    below, beyond = x_lo <= ends[0], x_hi >= ends[1]
    u_lo = np.where(below, 0.0, variate(x_lo) if not np.all(below) else 0.0)
    u_hi = np.where(beyond, 1.0, variate(x_hi) if not np.all(beyond) else 1.0)
    empty = (x_hi <= ends[0]) | (x_lo >= ends[1])
    return np.where(empty, 0.0, u_lo), np.where(empty, 0.0, u_hi)


def velocity_draw_tables(table: VelocityTable, model: Model, component: int) -> tuple:
    """Return the tables that `draw_velocity` reads, padded and placed on the
    device: the profile's potential tables and the velocity table's three arrays
    s, ds / dlambda_u and ds / dlambda_p of component `component`. A draw made
    in blocks forms the tables once."""
    potential = tuple(jnp.asarray(a) for a in _potential_tables(model))
    channels = tuple(
        jnp.asarray(_row_bucket(a[component]))
        for a in (table.s, table.ds_dlambda_u, table.ds_dlambda_p)
    )
    return potential, channels


def _row_bucket(a: Array) -> Array:
    """Return the rows of `a` padded to a power of two by repeating the last
    row. A draw never reads the padding, because a draw's row index stops at
    the last real row."""
    a = np.asarray(a)
    m = 1 << (a.shape[0] - 1).bit_length()
    return np.concatenate([a, np.repeat(a[-1:], m - a.shape[0], axis=0)])


def _potential_tables(model: Model) -> tuple[Array, ...]:
    """Return the profile's tables of the potential in the form that
    `_psi_kernel` takes."""
    tables = model.profile.interpolation_tables()
    return _padded_tables(
        tables.log_r,
        tables.log_p,
        tables.log_p_slope,
        tables.r_last if tables.has_edge_row else math.inf,
        tables.r_t,
        tables.mass_t,
    )


def _padded_tables(
    x: Array, y: Array, m: Array, r_last: float, r_t: float, mass_t: float
) -> tuple[Array, ...]:
    """Return ln r, ln Psi and the slope d ln Psi / d ln r on the interior
    radii, padded to a power of two (ln r infinite on the padding, which is
    never read); the count of real radii; the start `r_last` of the Kepler
    interval; the truncation radius ``r_t``; and the truncated mass M_t.
    `r_last` is infinite where the profile has no Kepler interval."""
    pad = (1 << (x.size - 1).bit_length()) - x.size
    return (
        np.concatenate([x, np.full(pad, np.inf)]),
        np.concatenate([y, np.zeros(pad)]),
        np.concatenate([m, np.zeros(pad)]),
        np.asarray(x.size, dtype=np.int32),
        np.float64(r_last),
        np.float64(r_t),
        np.float64(mass_t),
    )


@jax.jit
def _psi_kernel(r, x_nodes, y_nodes, m_nodes, n, r_last, r_t, mass_t):
    def one(r_i):
        x = jnp.log(r_i)
        i = jnp.clip(jnp.searchsorted(x_nodes, x, side="right") - 1, 0, n - 2)
        h = x_nodes[i + 1] - x_nodes[i]
        t = (x - x_nodes[i]) / h
        t2, t3 = t * t, t * t * t
        inside = (
            (2.0 * t3 - 3.0 * t2 + 1.0) * y_nodes[i]
            + (t3 - 2.0 * t2 + t) * h * m_nodes[i]
            + (-2.0 * t3 + 3.0 * t2) * y_nodes[i + 1]
            + (t3 - t2) * h * m_nodes[i + 1]
        )
        below, above = x < x_nodes[0], x > x_nodes[n - 1]
        log_p = jnp.where(
            below,
            y_nodes[0] + m_nodes[0] * (x - x_nodes[0]),
            jnp.where(
                above, y_nodes[n - 1] + m_nodes[n - 1] * (x - x_nodes[n - 1]), inside
            ),
        )
        kepler = mass_t * jnp.maximum(1.0 / r_i - 1.0 / r_t, 0.0)
        return jnp.where(r_i > r_last, kepler, jnp.exp(log_p))

    return jax.vmap(one)(r)


@jax.jit
def _velocity_kernel(
    u,
    psi,
    p_0,
    lambda_p_0,
    dlambda_p,
    n_rows,
    lambda_u_0,
    lambda_u_1,
    e_edge,
    s,
    ds_dlambda_u,
    ds_dlambda_p,
):
    n_nodes = s.shape[1]
    h_u = (lambda_u_1 - lambda_u_0) / (n_nodes - 1)

    def row(lam_u, i_rows):
        """Return s and the across-row slope at `lam_u` on the rows
        `i_rows`."""
        sx = (jnp.clip(lam_u, lambda_u_0, lambda_u_1) - lambda_u_0) / h_u
        j = jnp.clip(jnp.floor(sx).astype(jnp.int32), 0, n_nodes - 2)
        t = sx - j
        t2, t3 = t * t, t * t * t
        s_j = (
            (2.0 * t3 - 3.0 * t2 + 1.0) * s[i_rows, j]
            + (t3 - 2.0 * t2 + t) * h_u * ds_dlambda_u[i_rows, j]
            + (-2.0 * t3 + 3.0 * t2) * s[i_rows, j + 1]
            + (t3 - t2) * h_u * ds_dlambda_u[i_rows, j + 1]
        )
        r_a = ds_dlambda_p[i_rows, jnp.clip(j - 1, 0, n_nodes - 1)]
        r_b = ds_dlambda_p[i_rows, j]
        r_c = ds_dlambda_p[i_rows, j + 1]
        r_d = ds_dlambda_p[i_rows, jnp.clip(j + 2, 0, n_nodes - 1)]
        r_j = 0.5 * (
            (-t3 + 2.0 * t2 - t) * r_a
            + (3.0 * t3 - 5.0 * t2 + 2.0) * r_b
            + (-3.0 * t3 + 4.0 * t2 + t) * r_c
            + (t3 - t2) * r_d
        )
        return s_j, r_j

    def one(u_i, psi_i):
        lam = jnp.log(psi_i / jnp.maximum(p_0 - psi_i, LOG_ARGUMENT_MIN))
        sx_raw = (lam - lambda_p_0) / dlambda_p
        sx = jnp.clip(sx_raw, 1.0, n_rows - 2.0)
        i = jnp.clip(jnp.floor(sx).astype(jnp.int32), 1, n_rows - 3)
        w = sx - i
        w2, w3 = w * w, w * w * w
        s_rows, r_rows = row(_logit(u_i), i + jnp.array([0, 1], dtype=jnp.int32))
        s_i = (
            (2.0 * w3 - 3.0 * w2 + 1.0) * s_rows[0]
            + (w3 - 2.0 * w2 + w) * dlambda_p * r_rows[0]
            + (-2.0 * w3 + 3.0 * w2) * s_rows[1]
            + (w3 - w2) * dlambda_p * r_rows[1]
        )
        psi_c = p_0 / (1.0 + jnp.exp(-(lambda_p_0 + sx * dlambda_p)))
        outside = (sx_raw < 1.0) | (sx_raw > n_rows - 2.0)
        x_e = jnp.where(
            outside,
            jnp.sqrt(jnp.maximum(1.0 - e_edge / psi_i, 0.0)),
            jnp.sqrt(jnp.maximum(1.0 - e_edge / psi_c, 0.0)),
        )
        d_i = x_e / (1.0 + jnp.exp(s_i))
        return jnp.where(s_i > 0.0, x_e - d_i, x_e / (1.0 + jnp.exp(-s_i))), d_i

    return jax.vmap(one)(u, psi)


def draw_directions(
    r: Array, v: Array, cos_phi: Array, sin_phi: Array, numbers: RandomNumbers
) -> tuple[Array, Array]:
    """Return the Cartesian positions and velocities of the particles.

    The direction of each position is uniform on the sphere. The tangential
    velocity points along an isotropic vector projected onto the plane normal
    to the position. The radial velocity points outward where the particle's
    radial-sign number is below 1/2.
    """
    r, v = jnp.asarray(r), jnp.asarray(v)
    unit = jax.vmap(_unit_vector)(jnp.asarray(numbers.position))
    tangent = jax.vmap(_tangent_vector)(unit, jnp.asarray(numbers.velocity))
    v_radial = v * jnp.asarray(cos_phi)
    v_tangential = v * jnp.asarray(sin_phi)
    sign = jnp.where(jnp.asarray(numbers.radial_sign) < 0.5, 1.0, -1.0)
    position = unit * r[:, None]
    velocity = unit * (sign * v_radial)[:, None] + tangent * v_tangential[:, None]
    return np.asarray(position), np.asarray(velocity)


@jax.jit
def _unit_vector(key):
    """Return a direction uniform on the sphere: three normal numbers drawn
    from the three subkeys of `key`, divided by the three numbers' norm."""
    k_1, k_2, k_3 = jax.random.split(key, 3)
    g_1 = jax.random.normal(k_1, dtype=jnp.float64)
    g_2 = jax.random.normal(k_2, dtype=jnp.float64)
    g_3 = jax.random.normal(k_3, dtype=jnp.float64)
    norm = jnp.sqrt(g_1 * g_1 + g_2 * g_2 + g_3 * g_3) + LOG_ARGUMENT_MIN
    return jnp.array([g_1 / norm, g_2 / norm, g_3 / norm])


@jax.jit
def _tangent_vector(unit, key):
    """Return a direction uniform on the sphere, projected onto the plane
    normal to `unit` and normalized.

    The dot product and the norm are sums written in a fixed order, because
    under `vmap` XLA may order the reductions `jnp.dot` and `jnp.linalg.norm`
    according to the batch size, and a particle's velocity must not depend on
    the size of the block in which the particle is drawn.
    """
    direction = _unit_vector(key)
    dot = (direction[0] * unit[0] + direction[1] * unit[1]) + direction[2] * unit[2]
    along = direction - dot * unit
    norm = jnp.sqrt((along[0] * along[0] + along[1] * along[1]) + along[2] * along[2])
    return along / (norm + LOG_ARGUMENT_MIN)


VALIDATION_KEYS = ("phase_space_bound", "velocity_construction")


def validate_particles(
    model: Model,
    component: int,
    position: Array,
    velocity: Array,
    x: Array,
    d: Array,
    psi: Array,
) -> ValidationReport:
    """Return the validation report of a draw, with the draw's two checks.

    Every particle must lie inside the phase-space support of the
    component's DF. The phase-space bound is checked on the draw's own
    variables, the scaled velocity `x` and the distance `d` from the edge
    (``x <= 1`` and ``d >= 0``), which satisfy the bound exactly; a velocity
    rebuilt from the velocity carries the rounding of the conversion. The
    velocity is then checked against the bound from which the velocity was
    built, ``v^2 + sigma xi v_t^2 <= 2 Psi``, where ``xi = (r / r_a)^2`` (0
    for a constant anisotropy), sigma = +1 for Type I and sigma = -1 for
    Type II, and v_t is the tangential velocity, within `BOUND_SAFETY` times the
    bound's rounding floor eps (1 + xi).

    Parameters
    ----------
    model : Model
        The model from which the particles are drawn.
    component : int
        Index of the component.
    position : array, shape (n, 3)
        Positions of the particles, in internal units.
    velocity : array, shape (n, 3)
        Velocities of the particles, in internal units.
    x : array, shape (n,)
        The draw's scaled velocity of each particle.
    d : array, shape (n,)
        The draw's distance of each particle's scaled velocity from the support's
        edge.
    psi : array, shape (n,)
        The relative potential at each particle, as the draw read the
        potential.

    Returns
    -------
    report : ValidationReport
        The two checks listed in ``VALIDATION_KEYS``, both with error 0: a draw
        in which any particle fails either check raises AccuracyError instead
        of returning.

    Raises
    ------
    AccuracyError
        If a particle has x > 1, d < 0, or a non-finite x or d.
    AccuracyError
        If a particle's velocity exceeds the bound
        ``v^2 + sigma xi v_t^2 <= 2 Psi`` by more than `BOUND_SAFETY` times
        the rounding floor eps (1 + xi).
    """
    x, d = np.asarray(x), np.asarray(d)
    beyond = (x > 1.0) | (d < 0.0) | ~np.isfinite(x) | ~np.isfinite(d)
    if np.any(beyond):
        i = int(np.argmax(np.where(np.isfinite(x), x, np.inf)))
        raise AccuracyError(
            f"{int(np.sum(beyond))} drawn particles lie beyond the phase-space "
            "bound: the scaled velocity x is above 1, the distance d to the velocity "
            "support's edge is below 0, or x or d is not finite. The worst "
            f"particle has x = {float(x[i])!r} and d = {float(d[i])!r}."
        )
    anisotropy = model.components[component].anisotropy
    position, velocity = np.asarray(position), np.asarray(velocity)
    r = np.linalg.norm(position, axis=1)
    if math.isinf(anisotropy.r_a):
        sigma, xi = 0.0, np.zeros_like(r)
    else:
        sigma = 1.0 if anisotropy.kind == "I" else -1.0
        xi = (r / float(anisotropy.r_a)) ** 2
    v_esc2 = 2.0 * np.asarray(psi)
    unit = position / np.maximum(r, LOG_ARGUMENT_MIN)[:, None]
    v_t2 = np.sum(np.cross(unit, velocity) ** 2, axis=1)
    v_aux2 = np.sum(velocity**2, axis=1) + sigma * xi * v_t2
    excess = v_aux2 / np.maximum(v_esc2, LOG_ARGUMENT_MIN) - 1.0
    floor = BOUND_SAFETY * np.finfo(np.float64).eps * (1.0 + xi)
    if np.any(excess > floor):
        i = int(np.argmax(excess - floor))
        raise AccuracyError(
            f"The velocities of {int(np.sum(excess > floor))} drawn particles do "
            "not reproduce the draw's phase-space bound. The worst particle, at "
            f"r = {float(r[i])!r}, exceeds the bound by {float(excess[i]):+.3e}, "
            f"against the bound's rounding floor {float(floor[i]):.1e}."
        )
    return ValidationReport(
        keys=VALIDATION_KEYS, checks={key: Check(error=0.0) for key in VALIDATION_KEYS}
    )
