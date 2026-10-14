"""Stages of the particle sampler and their results.

The sampler is prepared once per model::

    radius_table, speed_table, angle_table

and each draw composes::

    random_numbers -> draw_radius -> draw_angle -> draw_speed
        -> draw_directions -> validate_particles

Every variable is drawn by inversion; nothing is rejected. A particle's random
numbers depend only on the seed, the component and the particle's index.
"""

__all__ = [
    "AngleTable",
    "RandomNumbers",
    "RadiusTable",
    "SpeedTable",
    "angle_table",
    "component_numbers",
    "draw_angle",
    "draw_directions",
    "draw_radius",
    "draw_speed",
    "radius_table",
    "random_numbers",
    "random_numbers_block",
    "speed_draw_tables",
    "speed_table",
    "validate_particles",
]

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy import special

from ..df import Model
from ..exceptions import InputError, ResultError
from ..numerics import hermite_inverse
from ..settings import Settings
from ..typing import Array
from ..validation import Check, ValidationReport
from .speed import build_speed_table

# Relative slack within which a radius counts as the profile's last one.
PROFILE_END_SLACK = 1e-12
# The channel of a particle's key each random variable is drawn from.
RNG_CHANNELS = dict(
    radius=0, speed=1, angle=2, radial_sign=3, position=4, velocity=5, component=6
)
# Floor of the argument of a logarithm and of the norm of a direction in the
# draws: a uniform double is never 0 or 1, so it only keeps a discarded branch
# finite.
LOG_ARGUMENT_MIN = 1e-300
# The velocity rebuilt from the particle's arrays must reproduce the draw's
# bound within this many times its own rounding floor, eps (1 + xi); the
# largest excess measured is 0.10 of it, on the 1e6 particles of each of the
# 33 members of the assembly subset.
BOUND_SAFETY = 1.0


@dataclass(frozen=True)
class RadiusTable:
    """The inverse of the enclosed-mass fraction of each component.

    Parameters
    ----------
    lambda_u : array, shape (n,)
        Nodes, uniform in the logit of the mass fraction.
    s : array, shape (n_components, n)
        Logit of the radius relative to the outer radius of the sampling.
    ds_dlambda : array, shape (n_components, n)
        Exact slope of `s` at the nodes.
    r_sample : tuple of float
        Outer radius of the sampling of each component, in internal units:
        the truncation radius of a spatially finite model, otherwise the
        radius given by the user.
    mass_sample : tuple of float
        Mass of each component inside `r_sample`.
    """

    lambda_u: Array
    s: Array
    ds_dlambda: Array
    r_sample: tuple[float, ...]
    mass_sample: tuple[float, ...]


@dataclass(frozen=True)
class SpeedTable:
    """The inverse distribution of the scaled speed, by potential.

    Parameters
    ----------
    lambda_p : array, shape (n_rows,)
        Rows, uniform in the logit of the potential relative to its central
        value, ln(Psi / (p_0 - Psi)). The row count is the table's own: from
        ``sampling.n_rows`` the rows are refined (n to 2 n - 3, every row
        kept) until the table's own error across rows is under the goal for
        every component, or until a further refinement would pass
        ``sampling.n_rows_max``; `validation` says which.
    dlambda_p : float
        Spacing of the rows in `lambda_p`, as the draw reads it: the rows are
        formed from it, and a difference of two rows does not return it to
        the bit.
    lambda_u : array, shape (n,)
        Nodes of each row, uniform in the logit of the cumulative
        probability.
    s : array, shape (n_components, n_rows, n)
        Logit of the scaled speed relative to the edge of its support.
    ds_dlambda_u, ds_dlambda_p : array, shape (n_components, n_rows, n)
        Exact slopes of `s` along a row and across rows.
    x_edge : array, shape (n_components, n_rows)
        Edge of the support of the scaled speed on each row.
    p_0 : float
        Central potential, the end of the logit `lambda_p`.
    e_edge : tuple of float
        Energy at which each component's DF ends: zero where the DF continues
        below its table, else the table's first energy. A draw forms the edge
        of the support at the particle's own potential, sqrt(1 - e_edge / Psi).
    validation : ValidationReport
        The table's checks (`SPEED_TABLE_KEYS`), each with the Psi of its
        worst row: ``row_error``, the table's own error across rows (every
        interior row predicted from its two neighbors by the interpolant of
        the draw, at twice the written spacing; the larger of the misses in
        the cumulative probability and in the relative speed), and
        ``polish_residual``, the largest miss of a node against the true
        cumulative probability, in its logit. A table beyond the hard limit
        is refused.
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


@dataclass(frozen=True)
class AngleTable:
    """The inverse distribution of the velocity angle of each component.

    The angle of a component follows from a variate y = sin^2 of the angle,
    Beta(1 - beta_c, 1/2) distributed and the same at every radius; the table
    holds its inverse from each end, both ends being power laws. An isotropic
    component takes the same path, with Beta(1, 1/2).

    Parameters
    ----------
    ln_u : array, shape (n,)
        Nodes, uniform in the logarithm of the cumulative probability from
        each end, from `tail_mass_min` to 1/2.
    ln_y : array, shape (n_components, 2, n)
        Logarithm of the variate from each end: ``ln_y[:, 0]`` is ln y against
        ln u, ``ln_y[:, 1]`` is ln(1 - y) against ln(1 - u).
    dln_y_dln_u : array, shape (n_components, 2, n)
        Exact slopes of `ln_y` at the nodes, from the Beta density.
    """

    ln_u: Array
    ln_y: Array
    dln_y_dln_u: Array


@dataclass(frozen=True)
class RandomNumbers:
    """Random numbers of a set of particles.

    Parameters
    ----------
    radius, speed, angle : array, shape (n,)
        One uniform random number per particle for each drawn variable.
    radial_sign : array, shape (n,)
        One uniform random number per particle for the sign of the radial velocity.
    position, velocity : array of uint32, shape (n, 2)
        The key of each of the two directions. Its three normal numbers are
        drawn from it inside `draw_directions`, in the same compiled function
        as the direction: the compiler rounds the direction otherwise, by one
        unit in the last place on about 2 percent of the components.
    """

    radius: Array
    speed: Array
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
        Numerical settings.
    r_sample : float, optional
        Outer radius of the sampling, in internal units. Required for a model
        that is not spatially finite.

    Returns
    -------
    table : RadiusTable

    Raises
    ------
    InputError
        If `r_sample` is missing for a model that is not spatially finite,
        given for one that is, or outside the profile.
    ResultError
        If the enclosed mass is not increasing in the table's variables.

    Notes
    -----
    With the mass fraction u = M(r) / M(r_e) and the edge r_e = `r_sample`,
    the variables s = ln(r / (r_e - r)) and lambda = ln(u / (1 - u)) are
    power laws at both ends, the center and the edge, whether the edge is
    smooth (the truncation radius, where the density vanishes) or sharp (a
    radius the user gives, where it does not). The mass fraction is exact at
    the profile's rows and its slope is exact everywhere, du/dr = 4 pi r^2
    rho / M(r_e); the inverse s(lambda) is tabulated at nodes uniform in
    lambda with its exact slope. Rows where 1 - u is below
    ``sampling.radius_edge_refined`` are left out (1 - u is a difference
    there); at a sharp edge the last interval is refined at halved distances
    from the edge by the cubic Hermite in (r_e - r, 1 - u), whose two ends
    are exact.
    """
    if len(model.components) != 1:
        raise NotImplementedError(
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
        # the last interval's cubic Hermite in (r_e - r, 1 - u), 1 - u = 0 at r_e
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
    # the exact slope of the inverse, du/dr between the nodes by interpolation
    # of its logarithm in s; beyond the nodes the continuation's own
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
            "The inverse of the enclosed mass is not finite and increasing."
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
    """The profile's rows out to the sampling edge, cut as the working code cuts them.

    Returns (r, psi, rho, mass, r_edge, smooth): the mass column is 4 pi times
    the profile's, as the working code holds it; a model that is not spatially
    finite ends on `r_sample`, its point by linear interpolation in the
    logarithms, and its edge is sharp; a spatially finite one ends on its
    truncation radius, a smooth edge.
    """
    profile = model.profile
    r, psi, rho = profile.r, profile.p_trunc, profile.rho_trunc[0]
    mass = 4.0 * np.pi * profile.mass_trunc
    if model.spatially_finite:
        if r_sample is not None:
            raise InputError(
                "A spatially finite model is sampled to its edge: no r_sample is given."
            )
        return r, psi, rho, mass, float(profile.r_t), True
    if r_sample is None:
        raise InputError(
            "A model that is not spatially finite needs r_sample, the outer "
            "radius of the sampling."
        )
    if not (r[0] < r_sample <= r[-1] * (1.0 + PROFILE_END_SLACK)):
        raise InputError(
            f"r_sample = {r_sample!r} is outside the profile ({r[0]!r}, {r[-1]!r}]."
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


def speed_table(model: Model, settings: Settings) -> SpeedTable:
    """Return the inverse distribution of the scaled speed, by potential.

    The density of the scaled speed depends on the radius only through the
    potential, so one table serves every radius and angle.

    Parameters
    ----------
    model : Model
        The model to sample.
    settings : Settings
        Numerical settings.

    Returns
    -------
    table : SpeedTable

    Raises
    ------
    ResultError
        If a row cannot be built, the model's mass lies below the lowest row,
        or a check of the table lies beyond the hard limit.

    References
    ----------
    Notes Eq. (I.22) [x], Notes Eq. (I.23) [factorized],
    Notes Section II.6.1 [speed-table].

    Notes
    -----
    The construction is `pianistpy.particles.speed.build_speed_table`.
    """
    if len(model.components) != 1:
        raise NotImplementedError(
            "The speed table of a model of several components is not built yet."
        )
    return SpeedTable(**build_speed_table(model, settings))


def angle_table(model: Model, settings: Settings) -> AngleTable:
    """Return the inverse distribution of the velocity angle.

    For each component, y = sin^2 of the angle between the velocity and the
    radius vector is Beta(a, 1/2) with a = 1 - beta_c (Notes Eq. (I.21) [f2]);
    1 - y is Beta(1/2, a). Each half of the inverse is tabulated against the
    logarithm of its own end's probability, where both ends are power laws, and
    the slope of the inverse is u / (y pdf(y)) in closed form.

    Parameters
    ----------
    model : Model
        The model to sample.
    settings : Settings
        Numerical settings: ``sampling.n_angle_nodes``, ``sampling.tail_mass_min``.

    Returns
    -------
    table : AngleTable

    Raises
    ------
    ResultError
        If an inverse is not finite and increasing.
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
                "increasing on the table's nodes."
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
    and `i`. They are the same on every machine and backend, for any batch
    size and worker count, and the first `n` particles of a larger draw are
    the same particles. What a set of particles was drawn with (the model,
    the settings, the version) is the record of its result, not a part of
    its key.

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

    Notes
    -----
    Particle `i` draws from the key k_i = fold_in(base, i), with base =
    fold_in(PRNGKey(seed), component), and each variable from its own
    channel of k_i: ``RNG_CHANNELS``; the two directions are kept as their
    keys. The keys are integer arithmetic, so the keys and the uniform numbers
    are the same bits on every backend; what is formed from them rounds.
    """
    return random_numbers_block(seed, component, 0, n)


def random_numbers_block(
    seed: int, component: int, start: int, n: int
) -> RandomNumbers:
    """Return the random numbers of particles `start` .. `start + n - 1`.

    The same numbers as those particles carry in `random_numbers` of any
    larger draw: each particle's key is formed from its own index.
    """
    return _random_numbers_from(
        jax.random.fold_in(jax.random.PRNGKey(seed), component), n, start
    )


def component_numbers(seed: int, start: int, n: int) -> Array:
    """Return the uniform number of particles `start` .. `start + n - 1` that
    chooses the component of a superposed model: the channel ``component`` of
    each particle's own key, beside the channels of its draw, so that the
    particle's other numbers are those of the draw of one component."""
    base = jax.random.fold_in(jax.random.PRNGKey(seed), 0)
    keys = jax.vmap(lambda i: jax.random.fold_in(base, i))(
        jnp.arange(start, start + n, dtype=jnp.int32)
    )
    return np.asarray(_channel_uniform(keys, RNG_CHANNELS["component"]))


def _random_numbers_from(base: Array, n: int, start: int = 0) -> RandomNumbers:
    """The random numbers of particles start .. start + n - 1 drawn from a base key."""
    keys = jax.vmap(lambda i: jax.random.fold_in(base, i))(
        jnp.arange(start, start + n, dtype=jnp.int32)
    )

    def uniform(channel):
        return np.asarray(_channel_uniform(keys, channel))

    def channel_keys(channel):
        return np.asarray(jax.vmap(lambda key: jax.random.fold_in(key, channel))(keys))

    return RandomNumbers(
        radius=uniform(RNG_CHANNELS["radius"]),
        speed=uniform(RNG_CHANNELS["speed"]),
        angle=uniform(RNG_CHANNELS["angle"]),
        radial_sign=uniform(RNG_CHANNELS["radial_sign"]),
        position=channel_keys(RNG_CHANNELS["position"]),
        velocity=channel_keys(RNG_CHANNELS["velocity"]),
    )


def _channel_uniform(keys: Array, channel: int) -> Array:
    """One uniform number per key from its channel."""
    return jax.vmap(
        lambda key: jax.random.uniform(
            jax.random.fold_in(key, channel), dtype=jnp.float64
        )
    )(keys)


def _hermite_uniform(x, x_0, x_1, f, m):
    """The cubic Hermite through (f, m) on nodes uniform over [x_0, x_1].

    The cell is found by index arithmetic, with no search; beyond the ends the
    end node holds.
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
    """ln(u / (1 - u)) of a uniform number; 1 - u is exact for u >= 1/2."""
    return jnp.log(jnp.maximum(u, LOG_ARGUMENT_MIN)) - jnp.log(
        jnp.maximum(1.0 - u, LOG_ARGUMENT_MIN)
    )


@jax.jit
def _radius_kernel(u, lambda_0, lambda_1, s, ds_dlambda, r_edge):
    def one(u_i):
        s_i = _hermite_uniform(_logit(u_i), lambda_0, lambda_1, s, ds_dlambda)
        return r_edge / (1.0 + jnp.exp(-s_i))

    return jax.vmap(one)(u)


def draw_radius(table: RadiusTable, component: int, numbers: RandomNumbers) -> Array:
    """Return the radius of each particle, in internal units.

    The logit of the radius relative to the edge of the sampling is the cubic
    Hermite of the table in the logit of the particle's uniform number.
    """
    return np.asarray(
        _radius_kernel(
            jnp.asarray(numbers.radius),
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
    """Return the velocity angle of each particle and its speed factor.

    Returns
    -------
    cos_phi, sin_phi : array, shape (n,)
        Cosine and sine of the angle between the velocity and the radius
        vector. Both are non-negative; the sign of the radial velocity is
        drawn apart.
    speed_factor : array, shape (n,)
        The factor of the anisotropy model that turns the scaled speed into
        v / v_esc: 1 for a constant anisotropy, and
        1 / sqrt(1 + sigma (r / r_a)^2 sin^2) for an Osipkov-Merritt one,
        sigma = +1 for type I and -1 for type II.

    References
    ----------
    Notes Eq. (I.21) [f2].

    Notes
    -----
    y = sin^2 is read from the table's lower half for u < 1/2 and 1 - y from
    its upper half otherwise, so that neither loses digits. With xi = sigma
    (r / r_a)^2 (0 for a constant anisotropy) and den = 1 + xi (1 - y):
    sin = sqrt(y / den), cos = sqrt((1 - y) (1 + xi) / den) and the factor
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


def draw_speed(
    table: SpeedTable,
    model: Model,
    component: int,
    r: Array,
    speed_factor: Array,
    numbers: RandomNumbers,
    prepared: tuple | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Return the speed of each particle, in internal units.

    `prepared` is `speed_draw_tables` of the same table, model and component,
    formed once where the draw is made in blocks; formed here otherwise.

    The scaled speed is read from the table at the particle's potential and
    converted with the escape speed and the factor of the anisotropy model.

    Returns
    -------
    v : array, shape (n,)
        Speed.
    x, d : array, shape (n,)
        The scaled speed and its distance from the edge of the support, the
        two halves of one logistic, which `validate_particles` checks.
    psi : array, shape (n,)
        The potential at each particle.

    References
    ----------
    Notes Eq. (I.22) [x].

    Notes
    -----
    The potential is the profile's own, `TruncatedProfile.potential`, read
    from its `interpolation_tables`: the cubic Hermite of ln Psi in ln r
    through the interior rows with their exact slopes, the straight line of
    the end row's slope beyond them, and on the last interval of a profile
    with its edge row the Kepler form M_t (1/r - 1/r_t). The row of the table
    is found by index arithmetic in lambda_p, clamped to the rows that have a
    neighbor on each side; across the two neighboring rows s is the cubic
    Hermite with the table's exact
    across-row slope, read along the row by Catmull-Rom. The edge of the
    support is formed at the particle's own potential when it lies outside the
    rows, else at the potential the clamped row position stands for. Past the
    middle of the row (s > 0) the speed is formed from its distance to the
    edge, so that x <= x_e holds in the arithmetic.
    """
    if prepared is None:
        prepared = speed_draw_tables(table, model, component)
    potential, (s, ds_dlambda_u, ds_dlambda_p) = prepared
    psi = _psi_kernel(jnp.asarray(r), *potential)
    n_rows = int(table.lambda_p.shape[0])
    x, d = _speed_kernel(
        jnp.asarray(numbers.speed),
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
    v = jnp.sqrt(2.0 * psi) * (x * jnp.asarray(speed_factor))
    return np.asarray(v), np.asarray(x), np.asarray(d), np.asarray(psi)


def speed_draw_tables(table: SpeedTable, model: Model, component: int) -> tuple:
    """The tables `draw_speed` reads, padded and on the device: the
    potential's and the speed table's three channels. Formed once for a draw
    made in blocks."""
    potential = tuple(jnp.asarray(a) for a in _potential_tables(model))
    channels = tuple(
        jnp.asarray(_row_bucket(a[component]))
        for a in (table.s, table.ds_dlambda_u, table.ds_dlambda_p)
    )
    return potential, channels


def _row_bucket(a: Array) -> Array:
    """The rows of an array padded to a power of two by repeating the last:
    the padding is never read, since a draw's row index stops at the last
    real row."""
    a = np.asarray(a)
    m = 1 << (a.shape[0] - 1).bit_length()
    return np.concatenate([a, np.repeat(a[-1:], m - a.shape[0], axis=0)])


def _potential_tables(model: Model) -> tuple[Array, ...]:
    """The profile's tables of the potential, as `_psi_kernel` takes them."""
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
    """ln r, ln Psi and its slope on the interior rows, padded to a power of
    two (ln r infinite there, never read), their count, and the Kepler
    interval's start, r_t and M_t (r_last infinite where there is none)."""
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
def _speed_kernel(
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
        """s and its across-row slope on the rows i_rows at lambda_u."""
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

    The position is uniform on the sphere. The tangential velocity points
    along an isotropic vector projected on the plane normal to the position.
    The radial velocity is outward for a radial-sign number below 1/2.
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
    """A direction uniform on the sphere: three normal numbers from the key's
    three subkeys, over their norm."""
    k_1, k_2, k_3 = jax.random.split(key, 3)
    g_1 = jax.random.normal(k_1, dtype=jnp.float64)
    g_2 = jax.random.normal(k_2, dtype=jnp.float64)
    g_3 = jax.random.normal(k_3, dtype=jnp.float64)
    norm = jnp.sqrt(g_1 * g_1 + g_2 * g_2 + g_3 * g_3) + LOG_ARGUMENT_MIN
    return jnp.array([g_1 / norm, g_2 / norm, g_3 / norm])


@jax.jit
def _tangent_vector(unit, key):
    """A direction uniform on the sphere, projected normal to `unit`.

    The dot product and the norm are sums written in one order: batched by
    `vmap`, `jnp.dot` and `jnp.linalg.norm` are reductions whose order XLA
    may choose by the batch size (JAX 0.10.2 does), and a particle's velocity
    must not depend on the size of the block it is drawn in.
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
    """Return the report of a draw: its essential checks.

    Every particle must lie inside the phase-space support of its
    component's DF. The bound is checked on the draw's own variables, the
    scaled speed `x` and its distance `d` from the edge (x <= 1 and d >= 0),
    which hold it exactly; a speed rebuilt from the velocity carries the
    rounding of the conversion. The velocity is then checked against the
    bound it was built from: v^2 + sigma xi v_t^2 <= 2 Psi, with xi = (r /
    r_a)^2 (0 for a constant anisotropy), within `BOUND_SAFETY` times its own
    rounding floor eps (1 + xi).

    Parameters
    ----------
    model : Model
        The model drawn from.
    component : int
        Index of the component.
    position, velocity : array, shape (n, 3)
        The particles, in internal units.
    x, d : array, shape (n,)
        The draw's scaled speed and its distance from the edge.
    psi : array, shape (n,)
        The potential at each particle, as the draw read it.

    Returns
    -------
    report : ValidationReport
        Both checks at zero: a particle that fails either is refused.

    Raises
    ------
    ResultError
        If a particle lies beyond the bound, or its velocity does not
        reproduce it.
    """
    x, d = np.asarray(x), np.asarray(d)
    beyond = (x > 1.0) | (d < 0.0) | ~np.isfinite(x) | ~np.isfinite(d)
    if np.any(beyond):
        i = int(np.argmax(np.where(np.isfinite(x), x, np.inf)))
        raise ResultError(
            f"{int(np.sum(beyond))} particles lie beyond the phase-space bound "
            f"(x > 1, d < 0 or not finite); the worst has x = {float(x[i])!r}, "
            f"d = {float(d[i])!r}."
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
        raise ResultError(
            f"The velocity of {int(np.sum(excess > floor))} particles does not "
            f"reproduce the draw's bound; the worst at r = {float(r[i])!r}: "
            f"{float(excess[i]):+.3e} against its floor {float(floor[i]):.1e}."
        )
    return ValidationReport(
        keys=VALIDATION_KEYS, checks={key: Check(error=0.0) for key in VALIDATION_KEYS}
    )
