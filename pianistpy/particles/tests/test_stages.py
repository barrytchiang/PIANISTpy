"""Tests of the stages of the particle sampler against closed forms."""

import dataclasses

import numpy as np
import pytest
from scipy import integrate, special

from pianistpy import Settings
from pianistpy.anisotropy import Anisotropy
from pianistpy.df.analytic import analytic_model, hernquist_df
from pianistpy.df.edge import density_edge_exponent
from pianistpy.df.inputs import Component
from pianistpy.df.lookup import log10_df
from pianistpy.df.stages import DfTable
from pianistpy.exceptions import AccuracyError, InputError, ResultError
from pianistpy.particles import Sampler, speed, stages
from pianistpy.particles.stages import (
    SpeedTable,
    angle_table,
    draw_angle,
    draw_directions,
    draw_radius,
    draw_speed,
    radius_table,
    random_numbers,
    speed_table,
    validate_particles,
)
from pianistpy.validation import GOAL, HARD_LIMIT, ValidationReport


def _model_with(beta_cs):
    r"""The analytic Hernquist model, its component repeated with the given
    :math:`\beta_{\rm c}`."""
    model = analytic_model("hernquist")
    profile = model.components[0].profile
    components = tuple(
        Component(profile=profile, anisotropy=Anisotropy(beta_c=b), name=f"c{i}")
        for i, b in enumerate(beta_cs)
    )
    return dataclasses.replace(model, components=components)


@pytest.mark.parametrize("beta_c", [0.0, 0.3, -0.5, -1.5])
def test_angle_table_inverts_the_beta_distribution(beta_c):
    table = angle_table(_model_with([beta_c]), Settings())
    a = 1.0 - beta_c
    u = np.exp(table.ln_u)
    y = np.exp(table.ln_y[0, 0])
    one_minus_y = np.exp(table.ln_y[0, 1])
    np.testing.assert_allclose(special.betainc(a, 0.5, y), u, rtol=1e-12)
    np.testing.assert_allclose(special.betainc(0.5, a, one_minus_y), u, rtol=1e-12)


@pytest.mark.parametrize("beta_c", [0.0, 0.3, -1.5])
def test_angle_table_slopes_are_the_derivative_of_the_inverse(beta_c):
    table = angle_table(_model_with([beta_c]), Settings())
    a = 1.0 - beta_c
    step = 1e-6
    for half, (p, q) in enumerate(((a, 0.5), (0.5, a))):
        # central difference of the exact inverse in ln u, step 1e-6: truncation 1e-12
        up = np.log(special.betaincinv(p, q, np.exp(table.ln_u + step)))
        down = np.log(special.betaincinv(p, q, np.exp(table.ln_u - step)))
        np.testing.assert_allclose(
            (up - down) / (2.0 * step), table.dln_y_dln_u[0, half], rtol=1e-8
        )


def test_angle_table_has_one_row_per_component():
    table = angle_table(_model_with([0.0, 0.25, -0.25]), Settings())
    n = Settings().sampling["n_angle_nodes"]
    assert table.ln_u.shape == (n,)
    assert table.ln_y.shape == (3, 2, n) and table.dln_y_dln_u.shape == (3, 2, n)
    assert table.ln_u[0] == pytest.approx(np.log(Settings().sampling["tail_mass_min"]))
    assert table.ln_u[-1] == pytest.approx(np.log(0.5))


def test_radius_table_inverts_the_hernquist_mass():
    model = analytic_model("hernquist")
    r_edge = float(model.profile.r[-1])
    table = radius_table(model, Settings(), r_edge)
    lambda_u = table.lambda_u
    r = r_edge / (1.0 + np.exp(-table.s[0]))
    d = r_edge / (1.0 + np.exp(table.s[0]))
    # M(r) = (r / (1 + r))^2; the mass beyond r, (a - b)(a + b), from d itself
    a, b = r_edge / (1.0 + r_edge), r / (1.0 + r)
    lambda_exact = np.log(b * b) - np.log(d / ((1.0 + r_edge) * (1.0 + r)) * (a + b))
    reach = np.abs(lambda_u) <= 53.0 * np.log(2.0)
    error = np.abs(lambda_exact - lambda_u)[reach]
    # in u, 1.5e-12 measured; in lambda 7.1e-6 at u = 1e-16, below the first
    # row of the profile (r = 1e-6), where the inverse continues as a power law
    u_weight = 1.0 / ((1.0 + np.exp(-lambda_u)) * (1.0 + np.exp(lambda_u)))[reach]
    assert np.max(error * u_weight) < 1e-11
    assert np.max(error) < 1e-5


def test_radius_table_needs_r_sample_for_an_infinite_model():
    with pytest.raises(InputError):
        radius_table(analytic_model("hernquist"), Settings())


def _numbers(n, seed=1):
    """Random numbers for n particles, keyed as the package keys them."""
    return random_numbers(seed, 0, n)


def _with_uniform(numbers, **fields):
    return dataclasses.replace(numbers, **{k: np.asarray(v) for k, v in fields.items()})


def test_draw_radius_inverts_the_hernquist_mass():
    model = analytic_model("hernquist")
    r_edge = float(model.profile.r[-1])
    table = radius_table(model, Settings(), r_edge)
    u = np.linspace(1e-6, 1.0 - 1e-6, 20001)
    r = draw_radius(table, 0, _with_uniform(_numbers(u.size), radius=u))
    # M(r) / M(r_e) for M(r) = (r / (1 + r))^2; the table's interpolation
    # between its nodes, 5.9e-8 measured
    u_drawn = (r / (1.0 + r)) ** 2 / (r_edge / (1.0 + r_edge)) ** 2
    np.testing.assert_allclose(u_drawn, u, rtol=1e-7)


@pytest.mark.parametrize("beta_c", [0.0, 0.3, -1.5])
def test_draw_angle_of_a_constant_anisotropy_is_the_beta_inverse(beta_c):
    model = _model_with([beta_c])
    u = np.linspace(1e-6, 1.0 - 1e-6, 20001)
    numbers = _with_uniform(_numbers(u.size), angle=u)
    cos_phi, sin_phi, factor = draw_angle(
        angle_table(model, Settings()), model, 0, np.ones(u.size), numbers
    )
    a = 1.0 - beta_c
    # the table's interpolation between its nodes, 1.7e-8 measured
    np.testing.assert_allclose(sin_phi**2, special.betaincinv(a, 0.5, u), rtol=1e-7)
    np.testing.assert_allclose(
        cos_phi**2, special.betaincinv(0.5, a, 1.0 - u), rtol=1e-7
    )
    assert np.all(factor == 1.0)


@pytest.mark.parametrize("kind", ["I", "II"])
def test_draw_angle_of_an_om_model_gives_its_speed_factor(kind):
    model = _model_with([0.0])
    profile = model.components[0].profile
    component = Component(profile=profile, anisotropy=Anisotropy(r_a=3.0, kind=kind))
    model = dataclasses.replace(model, components=(component,))
    n = 4096
    numbers = _numbers(n)
    r = np.geomspace(1e-3, 2.9, n)
    cos_phi, sin_phi, factor = draw_angle(
        angle_table(model, Settings()), model, 0, r, numbers
    )
    xi = (1.0 if kind == "I" else -1.0) * (r / 3.0) ** 2
    np.testing.assert_allclose(cos_phi**2 + sin_phi**2, 1.0, rtol=1e-14)
    np.testing.assert_allclose(factor, 1.0 / np.sqrt(1.0 + xi * sin_phi**2), rtol=1e-14)


def test_draw_speed_reads_a_linear_table_exactly():
    # a table whose inverse is s = lambda_u on every row: x = x_e u
    model = analytic_model("hernquist")
    r_edge = float(model.profile.r[-1])
    n_rows, n = 16, 64
    lambda_u = np.linspace(-41.0, 41.0, n)
    r = np.geomspace(1e-3, 0.5 * r_edge, 4001)
    psi_min = float(np.min(model.profile.p_trunc[model.profile.r <= r[-1]]))
    p_0, e_edge = float(model.profile.p_trunc[0]) * 1.5, 0.1 * psi_min
    table = SpeedTable(
        lambda_p=np.linspace(-30.0, 30.0, n_rows),
        dlambda_p=60.0 / (n_rows - 1),
        lambda_u=lambda_u,
        s=np.broadcast_to(lambda_u, (1, n_rows, n)),
        ds_dlambda_u=np.ones((1, n_rows, n)),
        ds_dlambda_p=np.zeros((1, n_rows, n)),
        x_edge=np.ones((1, n_rows)),
        p_0=p_0,
        e_edge=(e_edge,),
        validation=ValidationReport(keys=(), checks={}),
    )
    u = np.linspace(1e-6, 1.0 - 1e-6, r.size)
    numbers = _with_uniform(_numbers(u.size), speed=u)
    v, x, d, psi = draw_speed(table, model, 0, r, np.ones(u.size), numbers)
    x_e = np.sqrt(1.0 - e_edge / psi)
    np.testing.assert_allclose(x, x_e * u, rtol=1e-13)
    np.testing.assert_allclose(d, x_e * (1.0 - u), rtol=1e-13)
    np.testing.assert_allclose(v, np.sqrt(2.0 * psi) * x, rtol=1e-15)
    rows = model.profile.r[1:-1:50]
    _, _, _, psi_rows = draw_speed(
        table, model, 0, rows, np.ones(rows.size), _numbers(rows.size)
    )
    np.testing.assert_allclose(psi_rows, model.profile.p_trunc[1:-1:50], rtol=1e-14)


def test_draw_directions_builds_the_velocity_from_its_parts():
    n = 4096
    numbers = _numbers(n)
    r, v = np.geomspace(1e-3, 1e3, n), np.linspace(0.1, 2.0, n)
    phi = np.linspace(0.0, np.pi / 2.0, n)
    position, velocity = draw_directions(r, v, np.cos(phi), np.sin(phi), numbers)
    unit = position / r[:, None]
    np.testing.assert_allclose(np.linalg.norm(position, axis=1), r, rtol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(velocity, axis=1), v, rtol=1e-14)
    sign = np.where(numbers.radial_sign < 0.5, 1.0, -1.0)
    v_radial = np.sum(unit * velocity, axis=1)
    np.testing.assert_allclose(v_radial, sign * v * np.cos(phi), atol=1e-14 * v.max())


def test_validate_particles_refuses_a_particle_beyond_the_bound():
    model = analytic_model("hernquist")
    n = 8
    position = np.tile([1.0, 0.0, 0.0], (n, 1))
    velocity = np.tile([0.0, 0.1, 0.0], (n, 1))
    x, d, psi = np.full(n, 0.5), np.full(n, 0.5), np.full(n, 0.5)
    report = validate_particles(model, 0, position, velocity, x, d, psi)
    assert report.tier == "goal"
    with pytest.raises(AccuracyError):
        validate_particles(model, 0, position, velocity, x, -d, psi)
    with pytest.raises(AccuracyError):
        validate_particles(model, 0, position, 20.0 * velocity, x, d, psi)


def test_random_numbers_depend_only_on_seed_component_and_index():
    first = random_numbers(7, 0, 100)
    longer = random_numbers(7, 0, 1000)
    again = random_numbers(7, 0, 100)
    for field in dataclasses.fields(first):
        a, b = getattr(first, field.name), getattr(longer, field.name)
        assert np.array_equal(a, b[:100])
        assert np.array_equal(a, getattr(again, field.name))
    other = random_numbers(7, 1, 100)
    assert not np.any(other.radius == first.radius)


# reduced speed tables: 9 rows refined to at most 51 (9, 15, 27, 51), on
# 256 nodes; at 15 rows the error across rows is 1.7e-2 to 2.4e-2, beyond
# the hard limit, and the table is refused
REDUCED = {"n_rows": 9, "n_rows_max": 51, "n_nodes": 256}


def _lowered_hernquist(k_t=0.05, n_nodes=2001, stationary=False):
    r"""A truncated model from closed forms: the Hernquist DF lowered by its
    value at :math:`\mathcal{K}_{\rm t}`, ``F(K + k_t) - F(k_t)``, with its Taylor
    form below the table, on the Hernquist potential lowered by
    :math:`\mathcal{K}_{\rm t}`, whose last row is zero. With `stationary`, lowered
    by its slope there too, ``F(K + k_t) - F(k_t) - F'(k_t) K``, so that F_1 = 0
    (the stationary edge); its floor is higher, :math:`10^{-3}`, where the two
    subtractions keep 10 digits."""
    model = analytic_model("hernquist")
    p_0, k_floor = 1.0 - k_t, (1e-3 if stationary else 1e-6)
    n_lower = n_nodes // 2 + 1
    k = np.concatenate(
        [
            np.geomspace(k_floor, 0.5 * p_0, n_lower),
            p_0 - np.geomspace(0.5 * p_0, 1e-9, n_nodes - n_lower + 1)[1:],
        ]
    )
    df_0, f_1 = hernquist_df(k_t)
    df, ddf = hernquist_df(k + k_t)
    lowered = df - df_0 - (f_1 * k if stationary else 0.0)
    slope = ddf - (f_1 if stationary else 0.0)
    table = DfTable(
        k=k[None, :],
        log_df=np.log10(lowered)[None, :],
        log_df_slope=(k * slope / (lowered * np.log(10.0)))[None, :],
        i_first=(0,),
        i_last=(n_nodes - 1,),
        slopes_exact=(True,),
    )
    # F_2 = F''(k_t), the derivative of the closed-form F' by a central
    # difference of step 1e-5: truncation 1e-10 relative
    f_2 = float((hernquist_df(k_t + 1e-5)[1] - hernquist_df(k_t - 1e-5)[1]) / 2e-5)
    form = dataclasses.replace(
        model.form,
        p_0=p_0,
        tail_amplitude=(0.0,),
        below_floor=("taylor",),
        below_floor_coefficients=((0.0 if stationary else float(f_1), f_2, k_floor),),
    )
    r_t = 1.0 / k_t - 1.0
    r = np.geomspace(1e-6, r_t, n_nodes)
    p_trunc = 1.0 / (1.0 + r) - k_t
    p_trunc[-1] = 0.0
    # the last row is the edge: zero density there, and the density's edge
    # exponent of a DF with F ~ K at its edge (F ~ K^2 where stationary)
    rho_trunc = model.profile.rho_trunc[:, :1] * 0.0 + 2.0 / (r * (1.0 + r) ** 3)
    rho_trunc[:, -1] = 0.0
    profile = dataclasses.replace(
        model.profile,
        r=r,
        p_trunc=p_trunc,
        rho_trunc=rho_trunc,
        mass_trunc=(r / (1.0 + r)) ** 2,
        k_t=k_t,
        r_t=r_t,
        spatially_finite=True,
        edge_exponent=(density_edge_exponent(2.0 if stationary else 1.0, 0.0),),
    )
    return dataclasses.replace(model, table=table, form=form, profile=profile)


def _with_beta(model, beta_c):
    component = Component(
        profile=model.components[0].profile, anisotropy=Anisotropy(beta_c=beta_c)
    )
    return dataclasses.replace(model, components=(component,))


@pytest.mark.parametrize(
    ("truncated", "beta_c"),
    [
        (False, 0.0),
        (False, -0.5),
        (False, 0.25),
        (True, 0.0),
        ("stationary", 0.0),
    ],
)
def test_speed_table_rows_invert_the_speed_distribution(truncated, beta_c):
    # reduced tables of the Hernquist model, isotropic and anisotropic, and of
    # the lowered Hernquist model (a Taylor form below its table, a profile
    # ending at zero), also with a stationary edge (F_1 = 0); each row's nodes
    # against a direct quadrature of the density x^(2 - 2 beta_c)
    # F(Psi (1 - x^2)) of the model's own F
    settings = Settings(sampling=REDUCED)
    model = (
        _lowered_hernquist(stationary=truncated == "stationary")
        if truncated
        else analytic_model("hernquist")
    )
    model = dataclasses.replace(_with_beta(model, beta_c), settings=settings)
    table = speed_table(model, settings)
    p_0 = table.p_0
    power = 2.0 - 2.0 * beta_c

    def density(t, psi):
        return t**power * 10.0 ** log10_df(model.table, model.form, psi * (1.0 - t * t))

    rows = table.lambda_p.size
    lambda_u = table.lambda_u
    pick = np.flatnonzero(np.abs(lambda_u) < 20.0)[::25]
    for i in (1, rows // 2, rows - 2):
        psi = p_0 / (1.0 + np.exp(-table.lambda_p[i]))
        x = 1.0 / (1.0 + np.exp(-table.s[0, i, pick]))
        # the masses below and above each node, each from its own end
        below, above = (
            np.array(
                [
                    integrate.quad(
                        density, a, b, args=(psi,), epsabs=0.0, epsrel=1e-13
                    )[0]
                    for a, b in ends
                ]
            )
            for ends in ([(0.0, xi) for xi in x], [(xi, 1.0) for xi in x])
        )
        lambda_ref = np.log(below) - np.log(above)
        np.testing.assert_allclose(lambda_ref, lambda_u[pick], rtol=0, atol=1e-8)
    assert table.validation.keys == speed.SPEED_TABLE_KEYS


def test_the_taylor_rule_leads_with_k_unless_its_first_coefficient_is_zero():
    # the exponent of the leading term below the floor, F_1 K + F_2 K^2 / 2:
    # 1 where F_1 is not zero, 2 at a stationary edge
    settings = Settings(sampling=REDUCED)
    assert speed._Rows(_lowered_hernquist(), settings).floor_s == 1.0
    assert speed._Rows(_lowered_hernquist(stationary=True), settings).floor_s == 2.0


@pytest.mark.parametrize("truncated", [False, True])
def test_the_row_kernels_are_the_host_rows(truncated, monkeypatch):
    # the speed table by the row kernels (the default) against `_Rows.row`, one
    # row at a time on the host, the reference: the same operations on fixed
    # shapes, so the two differ by rounding (4e-13 in s on the battery)
    settings = Settings(sampling=REDUCED)
    model = _lowered_hernquist() if truncated else analytic_model("hernquist")
    model = dataclasses.replace(model, settings=settings)
    kernels = speed.build_speed_table(model, settings)
    monkeypatch.setattr(speed, "KERNEL_ROWS", False)
    host = speed.build_speed_table(model, settings)
    assert kernels["lambda_p"].size == host["lambda_p"].size
    for key in ("s", "ds_dlambda_u", "ds_dlambda_p", "x_edge"):
        np.testing.assert_allclose(kernels[key], host[key], rtol=0, atol=1e-11)


def test_speed_table_holds_the_goal_across_rows_at_the_default_rows():
    model = analytic_model("hernquist")
    table = speed_table(model, model.settings)
    row_error = table.validation.checks["row_error"]
    assert row_error.error <= GOAL
    assert table.validation.checks["polish_residual"].error <= GOAL
    assert 0.0 < row_error.where < table.p_0


def test_every_kernel_of_the_sampler_compiles_once_for_two_models():
    # two models of the same kinds and of other numbers (the DF table, the
    # anisotropy, the radius of the sampling): with their tables built, each
    # kernel called with the same shapes for both compiles once
    settings = Settings(sampling=REDUCED)
    models = [
        dataclasses.replace(analytic_model("hernquist"), settings=settings),
        dataclasses.replace(
            _with_beta(analytic_model("hernquist", n_nodes=1801), -0.25),
            settings=settings,
        ),
    ]
    r_samples = (100.0, 50.0)
    prepared = [
        (
            speed._Rows(model, settings),
            radius_table(model, settings, r_sample),
            angle_table(model, settings),
            speed_table(model, settings),
        )
        for model, r_sample in zip(models, r_samples, strict=True)
    ]
    kernels = [
        speed._log10_df_kernel,
        speed._kappa_kernel,
        stages._radius_kernel,
        stages._angle_kernel,
        stages._psi_kernel,
        stages._speed_kernel,
    ]
    energies = np.geomspace(1e-6, 0.9, 64)
    numbers = _numbers(256)
    sizes, outputs = [], []
    for model, (rows, radius, angle, speeds) in zip(models, prepared, strict=True):
        r = draw_radius(radius, 0, numbers)
        _, _, factor = draw_angle(angle, model, 0, r, numbers)
        v, _, _, _ = draw_speed(speeds, model, 0, r, factor, numbers)
        outputs.append((rows.log10_df(energies), rows.kappa(energies), v))
        sizes.append([kernel._cache_size() for kernel in kernels])
    assert sizes[1] == sizes[0]
    assert not np.array_equal(outputs[0][0], outputs[1][0])
    assert not np.array_equal(outputs[0][2], outputs[1][2])


def test_sampler_draws_the_hernquist_model_end_to_end(caplog):
    settings = Settings(sampling=REDUCED)
    model = dataclasses.replace(analytic_model("hernquist"), settings=settings)
    # the log, not the recent messages: those are a bounded queue, full after
    # the tests before this one, so that their length does not grow
    with caplog.at_level("DEBUG", logger="pianistpy"):
        sampler = Sampler(model, r_sample=100.0)
    # a sampler above the goal says so, with its check and its row
    said = [r.getMessage() for r in caplog.records]
    assert (sampler.validation.level > 0) == any("above the goal" in m for m in said)
    assert sampler.validation.checks["row_error"].error < HARD_LIMIT
    particles = sampler.sample(2000, seed=3)
    r = np.linalg.norm(particles.position, axis=1)
    assert particles.unit_system == "internal"
    assert np.all(np.isfinite(particles.position)) and np.all(
        np.isfinite(particles.velocity)
    )
    assert np.all(r < 100.0)
    # the mass inside r_sample, M(r) = (r / (1 + r))^2, to the profile's cut
    # there, linear in the logarithms between its rows: 4.2e-7 measured
    assert particles.mass.sum() == pytest.approx((100.0 / 101.0) ** 2, rel=1e-6)
    # the first particles of a larger draw are the same particles
    more = sampler.sample(4000, seed=3)
    assert np.array_equal(more.position[:2000], particles.position)
    assert np.array_equal(more.velocity[:2000], particles.velocity)
    # the same seed twice, and from a sampler built again, gives the same particles
    again = Sampler(model, r_sample=100.0).sample(2000, seed=3)
    assert np.array_equal(again.position, particles.position)
    assert np.array_equal(again.velocity, particles.velocity)
    # drawn in blocks (the last one cut), the same particles at any block size
    blocks = dataclasses.replace(
        model, settings=Settings(sampling={**REDUCED, "draw_block": 333})
    )
    in_blocks = Sampler(blocks, r_sample=100.0).sample(2000, seed=3)
    assert np.array_equal(in_blocks.position, particles.position)
    assert np.array_equal(in_blocks.velocity, particles.velocity)
    # converted at the output only
    in_model = Sampler(model.to("model", concentration=10.0), r_sample=100.0).sample(
        2000, seed=3
    )
    profile = model.components[0].profile
    factors = model.to("model", concentration=10.0).units.factors(profile)
    np.testing.assert_array_equal(in_model.position, particles.position)
    np.testing.assert_allclose(
        in_model.velocity, particles.velocity * factors.model_v, rtol=1e-15
    )


def test_the_speeds_window_is_two_speeds():
    sampler = Sampler(
        dataclasses.replace(
            analytic_model("hernquist"), settings=Settings(sampling=REDUCED)
        ),
        r_sample=100.0,
    )
    with pytest.raises(InputError, match="two speeds"):
        sampler.speeds(np.array([1.0]), v_range=("slow", "fast"))


def test_the_sampler_refuses_several_components():
    with pytest.raises(InputError, match="one component"):
        Sampler(_model_with([0.0, 0.25]), r_sample=100.0)


@pytest.mark.parametrize("edge_row", [False, True])
def test_the_draws_potential_is_the_profiles(edge_row):
    # the kernel reads the profile's tables as `potential` does: the Hermite
    # inside, the end rows' lines beyond them, and with an edge row the Kepler
    # form on the last interval and zero from r_t on
    model = _lowered_hernquist() if edge_row else analytic_model("hernquist")
    profile = model.profile
    assert profile.has_edge_row == edge_row
    rows = profile.r
    r = np.concatenate(
        [
            np.geomspace(0.1 * rows[0], rows[-1] * (2.0 if edge_row else 1.0), 20001),
            rows,
            0.5 * (rows[:-1] + rows[1:]),
        ]
    )
    psi = np.asarray(stages._psi_kernel(r, *stages._potential_tables(model)))
    np.testing.assert_allclose(psi, profile.potential(r), rtol=1e-14, atol=0.0)
    if edge_row:
        assert np.any(r > rows[-2]) and np.all(psi[r >= profile.r_t] == 0.0)


def test_a_potential_that_is_not_positive_inside_the_edge_is_refused_by_name():
    model = analytic_model("hernquist")
    p_trunc = np.array(model.profile.p_trunc)
    p_trunc[100] = 0.0
    model = dataclasses.replace(
        model, profile=dataclasses.replace(model.profile, p_trunc=p_trunc)
    )
    r = np.array([1.0])
    with pytest.raises(ResultError, match="positive inside its edge"):
        draw_speed(
            SpeedTable(
                lambda_p=np.zeros(4),
                dlambda_p=1.0,
                lambda_u=np.zeros(4),
                s=np.zeros((1, 4, 4)),
                ds_dlambda_u=np.zeros((1, 4, 4)),
                ds_dlambda_p=np.zeros((1, 4, 4)),
                x_edge=np.ones((1, 4)),
                p_0=1.0,
                e_edge=(0.0,),
                validation=ValidationReport(keys=(), checks={}),
            ),
            model,
            0,
            r,
            np.ones(1),
            _numbers(1),
        )


def test_an_empty_or_inverted_span_is_refused_by_name():
    settings = Settings(sampling=REDUCED)
    rows = speed._Rows(analytic_model("hernquist"), settings)
    with pytest.raises(ResultError, match="not above 1"):
        rows.hi_edges(0.5, 1.0)
    with pytest.raises(ResultError, match="not above 1"):
        speed._Rows(
            analytic_model("hernquist"),
            Settings(sampling=dict(REDUCED, speed_low_refined=0.5)),
        )
    flat = dataclasses.replace(
        analytic_model("hernquist"),
        profile=dataclasses.replace(
            analytic_model("hernquist").profile,
            p_trunc=np.zeros(analytic_model("hernquist").profile.r.size),
        ),
    )
    with pytest.raises(ResultError, match="potential"):
        speed.build_speed_table(flat, settings)


@pytest.fixture(scope="module")
def hernquist_sampler():
    """The analytic Hernquist model's sampler to r = 100, reduced settings."""
    settings = Settings(sampling=REDUCED)
    model = dataclasses.replace(analytic_model("hernquist"), settings=settings)
    return model, Sampler(model, r_sample=100.0)


def test_a_radius_range_draws_the_model_on_that_range(hernquist_sampler):
    model, whole = hernquist_sampler
    r_lo, r_hi = 0.5, 5.0
    ranged = Sampler(model, r_sample=100.0, r_range=(r_lo, r_hi))
    particles = ranged.sample(4000, seed=3)
    r = np.linalg.norm(particles.position, axis=1)
    # inside the range by construction, to the rounding of the logit
    assert np.all(r >= r_lo * (1.0 - 1e-12)) and np.all(r <= r_hi * (1.0 + 1e-12))

    # the range's mass, M(r) = (r / (1 + r))^2, to the profile's rows
    def mass(x):
        return (x / (1.0 + x)) ** 2

    assert particles.mass.sum() == pytest.approx(mass(r_hi) - mass(r_lo), rel=1e-6)
    # the radii follow M(r) conditioned on the range (Kolmogorov-Smirnov)
    cdf = (mass(np.sort(r)) - mass(r_lo)) / (mass(r_hi) - mass(r_lo))
    ecdf = np.arange(1, r.size + 1) / r.size
    distance = max(np.max(ecdf - cdf), np.max(cdf - (ecdf - 1.0 / r.size)))
    assert distance < 1.63 / np.sqrt(r.size)  # 1 % level
    # the same seed draws the same particles; the range is in the identity
    again = Sampler(model, r_sample=100.0, r_range=(r_lo, r_hi)).sample(4000, seed=3)
    assert np.array_equal(again.position, particles.position)
    assert ranged.identity != whole.identity


def test_the_whole_range_is_the_draw_without_one(hernquist_sampler):
    model, whole = hernquist_sampler
    full = Sampler(model, r_sample=100.0, r_range=(0.0, 100.0))
    assert full.bounds == [(0.0, 1.0)]
    a, b = whole.sample(1000, seed=5), full.sample(1000, seed=5)
    assert np.array_equal(a.position, b.position)
    assert np.array_equal(a.velocity, b.velocity)
    assert np.array_equal(a.mass, b.mass)


@pytest.mark.parametrize(
    "r_range, match",
    [
        ((5.0, 0.5), "r_lo < r_hi"),
        ((-1.0, 1.0), "r_lo < r_hi"),
        ((0.5, np.inf), "finite"),
        ((0.5,), "two radii"),
        ((0.5, 200.0), "beyond the outer radius"),
    ],
)
def test_a_radius_range_outside_the_sampling_is_refused(
    hernquist_sampler, r_range, match
):
    model, _ = hernquist_sampler
    with pytest.raises(InputError, match=match):
        Sampler(model, r_sample=100.0, r_range=r_range)


def _hernquist_speed_cdf(r, v):
    """The isotropic Hernquist model's speed distribution at r, by quadrature of
    F(P - v^2 / 2) v^2 on the closed-form DF, P = 1 / (1 + r)."""
    psi = 1.0 / (1.0 + r)

    def density(w):
        return float(hernquist_df(psi - 0.5 * w * w)[0]) * w * w

    v_esc = np.sqrt(2.0 * psi)
    total = integrate.quad(density, 0.0, v_esc, epsabs=0.0, epsrel=1e-12)[0]
    return (
        np.array(
            [
                integrate.quad(density, 0.0, min(x, v_esc), epsabs=0.0, epsrel=1e-12)[0]
                for x in np.atleast_1d(v)
            ]  # fmt: skip
        )
        / total
    )


@pytest.mark.parametrize("r", [0.3, 1.0, 5.0])
def test_a_speed_window_draws_the_speed_distribution_conditioned_on_it(
    hernquist_sampler, r
):
    _, sampler = hernquist_sampler
    v_esc = np.sqrt(2.0 / (1.0 + r))
    v_lo, v_hi = 0.25 * v_esc, 0.55 * v_esc
    n = 4000
    v, fraction = sampler.speeds(np.full(n, r), seed=4, v_range=(v_lo, v_hi))
    # every speed in the window, to the bisection's rounding
    assert np.all(v >= v_lo * (1.0 - 1e-10)) and np.all(v <= v_hi * (1.0 + 1e-10))
    # the window's fraction against the closed form, to the reduced table's
    # row error (warning tier, 3.4e-4)
    exact = np.diff(_hernquist_speed_cdf(r, [v_lo, v_hi]))[0]
    np.testing.assert_allclose(fraction, exact, rtol=2e-3)
    # the speeds follow the distribution conditioned on the window
    # (Kolmogorov-Smirnov against the closed form)
    base = _hernquist_speed_cdf(r, v_lo)[0]
    cdf = (_hernquist_speed_cdf(r, np.sort(v)) - base) / exact
    ecdf = np.arange(1, n + 1) / n
    distance = max(np.max(ecdf - cdf), np.max(cdf - (ecdf - 1.0 / n)))
    assert distance < 1.63 / np.sqrt(n)  # 1 % level


def test_the_whole_window_is_the_speed_draw_without_one(hernquist_sampler):
    # no window, or one wider than the escape speed: the speed numbers are
    # unmapped, so the speeds are draw_speed's at the same numbers, to the bit
    _, sampler = hernquist_sampler
    r = np.geomspace(1e-2, 50.0, 500)
    part = sampler.parts[0]
    numbers = stages.random_numbers_block(6, 0, 0, r.size)
    expected = stages.draw_speed(
        part.speed_table, part.model, 0, r, np.ones(r.size), numbers
    )[0]
    for window in (None, (0.0, 10.0)):
        v, fraction = sampler.speeds(r, seed=6, v_range=window)
        assert np.all(fraction == 1.0)
        assert np.array_equal(v, expected)


def test_adjacent_speed_windows_add_up(hernquist_sampler):
    _, sampler = hernquist_sampler
    r = np.array([0.1, 1.0, 10.0])
    v_esc = np.sqrt(2.0 / (1.0 + r))
    lo, mid, hi = 0.2 * v_esc.min(), 0.4 * v_esc.min(), 0.6 * v_esc.min()
    whole = sampler.speeds(r, v_range=(lo, hi))[1]
    parts = (
        sampler.speeds(r, v_range=(lo, mid))[1]
        + sampler.speeds(r, v_range=(mid, hi))[1]
    )
    np.testing.assert_allclose(parts, whole, rtol=0.0, atol=1e-13)


def test_a_window_at_each_radius_is_the_single_window_at_that_radius(hernquist_sampler):
    # bounds given per radius, here in units of each radius's escape speed: at
    # each radius the speed and the fraction are, to the byte, those of the one
    # window applied to every radius (the same speed numbers, particle i)
    _, sampler = hernquist_sampler
    r = np.array([0.3, 1.0, 5.0])
    v_esc = np.sqrt(2.0 / (1.0 + r))
    lo, hi = 0.25 * v_esc, 0.55 * v_esc
    v, fraction = sampler.speeds(r, seed=4, v_range=(lo, hi))
    for i in range(r.size):
        v_i, fraction_i = sampler.speeds(r, seed=4, v_range=(lo[i], hi[i]))
        assert v[i].tobytes() == v_i[i].tobytes()
        assert fraction[i].tobytes() == fraction_i[i].tobytes()
    # a scalar bound with an array bound: broadcast
    _, f_scalar_lo = sampler.speeds(r, seed=4, v_range=(0.0, hi))
    for i in range(r.size):
        assert (
            f_scalar_lo[i].tobytes()
            == sampler.speeds(r, seed=4, v_range=(0.0, hi[i]))[1][i].tobytes()
        )


@pytest.mark.parametrize(
    "v_range, match",
    [
        ((np.zeros(2), np.ones(2)), r"shape \(3,\).*v_lo of shape \(2,\)"),
        ((0.1, np.ones((3, 2))), r"v_hi of shape \(3, 2\)"),
        ((np.array([0.1, 0.6, 0.1]), np.full(3, 0.5)), "v_lo < v_hi at every radius"),
        ((np.array([0.1, -0.1, 0.1]), 0.5), "v_lo < v_hi at every radius"),
    ],
)
def test_windows_that_do_not_fit_the_radii_are_refused(
    hernquist_sampler, v_range, match
):
    _, sampler = hernquist_sampler
    with pytest.raises(InputError, match=match):
        sampler.speeds(np.array([0.3, 1.0, 5.0]), v_range=v_range)


def test_a_window_above_the_escape_speed_draws_nothing(hernquist_sampler):
    _, sampler = hernquist_sampler
    r = np.array([1.0, 10.0])
    v, fraction = sampler.speeds(r, v_range=(1.5, 2.0))  # v_esc(1) = 1
    assert np.all(fraction == 0.0) and np.all(np.isnan(v))


def test_a_speed_window_with_a_finite_anisotropy_radius_is_refused(hernquist_sampler):
    # the window couples to the drawn angle there; refused before any table is read
    import copy

    _, sampler = hernquist_sampler
    model = sampler.model
    om = Component(
        profile=model.components[0].profile, anisotropy=Anisotropy(r_a=3.0, kind="I")
    )
    coupled = copy.copy(sampler)
    coupled.model = dataclasses.replace(model, components=(om,))
    with pytest.raises(InputError, match="couples to the angle"):
        coupled.speeds([1.0], v_range=(0.1, 0.5))


@pytest.mark.parametrize(
    "r, v_range, match",
    [
        ([1.0], (0.5, 0.1), "v_lo < v_hi"),
        ([1.0], (-0.1, 0.5), "v_lo < v_hi"),
        ([1.0], (np.inf, np.inf), "v_lo finite"),
        ([1.0], (0.5,), "two speeds"),
        ([0.0], (0.1, 0.5), "0 < r"),
        ([200.0], (0.1, 0.5), "outer radius"),
    ],
)
def test_a_speed_window_outside_the_model_is_refused(
    hernquist_sampler, r, v_range, match
):
    _, sampler = hernquist_sampler
    with pytest.raises(InputError, match=match):
        sampler.speeds(r, v_range=v_range)
