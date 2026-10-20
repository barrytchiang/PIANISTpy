"""The log-likelihood of binned distributions of velocity (298 stage 2): the
exact cumulative of a distribution on its sub-panels, the Data, the value
against the members' distributions integrated in NumPy, the mass scale's
gradient, the length scale held, and a free component's anisotropy. The
aperture's distribution is tested on the isotropic member alone."""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pianistpy import (
    Anisotropy,
    Component,
    DoublePowerLaw,
    InputError,
    Settings,
    Truncation,
    superpose,
)
from pianistpy.df import compute_df, truncate
from pianistpy.inference import Data, _binned, likelihood
from pianistpy.observables._panels import read_panels
from pianistpy.observables.aperture import Aperture
from pianistpy.observables.stages import (
    dispersion_profile,
    losvd_row,
    projected_dispersion,
)
from pianistpy.particles.core import component_view

# The quick tier: the members' tables of rows, which the binned reads need,
# are minutes a member there against tens of minutes at the default tier.
SETTINGS = Settings(
    quick=True,
    observables={"n_radii": 400, "n_jeans_nodes": 200, "n_projection_nodes": 200},
)
MASS_0, R_0, CONCENTRATION = 1e9, 500.0, 10.0
K_T = 0.0079589980165958
THETA = (-0.2, 1.8)
WEIGHTS = np.array([0.4, 0.6])


def _physical(model):
    return model.to("physical", concentration=CONCENTRATION, mass_0=MASS_0, r_0=R_0)


def test_the_cumulative_is_exact_on_each_sub_panel():
    # a cubic on [0, 1] and a quadratic on [1, 3], at four Gauss nodes each
    x, w = np.polynomial.legendre.leggauss(4)
    nodes = np.concatenate([0.5 + 0.5 * x, 2.0 + x])
    f = [lambda v: 1.0 + v - v**3, lambda v: 2.0 - 0.5 * (v - 1.0) ** 2]
    big_f = [lambda v: v + v**2 / 2 - v**4 / 4, lambda v: 2.0 * v - (v - 1.0) ** 3 / 6]
    values = np.concatenate([f[0](nodes[:4]), f[1](nodes[4:])])
    p = _binned.panels(nodes, np.concatenate([0.5 * w, w]), values)
    np.testing.assert_allclose(p.a, [0.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(p.b, [1.0, 3.0], atol=1e-15)
    u = np.array([0.0, 0.3, 1.0, 1.7, 3.0, 4.0])
    exact = np.where(
        u <= 1.0,
        big_f[0](np.minimum(u, 1.0)),
        big_f[0](1.0) + big_f[1](np.minimum(u, 3.0)) - big_f[1](1.0),
    )
    got = np.asarray(_binned.cumulative(p.a, p.b, p.series, jnp.asarray(u)))
    np.testing.assert_allclose(got, exact, rtol=0.0, atol=1e-14)
    a, b, series = _binned.padded(p, 7)
    np.testing.assert_array_equal(
        np.asarray(_binned.cumulative(a, b, series, jnp.asarray(u))), got
    )
    np.testing.assert_allclose(
        read_panels(p, u[1:4]), [f[0](0.3), f[0](1.0), f[1](1.7)], atol=1e-13
    )


def test_the_sub_panels_of_a_graded_rule():
    # the speed distribution's rule (Gauss-Legendre panels graded toward both
    # ends of each interval between breaks): its panels found from the rule,
    # and a cubic integrated exactly across them
    from pianistpy.observables._speed import graded_rule

    rules = [graded_rule(0.0, 0.4, 5, 6), graded_rule(0.4, 1.0, 5, 6)]
    nodes = np.concatenate([r[0] for r in rules])
    weights = np.concatenate([r[1] for r in rules])
    p = _binned.panels(nodes, weights, 1.0 + nodes - nodes**3)
    np.testing.assert_allclose(p.b[:-1], p.a[1:], rtol=0.0, atol=1e-15)
    assert (p.a[0], p.b[-1]) == pytest.approx((0.0, 1.0), abs=1e-15)
    u = np.array([0.0, 0.05, 0.4, 0.77, 1.0])
    np.testing.assert_allclose(
        np.asarray(_binned.cumulative(p.a, p.b, p.series, jnp.asarray(u))),
        u + u**2 / 2 - u**4 / 4,
        rtol=0.0,
        atol=1e-14,
    )


def test_the_binned_data():
    edges = np.linspace(-50.0, 50.0, 6)
    Data("losvd", np.full(5, 0.2), r_2d=[100.0], v_edges=edges, errors=np.full(5, 0.01))
    Data(
        "pdf_v",
        np.full(5, 0.2),
        r=[500.0],
        v_edges=edges + 50.0,
        errors=np.full(5, 0.01),
    )
    with pytest.raises(InputError, match="v_edges"):
        Data("losvd", np.full(5, 0.2), r_2d=[100.0], errors=np.full(5, 0.01))
    with pytest.raises(InputError, match="not negative"):
        Data(
            "pdf_v", np.full(5, 0.2), r=[500.0], v_edges=edges, errors=np.full(5, 0.01)
        )
    with pytest.raises(InputError, match="1 apertures"):
        Data(
            "losvd_aperture",
            np.full(5, 0.2),
            apertures=(Aperture.circle(100.0), Aperture.circle(50.0)),
            v_edges=edges,
            errors=np.full(5, 0.01),
        )
    with pytest.raises(InputError, match="no v_edges"):
        Data(
            "sigma_los",
            np.ones(2),
            r_2d=[1.0, 2.0],
            v_edges=edges[:3],
            errors=np.ones(2),
        )


def _pair():
    """S_E1's pair on its truncated NFW profile, isotropic and COM Type I, in
    physical units (the truncated Hernquist COM's table of rows is refused, a
    panel layout split at its edge), and the velocity unit."""
    parent = truncate(
        compute_df(Component(DoublePowerLaw.nfw(), Anisotropy()), SETTINGS),
        Truncation(k_t=K_T),
    )
    models = [
        _physical(
            truncate(
                compute_df(Component(profile=parent.profile, anisotropy=a), SETTINGS)
            )
        )
        for a in (Anisotropy(), Anisotropy(beta_c=THETA[0], r_a=THETA[1]))
    ]
    v = float(
        models[0].units.factors(models[0].components[0].unit_profile).physical_v_kms
    )
    return models, v


def _data(kinds, v):
    """Data of each kind in eight bins, an even share moved by 3 percent
    cosines, errors of 2 percent: a LOSVD at a radius, one in an aperture, and
    a speed distribution."""
    shapes = {
        "losvd": dict(r_2d=[0.5 * R_0], v_edges=v * np.linspace(-0.6, 0.6, 9)),
        "losvd_aperture": dict(
            apertures=(Aperture.circle(0.3 * R_0, r_off=0.4 * R_0),),
            v_edges=v * np.linspace(-0.5, 0.7, 9),
        ),
        "pdf_v": dict(r=[R_0], v_edges=v * np.linspace(0.0, 0.9, 9)),
    }
    return [
        Data(
            k,
            0.125 * (1.0 + 0.03 * np.cos(np.arange(8))),
            errors=np.full(8, 0.02 * 0.125),
            unit_system="physical",
            **shapes[k],
        )
        for k in kinds
    ]


@pytest.fixture(scope="module")
def pair():
    return _pair()


@pytest.fixture(scope="module")
def case(pair):
    """The pair superposed with a LOSVD at a radius and a speed distribution,
    and the fixed likelihood, built once."""
    models, v = pair
    model = superpose({models[0]: WEIGHTS[0], models[1]: WEIGHTS[1]})
    data = _data(("losvd", "pdf_v"), v)
    return model, data, v, likelihood(model, data, SETTINGS)


def _numpy_value(function, data, v, weights):
    """The log-likelihood with each block's shares integrated in NumPy."""
    expected = 0.0
    for block, d in zip(function.binned, data, strict=True):
        share = _numpy_shares(block, weights, np.asarray(d.v_edges) / v)
        r = d.values - share
        expected += -0.5 * np.sum((r / d.errors) ** 2) - np.sum(np.log(d.errors))
        expected -= 0.5 * r.size * math.log(2.0 * math.pi)
    return expected


def _numpy_shares(block, weights, u):
    """The window's share in each bin of the mixture, each member's
    distribution integrated over the bins by Gauss-Legendre (64 nodes a
    sub-panel piece) in NumPy, through `read_panels`."""
    x, wx = np.polynomial.legendre.leggauss(64)
    a, b, series = np.asarray(block.a), np.asarray(block.b), np.asarray(block.series)
    signed = block.kind != "pdf_v"
    inside = np.zeros(u.size - 1)
    for k in range(a.shape[0]):
        p = _binned.Panels(a[k], b[k], np.zeros(0), np.zeros(0), series[k])
        cuts = np.unique(np.concatenate([np.abs(u), a[k], b[k], [0.0]]))

        def integral(lo, hi, p=p, cuts=cuts):
            # from lo to hi >= lo >= 0, piece by piece between the cuts
            pts = cuts[(cuts > lo) & (cuts < hi)]
            ends = np.concatenate([[lo], pts, [hi]])
            total = 0.0
            for s0, s1 in zip(ends[:-1], ends[1:], strict=True):
                v = 0.5 * (s0 + s1) + 0.5 * (s1 - s0) * x
                total += 0.5 * (s1 - s0) * float(wx @ read_panels(p, v))
            return total

        def signed_cumulative(t):
            return (
                math.copysign(integral(0.0, abs(t)), t) if signed else integral(0.0, t)
            )

        c = np.array([signed_cumulative(t) for t in u])
        inside += weights[k] * float(block.share[k]) * np.diff(c)
    return inside / inside.sum()


def test_the_value_against_numpy(case):
    _, data, v, function = case
    expected = _numpy_value(function, data, v, WEIGHTS)
    assert abs(float(function()) / expected - 1.0) < 1e-12


def test_the_aperture_value_against_numpy(pair):
    # the aperture's distribution on the isotropic member alone
    models, v = pair
    data = _data(("losvd_aperture",), v)
    function = likelihood(models[0], data, SETTINGS)
    expected = _numpy_value(function, data, v, np.ones(1))
    assert abs(float(function()) / expected - 1.0) < 1e-12


def test_the_mass_scale_through_the_bin_edges(case):
    _, _, _, function = case
    scale = np.array([MASS_0, R_0])
    gradient = np.asarray(
        jax.grad(lambda s: function(scale=jnp.stack([s, R_0])))(MASS_0)
    )
    for h in (1e-4, 5e-5):
        central = (
            float(function(scale=[MASS_0 * (1 + h), R_0]))
            - float(function(scale=[MASS_0 * (1 - h), R_0]))
        ) / (2 * h * MASS_0)
        if h == 1e-4:
            first = central
    estimate = abs(central - first) / 3.0
    reference = central + (central - first) / 3.0
    assert abs(gradient - reference) <= 3.0 * estimate + 1e-9 * abs(reference)
    # the length scale is held at the model's own: another is refused, and its
    # gradient is not carried
    assert float(function(scale=[MASS_0, 0.9 * R_0])) == -np.inf
    assert np.isnan(np.asarray(jax.grad(lambda s: function(scale=s))(scale))[1])


def test_a_free_component_with_binned_data_is_refused_by_name(case):
    # until the observables' VJP of the distributions lands
    model, data, _, _ = case
    with pytest.raises(InputError, match="binned data"):
        likelihood(model, data, SETTINGS, free=(1,))


def test_the_frozen_reads_at_the_own_theta(case):
    # the reference the VJP will be tested against (central differences of
    # the reads on a placement held at theta): at the component's own theta
    # the frozen reads are its reads, bit for bit
    model, data, v, _ = case
    member = component_view(model, 1)
    own = _binned.pieces(member, SETTINGS, False)
    frozen = _frozen_pieces(member, SETTINGS, own)
    for d in data:
        if d.observable == "losvd_aperture":
            continue
        where = float(np.asarray(d.r_2d if d.observable == "losvd" else d.r)[0])
        got = [
            _binned.read(d.observable, where / R_0, p, SETTINGS) for p in (own, frozen)
        ]
        np.testing.assert_array_equal(got[0][0].series, got[1][0].series)
        assert got[0][1] == got[1][1]


def _frozen_pieces(model, settings, at):
    """The tables of `model` on the placement of `at`: the rows recomputed
    at `at`'s radii with `at`'s kept ranges, every other choice of the rows
    their own; read at theta +- h and differenced, the reference for the
    derivative of the reads at frozen placement."""
    profile = dispersion_profile(model, settings)
    projected = projected_dispersion(model, profile, settings)
    table = None
    if at.table is not None:
        rows = tuple(
            losvd_row(model, float(row.r_2d), at.kept, projected, settings)
            for row in at.table.rows
        )
        table = dataclasses.replace(at.table, rows=rows)
    return _binned.Pieces(model, profile, projected, at.kept, table)
