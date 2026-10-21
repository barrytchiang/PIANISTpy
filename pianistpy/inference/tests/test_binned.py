"""Tests of the log-likelihood of binned distributions of velocity.

The tests check the exact cumulative distribution on each velocity interval,
the checks of `Data` for binned observables, the log-likelihood against the
components' distributions integrated in NumPy, the gradient in the mass
scale, the length scale held at the model's own length scale, and the
exception for a free component with binned data. The distribution in an
aperture is tested on the isotropic component alone."""

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

pytestmark = pytest.mark.slow

# The quick settings keep the components' tables of line-of-sight velocity
# distributions, which the binned distributions read, short enough to compute
# within a test.
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
    # A cubic on [0, 1] and a quadratic on [1, 3], each sampled at four
    # Gauss-Legendre nodes, have exact antiderivatives to compare with.
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
    # The velocity distribution's quadrature rule grades Gauss-Legendre intervals
    # toward both ends of each interval between breaks; the velocity
    # intervals recovered from the rule's nodes must integrate a cubic exactly.
    from pianistpy.observables._velocity import graded_rule

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
    with pytest.raises(InputError, match="must not be negative"):
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
    with pytest.raises(InputError, match="neither v_edges nor r"):
        Data(
            "sigma_los",
            np.ones(2),
            r_2d=[1.0, 2.0],
            v_edges=edges[:3],
            errors=np.ones(2),
        )


def _pair():
    """Return two Method II components on one truncated NFW profile, one
    isotropic and one Cuddeford-Osipkov-Merritt (COM) of Type I, in physical
    units, and the velocity unit in kilometers per second. The truncated NFW
    profile is used because the table of line-of-sight velocity
    distributions of a COM component on a truncated Hernquist profile raises
    an exception at the profile's edge."""
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
    """Return one `Data` for each observable in `kinds`, in eight bins: the
    fraction 1/8 in each bin, moved by cosines of amplitude 3 percent, with
    errors of 2 percent of 1/8. The shapes are a line-of-sight velocity
    distribution at a projected radius, a line-of-sight velocity
    distribution in an aperture, and a velocity distribution at a 3D radius."""
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
    """Return the two components superposed at WEIGHTS, `Data` of a
    line-of-sight velocity distribution at a projected radius and of a velocity
    distribution, the velocity unit, and the likelihood, built once for the
    module."""
    models, v = pair
    model = superpose({models[0]: WEIGHTS[0], models[1]: WEIGHTS[1]})
    data = _data(("losvd", "pdf_v"), v)
    return model, data, v, likelihood(model, data, SETTINGS)


def _numpy_value(function, data, v, weights):
    """Return the log-likelihood with the fraction in each bin of each block
    integrated in NumPy."""
    expected = 0.0
    for block, d in zip(function.binned, data, strict=True):
        share = _numpy_shares(block, weights, np.asarray(d.v_edges) / v)
        r = d.values - share
        expected += -0.5 * np.sum((r / d.errors) ** 2) - np.sum(np.log(d.errors))
        expected -= 0.5 * r.size * math.log(2.0 * math.pi)
    return expected


def _numpy_shares(block, weights, u):
    """Return the fraction of the mixture's distribution inside the
    user-specified range that falls in each bin. Each component's
    distribution is integrated over the bins in NumPy by Gauss-Legendre
    quadrature with 64 nodes on each piece between velocity-interval ends
    and bin edges, through `read_panels`."""
    x, wx = np.polynomial.legendre.leggauss(64)
    a, b, series = np.asarray(block.a), np.asarray(block.b), np.asarray(block.series)
    signed = block.kind != "pdf_v"
    inside = np.zeros(u.size - 1)
    for k in range(a.shape[0]):
        p = _binned.Panels(a[k], b[k], np.zeros(0), np.zeros(0), series[k])
        cuts = np.unique(np.concatenate([np.abs(u), a[k], b[k], [0.0]]))

        def integral(lo, hi, p=p, cuts=cuts):
            # The integral from lo to hi, with hi >= lo >= 0, is summed over
            # the pieces between consecutive cuts, on each of which the
            # distribution is one polynomial.
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
    # The distribution in an aperture is tested on the isotropic component
    # alone, which keeps the test to one table of distributions.
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
    # The binned distributions are computed at the model's own length scale:
    # any other length scale gives minus infinity, and the gradient in the
    # length scale is NaN.
    assert float(function(scale=[MASS_0, 0.9 * R_0])) == -np.inf
    assert np.isnan(np.asarray(jax.grad(lambda s: function(scale=s))(scale))[1])


def test_a_free_component_with_binned_data_is_refused_with_cause(case):
    # The likelihood has no derivative of the binned distributions in the
    # anisotropy, so a free component with binned data raises InputError.
    model, data, _, _ = case
    with pytest.raises(InputError, match="binned data"):
        likelihood(model, data, SETTINGS, free=(1,))


def test_the_frozen_reads_at_the_own_theta(case):
    # At the component's own anisotropy, the reads on the frozen placement of
    # the tables must equal the component's own reads bit for bit, so that
    # central differences on the frozen placement are a valid reference for
    # a derivative of the reads.
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
    """Return the tables of `model` on the placement of `at`: the tables
    recomputed at the radii of `at` with the kept ranges of `at`, and every
    other choice of the tables taken from `model`. Reads of the returned
    tables at the anisotropy plus and minus a step, differenced, are the
    reference for a derivative of the reads at a frozen placement."""
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
