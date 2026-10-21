"""Tests of the observables of a superposed model.

The tests compare the observables of a model built by `superpose` against the
log-likelihood of `log_likelihood`, against the second moments of
`pianistpy.superposition.moments`, against the observables of a single
component, and against the observables of the superposed model's `LosTable`.
"""

import dataclasses
import math

import numpy as np
import pytest

from pianistpy import (
    AccuracyError,
    Anisotropy,
    Component,
    DoublePowerLaw,
    Settings,
    Truncation,
    superpose,
)
from pianistpy.df import compute_df, truncate
from pianistpy.inference import Data, log_likelihood
from pianistpy.observables import Aperture, LosTable, sigma_los, sigma_los_aperture
from pianistpy.observables.stages import dispersion_profile
from pianistpy.particles.core import _component_view
from pianistpy.particles.tests.test_stages import _lowered_hernquist
from pianistpy.superposition import mixture, moments
from pianistpy.superposition.tests.test_core import _member
from pianistpy.validation import TIER_BOUNDS, ValidationReport

pytestmark = pytest.mark.slow

SETTINGS = Settings(
    observables={"n_radii": 400, "n_jeans_nodes": 200, "n_projection_nodes": 200}
)
MASS_0, R_0, CONCENTRATION = 1e9, 500.0, 10.0
WEIGHTS = (0.4, 0.6)
RADII = np.geomspace(0.05, 0.5, 6)
APERTURES = (
    Aperture.circle(0.3),
    Aperture.circle(0.2, r_off=0.8),
    Aperture.rectangle(0.4, 0.1, r_off=0.4, angle=0.4),
)


def _physical(model):
    """Return `model` in the physical unit system with concentration
    CONCENTRATION, mass_0 = MASS_0 solar masses and r_0 = R_0 parsecs."""
    return model.to("physical", concentration=CONCENTRATION, mass_0=MASS_0, r_0=R_0)


def _in_parsecs(aperture):
    """Return `aperture` with every length (r_off, r_ape, side_x, side_y)
    multiplied by R_0, from units of r_0 to parsecs."""
    record = aperture.to_dict()
    for name in ("r_off", "r_ape", "side_x", "side_y"):
        if record[name] is not None:
            record[name] = record[name] * R_0
    return Aperture.from_dict(record)


@pytest.fixture(scope="module")
def members():
    r"""Return two Method II models on the truncated profile of
    `_lowered_hernquist`, with :math:`\beta_{\rm c} = 0` and
    :math:`\beta_{\rm c} = -0.2`, both with SETTINGS."""
    parent = _lowered_hernquist()
    return [
        dataclasses.replace(_member(parent, beta_c), settings=SETTINGS)
        for beta_c in (0.0, -0.2)
    ]


@pytest.fixture(scope="module")
def superposed(members):
    """Return the superposition of the two models of `members`, with the
    weights WEIGHTS = (0.4, 0.6)."""
    return superpose(dict(zip(members, WEIGHTS, strict=True)))


@pytest.fixture(scope="module")
def tabled():
    r"""Return a superposed model, the model's `LosTable` and the model's first
    component.

    The first component is a Method II model with :math:`\beta_{\rm c} = 0`, the
    quick settings, and the DF table and truncated profile of the isotropic
    Hernquist model truncated by ``Truncation(k_t=0.05)``. The second
    component is the first component with its DF table kept only at every
    other energy and at the last energy. The two components therefore hold
    one DF on two energy grids. The weights are 0.3 for the first component
    and 0.7 for the second component.
    """
    df = compute_df(Component(DoublePowerLaw.hernquist(), Anisotropy(), name="hern"))
    truncated = truncate(df, Truncation(k_t=0.05))
    parent = dataclasses.replace(_member(truncated, 0.0), settings=Settings(quick=True))
    t = parent.table
    n = t.k.shape[1]
    keep = np.unique(np.r_[np.arange(0, n, 2), n - 1])
    thinned = dataclasses.replace(
        parent,
        table=dataclasses.replace(
            t,
            k=t.k[:, keep],
            log_df=t.log_df[:, keep],
            log_df_slope=t.log_df_slope[:, keep],
            i_last=(keep.size - 1,),
        ),
    )
    model = superpose({parent: 0.3, thinned: 0.7})
    return model, LosTable(model), parent


def test_the_mixture_is_the_inferences_at_rounding(superposed):
    """Assert that `log_likelihood` reaches its Gaussian maximum on exact data.

    The data are the superposed model's own `sigma_los` at 6 projected radii and
    `sigma_los_aperture` in 3 apertures, in the physical system, each with an
    error of :math:`10^{-6}` times the value. The maximum is
    ``-sum(log(errors)) - n/2 log(2 pi)`` over the data, with every residual
    zero. The test asserts that the log-likelihood of the superposed model
    differs from the maximum by less than :math:`10^{-9}`.
    """
    physical = _physical(superposed)
    points = sigma_los(physical, RADII * R_0).value
    apertures = np.array(
        [sigma_los_aperture(physical, _in_parsecs(a)).value for a in APERTURES]
    )
    data = [
        Data(
            "sigma_los",
            points,
            r_2d=RADII * R_0,
            errors=1e-6 * points,
            unit_system="physical",
        ),
        Data(
            "sigma_los_aperture",
            apertures,
            apertures=tuple(_in_parsecs(a) for a in APERTURES),
            errors=1e-6 * apertures,
            unit_system="physical",
        ),
    ]
    best = sum(
        -float(np.sum(np.log(d.errors))) - 0.5 * d.values.size * math.log(2.0 * math.pi)
        for d in data
    )
    assert abs(log_likelihood(physical, data, SETTINGS) - best) < 1e-9


def test_a_member_beyond_0_01_refuses_the_superposition(members):
    """Assert that a member of tier "refused" makes `sigma_los` raise AccuracyError.

    The second member's validation report is replaced by a report whose
    ``parents_level`` is 3, the tier "refused", above :math:`10^{-2}`. The test
    asserts that `sigma_los` of the superposition of the first member and the
    modified second member raises AccuracyError. With
    ``Settings(write_anyway=True)`` on both members, the test asserts that
    `sigma_los` of the superposed model equals, bit for bit, `sigma_los` of the
    superposition of the two unmodified members.
    """
    a, b = members

    def refused(model, settings):
        report = model.validation
        return dataclasses.replace(
            model,
            settings=settings,
            validation=ValidationReport(
                keys=report.keys, checks=report.checks, parents_level=3
            ),
        )

    model = superpose({a: 0.4, refused(b, SETTINGS): 0.6})
    with pytest.raises(AccuracyError, match="beyond 0.01"):
        sigma_los(model, RADII)
    anyway = Settings(
        write_anyway=True,
        observables={"n_radii": 400, "n_jeans_nodes": 200, "n_projection_nodes": 200},
    )
    asked = superpose(
        {
            dataclasses.replace(a, settings=anyway): 0.4,
            refused(b, anyway): 0.6,
        }
    )
    np.testing.assert_array_equal(
        sigma_los(asked, RADII),
        sigma_los(superpose({a: 0.4, b: 0.6}), RADII),
    )


def test_a_weight_near_one_is_the_member(members):
    """Assert that weights 1 and :math:`10^{-14}` give the heavier member.

    For each order of the two members, `sigma_los` at 6 projected radii and
    `sigma_los_aperture` in 3 apertures of the superposed model agree with the
    observables of the member of weight 1 to a relative difference of
    :math:`10^{-13}`.
    """
    a, b = members
    for heavy, light in ((a, b), (b, a)):
        model = superpose({heavy: 1.0, light: 1e-14})
        np.testing.assert_allclose(
            sigma_los(model, RADII), sigma_los(heavy, RADII), rtol=1e-13
        )
        for aperture in APERTURES:
            assert sigma_los_aperture(model, aperture) == pytest.approx(
                sigma_los_aperture(heavy, aperture), rel=1e-13
            )


def test_the_table_and_the_model_routes(tabled):
    """Assert that the observables of a superposed model and its LosTable agree.

    The test asserts that the LosTable holds 2 parts and that its validation
    tier is "silent". `sigma_los` of the LosTable equals `sigma_los` of the
    model bit for bit. `sigma_los_aperture` of the LosTable agrees, to a
    relative difference of :math:`10^{-6}`, with `sigma_los_aperture` of the
    model and with `sigma_los_aperture` of the first component alone.
    `sigma_los` of the LosTable converted to the physical system equals, bit
    for bit, `sigma_los` of the model converted to the physical system.
    """
    model, table, parent = tabled
    assert len(table.parts) == 2 and table.validation.tier == "silent"
    np.testing.assert_array_equal(sigma_los(table, RADII), sigma_los(model, RADII))
    for aperture in APERTURES:
        assert sigma_los_aperture(table, aperture) == pytest.approx(
            sigma_los_aperture(model, aperture), rel=TIER_BOUNDS[0]
        )
        # The two components hold one DF on two energy grids, so the mixture of
        # the two components equals the first component alone.
        assert sigma_los_aperture(table, aperture) == pytest.approx(
            sigma_los_aperture(parent, aperture), rel=TIER_BOUNDS[0]
        )
    # The LosTable converted to the physical system must return the values of
    # the model converted to the physical system.
    physical = table.to("physical", concentration=CONCENTRATION, mass_0=MASS_0, r_0=R_0)
    np.testing.assert_array_equal(
        sigma_los(physical, RADII * R_0).value,
        sigma_los(_physical(model), RADII * R_0).value,
    )


def test_the_intrinsic_moments_are_the_superpositions(superposed):
    """Assert that the mixed radial dispersions match `moments`.

    The squared radial dispersion of each component, computed by
    `dispersion_profile` on the component as a model of one component, is
    mixed with the weights by `mixture`. At 20 radii up to 0.9 times the
    truncation radius, the test asserts that the mixture agrees with the
    ``sigma_r2`` of `moments`, with 3200 Gauss-Legendre nodes of the Jeans
    integral, to a relative difference of :math:`10^{-6}`.
    """
    model = superposed
    r = np.geomspace(1e-2, 0.9 * float(model.profile.r_t), 20)
    sigma_r2 = np.array(
        [
            dispersion_profile(view, SETTINGS).read(r)[0][0] ** 2
            for view in (_component_view(model, k) for k in range(2))
        ]
    )
    expected = moments(model, r, n_nodes=3200)["sigma_r2"]
    np.testing.assert_allclose(
        mixture(np.array(WEIGHTS), sigma_r2), expected, rtol=TIER_BOUNDS[0]
    )


def test_the_public_intrinsic_moments_of_a_superposition(superposed):
    """Assert that `sigma_r` and `sigma_t` of a superposed model are the mixtures.

    At 20 radii up to 0.9 times the truncation radius, the test asserts that
    the square of `sigma_r` agrees with the ``sigma_r2`` of `moments`, with 3200
    Gauss-Legendre nodes, to a relative difference of :math:`10^{-6}`. The test
    asserts that the square of `sigma_t` agrees with the squared tangential
    dispersions of the components, from `dispersion_profile` and mixed with
    the weights by `mixture`, to a relative difference of :math:`10^{-12}`. The
    two members share one density, so the density-weighted mixture of the
    components' dispersions equals the mixture with the weights alone.
    """
    from pianistpy.observables import sigma_r, sigma_t

    model = superposed
    r = np.geomspace(1e-2, 0.9 * float(model.profile.r_t), 20)
    expected = moments(model, r, n_nodes=3200)["sigma_r2"]
    np.testing.assert_allclose(sigma_r(model, r) ** 2, expected, rtol=TIER_BOUNDS[0])
    views = [_component_view(model, k) for k in range(2)]
    reads = [dispersion_profile(v, SETTINGS).read(r)[1][0] ** 2 for v in views]
    np.testing.assert_allclose(
        sigma_t(model, r) ** 2, mixture(np.array(WEIGHTS), np.array(reads)), rtol=1e-12
    )


def test_the_aperture_losvd_of_a_superposition_integrates_to_one(tabled):
    """Assert that aperture_losvd of a superposed LosTable is a normalized PDF.

    In a circular aperture of radius 0.2 and on 6001 velocities from -1.5 to
    1.5, the test asserts that the aperture LOSVD is nonnegative, that the
    trapezoidal integral of the aperture LOSVD equals 1 to a relative
    difference of :math:`10^{-4}`, and that the square root of the trapezoidal
    second moment equals sigma_los_aperture to a relative difference of
    :math:`10^{-4}`.
    """
    from pianistpy.observables import aperture_losvd

    _, table, _ = tabled
    half = np.linspace(0.0, 1.5, 3001)
    v = np.concatenate([-half[:0:-1], half])
    d = aperture_losvd(table, Aperture.circle(0.2), v)
    assert np.all(d >= 0.0)
    assert np.trapezoid(d, v) == pytest.approx(1.0, rel=1e-4)
    assert math.sqrt(np.trapezoid(v**2 * d, v)) == pytest.approx(
        float(sigma_los_aperture(table, Aperture.circle(0.2))), rel=1e-4
    )


def test_the_public_losvd_of_a_superposition_integrates_to_one(tabled):
    """Assert that losvd of a superposed LosTable is a normalized PDF.

    At projected radius 0.1 and on 6001 velocities from -1.5 to 1.5, the test
    asserts that the LOSVD is nonnegative and that the trapezoidal integral of
    the LOSVD equals 1 to a relative difference of :math:`10^{-4}`.
    """
    from pianistpy.observables import losvd

    _, table, _ = tabled
    # The velocity 1.5 exceeds sqrt(2), the escape velocity at the center of the
    # untruncated Hernquist model. The truncated model's escape velocity is lower.
    half = np.linspace(0.0, 1.5, 3001)
    v = np.concatenate([-half[:0:-1], half])
    d = losvd(table, 0.1, v)
    assert np.all(d >= 0.0)
    assert np.trapezoid(d, v) == pytest.approx(1.0, rel=1e-4)
