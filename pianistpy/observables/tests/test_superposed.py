"""The observables of a superposed model: its components summed with their
weights, against the inference's mixture, the superposition's moments and
the members alone, and its prepared table against the model."""

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
from pianistpy.validation import GOAL, ValidationReport

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
    return model.to("physical", concentration=CONCENTRATION, mass_0=MASS_0, r_0=R_0)


def _in_parsecs(aperture):
    record = aperture.to_dict()
    for name in ("r_off", "r_ape", "side_x", "side_y"):
        if record[name] is not None:
            record[name] = record[name] * R_0
    return Aperture.from_dict(record)


@pytest.fixture(scope="module")
def members():
    parent = _lowered_hernquist()
    return [
        dataclasses.replace(_member(parent, beta_c), settings=SETTINGS)
        for beta_c in (0.0, -0.2)
    ]


@pytest.fixture(scope="module")
def superposed(members):
    return superpose(dict(zip(members, WEIGHTS, strict=True)))


@pytest.fixture(scope="module")
def tabled():
    """A superposition whose components' tables are certified, at the quick
    tier: the truncated Hernquist model (`truncate`) as a child of its own
    truncated profile, and the same with its DF table on every other energy,
    a grid of its own; and its LosTable. (The members above share a lowered
    DF with the untruncated density, which their LOSVD's norm check refuses
    at the edge.)"""
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
    # data equal to the observables of the mixture, with errors of 1e-6 of
    # them: the inference's log-likelihood is its maximum, the residuals at
    # rounding
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


def test_a_member_beyond_the_hard_limit_refuses_the_superposition(members):
    # the superposition's report holds each check at its worst over its
    # members: one refused member refuses it, unless asked for anyway
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
    with pytest.raises(AccuracyError, match="hard limit"):
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
    model, table, parent = tabled
    assert len(table.parts) == 2 and table.validation.tier == "goal"
    np.testing.assert_array_equal(sigma_los(table, RADII), sigma_los(model, RADII))
    for aperture in APERTURES:
        assert sigma_los_aperture(table, aperture) == pytest.approx(
            sigma_los_aperture(model, aperture), rel=GOAL
        )
        # the two components are one DF on two grids: the mixture is the model
        assert sigma_los_aperture(table, aperture) == pytest.approx(
            sigma_los_aperture(parent, aperture), rel=GOAL
        )
    # in the physical system, from the table presented in it
    physical = table.to("physical", concentration=CONCENTRATION, mass_0=MASS_0, r_0=R_0)
    np.testing.assert_array_equal(
        sigma_los(physical, RADII * R_0).value,
        sigma_los(_physical(model), RADII * R_0).value,
    )


def test_the_intrinsic_moments_are_the_superpositions(superposed):
    # each component's profile, of the component as a model of one component,
    # mixed with the weights against `moments`, its Jeans rule converged (3,200
    # nodes a row); the two rules agree to the goal
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
        mixture(np.array(WEIGHTS), sigma_r2), expected, rtol=GOAL
    )


def test_the_public_intrinsic_moments_of_a_superposition(superposed):
    # sigma_r and sigma_t of the superposition: the components' second moments
    # mixed by their density at each radius (the members share the parent's
    # density, so the mixture is the one `moments` makes)
    from pianistpy.observables import sigma_r, sigma_t

    model = superposed
    r = np.geomspace(1e-2, 0.9 * float(model.profile.r_t), 20)
    expected = moments(model, r, n_nodes=3200)["sigma_r2"]
    np.testing.assert_allclose(sigma_r(model, r) ** 2, expected, rtol=GOAL)
    views = [_component_view(model, k) for k in range(2)]
    reads = [dispersion_profile(v, SETTINGS).read(r)[1][0] ** 2 for v in views]
    np.testing.assert_allclose(
        sigma_t(model, r) ** 2, mixture(np.array(WEIGHTS), np.array(reads)), rtol=1e-12
    )


def test_the_aperture_losvd_of_a_superposition_integrates_to_one(tabled):
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
    from pianistpy.observables import losvd

    _, table, _ = tabled
    half = np.linspace(0.0, 1.5, 3001)  # past the escape speed at R 0.1
    v = np.concatenate([-half[:0:-1], half])
    d = losvd(table, 0.1, v)
    assert np.all(d >= 0.0)
    assert np.trapezoid(d, v) == pytest.approx(1.0, rel=1e-4)
