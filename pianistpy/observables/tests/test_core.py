"""Tests of the public observables `sigma_los`, `sigma_los_aperture`, `losvd`,
`sigma_r` and `LosTable` on the isotropic Plummer model.

The tests compare a Model's values with its LosTable's values, with the Plummer
model's closed forms in internal units (G = M_0 = r_0 = 1) and across the
internal and physical unit systems.
"""

import dataclasses
import math
import warnings

import numpy as np
import pytest

from pianistpy import AccuracyWarning, InputError, Settings
from pianistpy.df.analytic import analytic_model
from pianistpy.observables import Aperture, LosTable, sigma_los, sigma_los_aperture
from pianistpy.observables._ellipse_rules import EllipseRules, layer_counts
from pianistpy.observables.stages import validate_table
from pianistpy.validation import TIER_BOUNDS, ValidationReport

pytestmark = pytest.mark.slow

# The observables' radial and node counts of SETTINGS are smaller than the
# counts of the default settings, as in the tests of observables.stages.
SETTINGS = Settings(
    observables={"n_radii": 400, "n_jeans_nodes": 200, "n_projection_nodes": 200}
)
# The physical system's scales: M_0 = 1e9 solar masses is the mass inside
# CONCENTRATION scale radii, and r_0 = 500 parsecs is the scale radius.
MASS_0, R_0, CONCENTRATION = 1e9, 500.0, 10.0


@pytest.fixture(scope="module")
def plummer():
    """Return the isotropic Plummer model with SETTINGS, and its LosTable."""
    model = dataclasses.replace(analytic_model("plummer"), settings=SETTINGS)
    return model, LosTable(model)


def test_sigma_los_from_a_model_and_from_its_table(plummer):
    # The Plummer model's closed form in internal units is
    # sigma_los^2 = (3 pi / 64) (1 + R^2)^(-1/2), with R the projected radius.
    model, table = plummer
    r_2d = np.logspace(-3, 2, 11)
    exact = np.sqrt(3.0 * math.pi / 64.0 / np.sqrt(1.0 + r_2d**2))
    from_model, from_table = sigma_los(model, r_2d), sigma_los(table, r_2d)
    assert from_model.shape == r_2d.shape
    np.testing.assert_allclose(from_model, exact, rtol=TIER_BOUNDS[0])
    np.testing.assert_array_equal(from_table, from_model)


def test_sigma_los_aperture_by_its_two_routes(plummer):
    # The Plummer model's mean square line-of-sight velocity over a centered
    # circle of radius A is (pi / 32) [1 - (1 + A^2)^(-3/2)] / [1 - (1 + A^2)^-1].
    # The Model and its LosTable each give sigma_los_aperture within 1e-6 of the
    # closed form, and the LosTable's value is within 1e-6 of the Model's value.
    model, table = plummer
    for a in (0.5, 3.0):
        aperture = Aperture.circle(a)
        exact = math.sqrt(
            math.pi / 32.0 * (1.0 - (1.0 + a * a) ** -1.5) / (1.0 - 1.0 / (1.0 + a * a))
        )
        from_model = sigma_los_aperture(model, aperture)
        from_table = sigma_los_aperture(table, aperture)
        assert isinstance(from_model, float) and isinstance(from_table, float)
        assert from_model == pytest.approx(exact, rel=TIER_BOUNDS[0])
        assert from_table == pytest.approx(from_model, rel=TIER_BOUNDS[0])


def test_the_model_unit_system_is_read_and_returned(plummer):
    model, table = plummer
    physical = model.to("physical", concentration=CONCENTRATION, mass_0=MASS_0, r_0=R_0)
    factors = physical.units.factors(physical.components[0].unit_profile)
    r_2d = np.array([0.1, 1.0, 10.0])
    value = sigma_los(physical, r_2d * R_0)
    assert str(value.unit) == "km / s"
    np.testing.assert_allclose(
        value.value, sigma_los(model, r_2d) * factors.physical_v_kms, rtol=1e-14
    )
    # On the LosTable in the physical system, an aperture of radius r_0 given in
    # parsecs gives the internal-units sigma_los_aperture times the factor from
    # internal velocity units to km/s.
    in_parsecs = sigma_los_aperture(
        table.to("physical", concentration=CONCENTRATION, mass_0=MASS_0, r_0=R_0),
        Aperture.circle(1.0 * R_0),
    )
    assert in_parsecs.value == pytest.approx(
        sigma_los_aperture(table, Aperture.circle(1.0)) * factors.physical_v_kms,
        rel=1e-14,
    )


def test_lengths_given_as_quantities_are_the_floats_in_parsecs(plummer):
    import astropy.units as u

    model, _ = plummer
    physical = model.to("physical", concentration=CONCENTRATION, mass_0=MASS_0, r_0=R_0)
    r_2d = np.array([0.1, 1.0, 10.0]) * R_0
    np.testing.assert_array_equal(
        sigma_los(physical, r_2d * u.pc).value, sigma_los(physical, r_2d).value
    )
    np.testing.assert_array_equal(
        sigma_los(physical, (r_2d * u.pc).to(u.kpc)).value,
        sigma_los(physical, r_2d).value,
    )


def test_an_aperture_rule_finer_than_the_default_agrees(plummer):
    # aperture_nodes_per_decade = 200 exceeds the panel's minimum node count, so
    # the aperture rule is built from aperture_nodes_per_decade. The finer rule's
    # sigma_los_aperture is within 1e-6 of the default rule's value.
    model, _ = plummer
    finer = dataclasses.replace(
        model,
        settings=Settings(
            observables={
                "n_radii": 400,
                "n_jeans_nodes": 200,
                "n_projection_nodes": 200,
                "aperture_nodes_per_decade": 200,
            }
        ),
    )
    for aperture in (Aperture.circle(0.5), Aperture.circle(3.0)):
        assert sigma_los_aperture(finer, aperture) == pytest.approx(
            sigma_los_aperture(model, aperture), rel=TIER_BOUNDS[0]
        )


def test_the_numpy_rows_route_gives_the_kernels_table(plummer):
    # rows_route = "numpy" selects the NumPy reference route of the rows' angle
    # rule. The losvd of the NumPy route's LosTable agrees with the losvd of the
    # JAX kernel route's LosTable to rtol = 1e-6 and atol = 1e-6 times the peak.
    from pianistpy.observables import losvd

    model, table = plummer
    numpy_route = dataclasses.replace(
        model,
        settings=Settings(
            observables={
                "n_radii": 400,
                "n_jeans_nodes": 200,
                "n_projection_nodes": 200,
                "rows_route": "numpy",
            }
        ),
    )
    reference = LosTable(numpy_route)
    v = np.linspace(-1.0, 1.0, 41)
    for r_2d in (0.1, 1.0, 5.0):
        want = losvd(table, r_2d, v)
        np.testing.assert_allclose(
            losvd(reference, r_2d, v),
            want,
            rtol=TIER_BOUNDS[0],
            atol=TIER_BOUNDS[0] * float(np.max(want)),
        )


def test_a_model_of_two_components_is_refused(plummer):
    model, _ = plummer
    two = dataclasses.replace(model, components=model.components * 2)
    with pytest.raises(InputError, match="one component"):
        sigma_los(two, np.array([1.0]))


def test_the_projected_tables_traced_reader_gives_the_same_numbers(plummer):
    import jax
    import jax.numpy as jnp

    _, table = plummer
    projected = table.projected
    r_2d = np.geomspace(projected.r_2d[0], projected.r_2d[-1], 37)
    surface, sigma = projected.read(r_2d)
    traced = jax.jit(lambda r: projected.read_traced(r, jnp))(jnp.asarray(r_2d))
    # The surface density is the exponential of a cubic in ln Sigma. A rounding
    # error in ln Sigma is a relative error in Sigma, and the rounding error of
    # ln Sigma grows with |ln Sigma| at the last rows, so the surface density
    # is compared at rtol = 1e-12 and the squared dispersion at rtol = 1e-13.
    np.testing.assert_allclose(np.asarray(traced[0]), surface, rtol=1e-12)
    np.testing.assert_allclose(np.asarray(traced[1]), sigma**2, rtol=1e-13)
    # read_traced gives a zero surface density and a zero squared dispersion
    # at twice the last row's projected radius.
    beyond = projected.read_traced(np.array([2.0 * projected.r_2d[-1]]))
    assert float(beyond[0][0, 0]) == 0.0 and float(beyond[1][0, 0]) == 0.0


def test_a_table_reports_the_floor_of_its_layers(plummer):
    # Above beta_c = 1/2, the layers toward L = 0 stop at the ellipse rule's
    # floor, which layer_counts gives for beta_c = 3/4. The table's report
    # carries the floor in its check "layer_floor", in the tier "silent".
    _, table = plummer
    floor = layer_counts(0.75, EllipseRules())[1]
    certificate = dict(table.table.certificate, layer_floor=floor)
    report = validate_table(
        dataclasses.replace(table.table, certificate=certificate), table.projected
    )
    assert report.checks["layer_floor"].error == floor
    assert report.checks["layer_floor"].tier == "silent"


def test_a_table_carries_its_report_and_its_models_tier(plummer, caplog):
    model, table = plummer
    assert table.validation.keys == (
        "norm",
        "second_moment",
        "unjudged",
        "t_converged",
        "certificate",
        "layer_floor",
    )
    assert table.validation.tier == "silent"
    # On an isotropic model the layers toward L = 0 converge, so the check
    # "layer_floor" holds zero.
    assert table.validation.checks["layer_floor"].error == 0.0
    # A model whose parents_level = 1 is in the tier "recorded" (error above
    # 1e-6 and at most 1e-4). The LosTable of the flagged model logs the tier at INFO,
    # raises no AccuracyWarning, and its own report is in the tier "recorded".
    flagged = dataclasses.replace(
        model,
        validation=ValidationReport(
            keys=model.validation.keys, checks=model.validation.checks, parents_level=1
        ),
    )
    with caplog.at_level("INFO", logger="pianistpy"), warnings.catch_warnings():
        warnings.simplefilter("error", AccuracyWarning)
        above = LosTable(flagged)
    assert above.validation.tier == "recorded"
    assert any(
        "LosTable" in r.getMessage()
        and "above 0.000001" in r.getMessage()
        and r.levelname == "INFO"
        for r in caplog.records
    )
    # sigma_los_aperture on the "recorded" LosTable logs the tier at INFO.
    # sigma_los_aperture on the "silent" LosTable logs no tier.
    caplog.clear()
    with caplog.at_level("INFO", logger="pianistpy"), warnings.catch_warnings():
        warnings.simplefilter("error", AccuracyWarning)
        sigma_los_aperture(table, Aperture.circle(1.0))
    assert not any("above 0.000001" in r.getMessage() for r in caplog.records)
    with caplog.at_level("INFO", logger="pianistpy"), warnings.catch_warnings():
        warnings.simplefilter("error", AccuracyWarning)
        sigma_los_aperture(above, Aperture.circle(1.0))
    assert any(
        "aperture distribution" in r.getMessage() and "'recorded'" in r.getMessage()
        for r in caplog.records
    )


def test_losvd_on_a_model_says_once_that_its_table_is_worth_keeping(
    plummer, caplog, monkeypatch
):
    # losvd and aperture_losvd build the LosTable of a Model at every call. The
    # first such call in the process logs one warning that suggests keeping the
    # LosTable, and the later calls log no warning. The patched LosTable raises
    # Built, so no table is built.
    from pianistpy.observables import aperture_losvd, core, losvd

    class Built(Exception):
        pass

    class Unbuilt:
        def __init__(self, model):
            raise Built

    model, _ = plummer
    monkeypatch.setitem(core._TABLE_HINT, "given", False)
    monkeypatch.setattr(core, "LosTable", Unbuilt)
    with caplog.at_level("WARNING", logger="pianistpy"):
        for _ in range(2):
            with pytest.raises(Built):
                losvd(model, 0.5, 0.1)
            with pytest.raises(Built):
                aperture_losvd(model, Aperture.circle(0.5), 0.1)
    said = [
        r.getMessage() for r in caplog.records if "pp.LosTable(model)" in r.getMessage()
    ]
    assert said == [
        "losvd was called on a Model, so the call builds the model's LosTable. A "
        "LosTable kept with pp.LosTable(model) and passed to later calls is "
        "built once."
    ]


def test_a_models_dispersion_profile_is_built_once_and_held(monkeypatch):
    # The dispersion profile (the intrinsic second moments) depends on the model
    # only. sigma_r and sigma_los build the dispersion profile once, at the first
    # call on a Model, and hold it on the Model. The held profile equals, bit for
    # bit, a dispersion profile built by a separate call of dispersion_profile.
    from pianistpy.observables import core, sigma_r
    from pianistpy.observables.stages import dispersion_profile

    model = dataclasses.replace(analytic_model("plummer"), settings=SETTINGS)
    built = []

    def counted(m, settings, tangents=()):
        built.append(m)
        return dispersion_profile(m, settings, tangents)

    monkeypatch.setattr(core, "dispersion_profile", counted)
    r = np.logspace(-2, 1, 7)
    first = sigma_r(model, r)
    second = sigma_r(model, r)
    sigma_los(model, r)
    assert len(built) == 1 and built[0] is model
    np.testing.assert_array_equal(second, first)
    held, alone = (
        model.__dict__["_dispersion_profile"],
        dispersion_profile(model, model.settings),
    )
    for name in ("r", "sigma_r", "sigma_t", "beta_sigma"):
        np.testing.assert_array_equal(getattr(held, name), getattr(alone, name))
    np.testing.assert_array_equal(held.edge_limit, alone.edge_limit)
