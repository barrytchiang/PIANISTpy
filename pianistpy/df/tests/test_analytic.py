"""Tests of the closed-form DFs of the isotropic Hernquist and Plummer models.

The tests hold `hernquist_df` and `plummer_df` to their densities, to the
textbook Hernquist DF in 40-digit arithmetic and to their own derivatives, and
they check the `Model` that `analytic_model` builds from each closed form.
"""

import dataclasses
import math

import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import quad

from pianistpy import DoublePowerLaw, InputError, Model, read
from pianistpy.df.analytic import (
    HERNQUIST_CENTRAL_AMPLITUDE,
    ISOTROPIC_DENSITY_PREFACTOR,
    PLUMMER_AMPLITUDE,
    analytic_model,
    hernquist_df,
    plummer_df,
)

CASES = [
    (hernquist_df, DoublePowerLaw.hernquist(), lambda p: 1.0 / p - 1.0),
    (plummer_df, DoublePowerLaw.plummer(), lambda p: math.sqrt(1.0 / p**2 - 1.0)),
]


@pytest.mark.parametrize("df_function, profile, r_of_p", CASES)
def test_the_df_reproduces_the_density(df_function, profile, r_of_p):
    for p in (1e-6, 1e-3, 0.1, 0.5, 0.9, 0.999):
        integral, _ = quad(
            lambda k, p=p: float(df_function(k)[0]),
            0.0,
            p,
            weight="alg",
            wvar=(0.0, 0.5),
            epsabs=0.0,
            epsrel=1e-12,
            limit=200,
        )
        rho = float(profile.rho(r_of_p(p)))
        assert ISOTROPIC_DENSITY_PREFACTOR * integral == pytest.approx(rho, rel=2e-10)


def test_hernquist_against_the_textbook_form_in_high_precision():
    mp.mp.dps = 40

    def textbook(k):
        q = mp.sqrt(k)
        bracket = 3 * mp.asin(q) + q * mp.sqrt(1 - q * q) * (1 - 2 * q * q) * (
            8 * q**4 - 8 * q * q - 3
        )
        return bracket / (2 * mp.sqrt(2) * mp.pi**2 * (1 - q * q) ** mp.mpf(2.5))

    for k in (1e-12, 1e-8, 1e-4, 0.03, 0.5, 0.97, 1.0 - 1e-6):
        df, ddf_dk = hernquist_df(k)
        exact = textbook(mp.mpf(k))
        slope = mp.diff(textbook, mp.mpf(k))
        assert float(abs(df / exact - 1)) < 5e-14
        assert float(abs(ddf_dk / slope - 1)) < 5e-13


@pytest.mark.parametrize("df_function", [hernquist_df, plummer_df])
def test_the_derivative_is_the_derivative(df_function):
    k = np.array([1e-6, 0.01, 0.3, 0.7, 0.99])
    h = 1e-6 * k
    upper, _ = df_function(k + h)
    lower, _ = df_function(k - h)
    _, ddf_dk = df_function(k)
    assert np.allclose((upper - lower) / (2.0 * h), ddf_dk, rtol=1e-6)


def test_the_ends_of_the_closed_forms():
    df, _ = hernquist_df(1.0 - 1e-9)
    assert float(df) * (1e-9) ** 2.5 == pytest.approx(
        HERNQUIST_CENTRAL_AMPLITUDE, rel=1e-7
    )
    df, _ = hernquist_df(1e-10)
    tail_amplitude = analytic_model("hernquist", 11).form.tail_amplitude[0]
    assert float(df) / (1e-10) ** 2.5 == pytest.approx(tail_amplitude, rel=1e-9)
    assert tail_amplitude == pytest.approx(64.0 / (5.0 * math.sqrt(2.0) * math.pi**2))
    assert PLUMMER_AMPLITUDE == pytest.approx(
        192.0 / (7.0 * math.sqrt(2.0) * math.pi**2)
    )


@pytest.mark.parametrize("name", ["hernquist", "plummer"])
def test_the_analytic_model_is_a_model_and_survives_its_file(name, tmp_path):
    model = analytic_model(name, n_nodes=201)
    assert isinstance(model, Model)
    assert not model.spatially_finite
    assert model.truncation is None
    assert model.validation.tier == "silent"
    assert model.table.log_df.shape == (1, 201)
    assert np.all(np.diff(model.table.k[0]) > 0.0)
    assert np.all(np.diff(model.profile.p_trunc) < 0.0)
    assert np.all(np.isfinite(model.table.log_df))
    assert model.identity != analytic_model(name, n_nodes=203).identity
    path = tmp_path / "model.h5"
    model.write(path)
    back = read(path)
    assert back.identity == model.identity
    assert back.components == model.components
    for stage in ("table", "profile"):
        for key, value in vars(getattr(model, stage)).items():
            other = getattr(getattr(back, stage), key)
            if isinstance(value, np.ndarray):
                assert np.array_equal(other, value), key
            elif isinstance(value, float) and math.isnan(value):
                assert math.isnan(other), key
            else:
                assert other == value, key
    assert back.form.tail_amplitude == model.form.tail_amplitude
    assert back.r_trust == model.r_trust == {"silent": math.inf, "recorded": math.inf}
    # A model whose r_trust holds finite radii reads those radii back from its
    # file unchanged.
    radii = dataclasses.replace(model, r_trust={"silent": 7.8e5, "recorded": 7.8e7})
    radii.write(tmp_path / "radii.h5")
    assert read(tmp_path / "radii.h5").r_trust == {"silent": 7.8e5, "recorded": 7.8e7}
    # A file whose r_trust is keyed by "goal" and "ceiling" reads back with the
    # keys "silent" and "recorded".
    former = dataclasses.replace(model, r_trust={"goal": 7.8e5, "ceiling": 7.8e7})
    former.write(tmp_path / "former.h5")
    assert read(tmp_path / "former.h5").r_trust == {"silent": 7.8e5, "recorded": 7.8e7}
    assert back.form.central_label == model.form.central_label


def test_an_unknown_name_is_refused():
    with pytest.raises(InputError):
        analytic_model("jaffe")
    with pytest.raises(InputError):
        analytic_model("plummer", n_nodes=3)
