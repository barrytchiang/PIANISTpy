"""Tests of the DF results and their files."""

import dataclasses
import math

import numpy as np
import pytest

from pianistpy import (
    Anisotropy,
    Component,
    DoublePowerLaw,
    ResultError,
    Settings,
    Truncation,
    read,
)
from pianistpy.df import DistributionFunction, Model
from pianistpy.df.core import model_identity
from pianistpy.df.stages import (
    AsymptoticForm,
    DfTable,
    EnergyGrid,
    InversionRecord,
    TruncatedProfile,
)
from pianistpy.validation import Check, ValidationReport

N = 9
COMPONENTS = (
    Component(DoublePowerLaw.nfw(), Anisotropy(beta_c=-0.25, r_a=10.0), name="halo"),
    Component(DoublePowerLaw.hernquist(), mass=0.05, name="stars"),
)
SETTINGS = Settings()


def _table():
    log_df = np.linspace(-3.0, 2.0, 2 * N).reshape(2, N)
    log_df[1, 0] = np.nan
    log_df[1, 1] = -np.inf
    return DfTable(
        k=np.tile(np.linspace(1e-3, 0.99, N), (2, 1)),
        log_df=log_df,
        log_df_slope=np.ones((2, N)),
        i_first=(0, 2),
        i_last=(N - 1, N - 1),
        slopes_exact=(True, False),
    )


def _form():
    return AsymptoticForm(
        p_0=1.0,
        central_label=("regular", "critical_analytic"),
        central_exponent=(-2.5, 0.0),
        central_amplitude=(0.17, 0.0),
        tail_exponent=(0.75, 0.0),
        tail_amplitude=(0.3, 0.0),
        tail_log_exponent=(1.0, 0.0),
        central_form=("power", "analytic"),
        central_log_coefficient=(0.0, 0.0),
        central_next_form=("log_power", "none"),
        central_next_exponent=(1.0, 0.0),
        central_next_coefficient=(0.2, 0.0),
        tail_log_scale=(2.0, 1.0),
        below_floor=("tail", "taylor"),
        below_floor_coefficients=((), (2.0, -3.0, 1e-3)),
    )


def _inversion():
    return InversionRecord(
        form=("integrated_by_parts", "derivative_outside"),
        order_j=(1, 2),
        order_n=(0.5, -0.25),
        floor_m=(1, 1),
        floor_n=(0.5, 0.75),
        k_interior=(np.nan, 0.3),
        interior_exponent=(np.nan, 1.5),
        outer_power=(np.nan, 0.5),
        outer_end=("bounded", "unbounded"),
        high_k_override_k=(0.95, 1.0),
        high_k_override_r=(1e-9, 0.0),
        high_k_override_rows=(1, 0),
        high_k_override_rule=("G7 data top", "none"),
        low_k_override_k=(0.0, 0.0),
        low_k_override_r=(23.4, np.inf),
        method2_bands=(
            {"patch_anchor_K": 0.95, "floor_term_exact": True, "interior_negatives": 0},
        ),
        critical_band=(
            {"critical_band": "closed-form series band", "critical_band_u": np.nan},
        ),
    )


def _report():
    return ValidationReport(
        ("density", "potential"),
        {"density": Check(3e-7, where=1.5), "potential": Check(2e-8)},
    )


def _same(a, b):
    for item in dataclasses.fields(a):
        x, y = getattr(a, item.name), getattr(b, item.name)
        if isinstance(x, np.ndarray):
            assert np.array_equal(x, y, equal_nan=True) and x.dtype == y.dtype
        elif isinstance(x, float) and math.isnan(x):
            assert math.isnan(y), item.name
        else:
            assert x == y, item.name


def _same_inversion(a, b):
    """Equal records: NaN equals NaN, in the tuples and in the band records."""
    for item in dataclasses.fields(a):
        x, y = getattr(a, item.name), getattr(b, item.name)
        assert len(x) == len(y), item.name
        for u, v in zip(x, y, strict=True):
            if isinstance(u, dict):
                assert set(u) == set(v), item.name
                for key in u:
                    assert u[key] == v[key] or (
                        isinstance(u[key], float)
                        and math.isnan(u[key])
                        and math.isnan(v[key])
                    ), (item.name, key)
            elif isinstance(u, float) and math.isnan(u):
                assert math.isnan(v), item.name
            else:
                assert u == v, item.name


def test_distribution_function_round_trip(tmp_path):
    df = DistributionFunction(
        components=COMPONENTS,
        settings=SETTINGS,
        identity=model_identity(COMPONENTS, None, SETTINGS),
        grid=EnergyGrid(
            k=np.tile(np.linspace(1e-3, 0.99, N), (2, 1)),
            r_of_k=np.tile(np.logspace(3, -3, N), (2, 1)),
            k_floor=(1e-3, 1e-3),
            k_ceiling=(0.99, 0.99),
        ),
        table=_table(),
        form=_form(),
        k_t_min=0.278,
        validation=_report(),
        timing={"energy_grid": 0.01, "invert_rows": 1.5},
        inversion=_inversion(),
    )
    path = tmp_path / "df.h5"
    df.write(path)
    back = read(path)
    assert isinstance(back, DistributionFunction)
    assert back.components == df.components
    assert back.settings == df.settings
    assert (back.identity, back.k_t_min, back.timing) == (
        df.identity,
        df.k_t_min,
        df.timing,
    )
    assert not back.is_physical
    for name in ("grid", "table", "form"):
        _same(getattr(df, name), getattr(back, name))
    _same_inversion(df.inversion, back.inversion)
    assert back.validation.to_dict()["checks"]["density"]["error"] == 3e-7
    assert back.validation.checks["potential"].where is None
    with pytest.raises(ResultError):
        df.write(path)


def test_model_round_trip(tmp_path):
    truncation = Truncation(mass_t=2.0, concentration=10.0)
    model = Model(
        components=COMPONENTS,
        truncation=truncation,
        settings=SETTINGS,
        identity=model_identity(COMPONENTS, truncation, SETTINGS),
        parent=model_identity(COMPONENTS, None, SETTINGS),
        table=_table(),
        form=_form(),
        profile=TruncatedProfile(
            r=np.logspace(-3, 1, N),
            p_trunc=np.linspace(0.9, 0.0, N),
            rho_trunc=np.hstack([np.ones((2, N - 1)), np.zeros((2, 1))]),
            mass_trunc=np.linspace(0.0, 2.0, N),
            k_t=0.1,
            r_t=10.0,
            mass_t=2.0,
            spatially_finite=True,
            source=COMPONENTS[0].profile,
            source_anisotropy=COMPONENTS[0].anisotropy,
            model=model_identity(COMPONENTS, truncation, SETTINGS),
            edge_exponent=(2.5, 2.5),
            central_deficit=((0.006, 0.0096), (0.0, 0.0)),
        ),
        validation=_report(),
        timing={"solve_poisson": 2.0},
        r_trust={},
        inversion=_inversion(),
    )
    path = tmp_path / "model.h5"
    model.write(path)
    back = read(path)
    assert isinstance(back, Model)
    assert back.components == model.components
    assert back.truncation == truncation
    assert (back.identity, back.parent) == (model.identity, model.parent)
    assert back.spatially_finite
    for name in ("table", "form", "profile"):
        _same(getattr(model, name), getattr(back, name))
    _same_inversion(model.inversion, back.inversion)
    # the table's end is its last row
    assert (back.profile.r_table_max, back.profile.mass_at_r_table_max) == (10.0, 2.0)
    assert back.r_trust == {}  # spatially finite: valid to its edge
    with pytest.raises(ResultError):
        model.write(path)


def test_a_form_says_where_its_df_diverges():
    form = _form()
    # a tail A K^-0.75 diverges as K -> 0; a Taylor rule vanishes there
    assert form.exponent_at_zero(0) == -0.75
    assert form.exponent_at_zero(1) is None
    assert form.diverges_at_zero == (True, False)
    edge = dataclasses.replace(
        form,
        below_floor=("edge_law", "edge_law"),
        below_floor_coefficients=((-0.5, 1.0, 0.0), (0.25, 1.0, 0.0)),
    )
    assert edge.diverges_at_zero == (True, False)
    assert edge.exponent_at_zero(1) == 0.25


def test_a_profile_carries_its_models_df_and_a_child_needs_it(tmp_path):
    # the model's DF table and Taylor edge ride with its profile through the file
    one = COMPONENTS[:1]
    truncation = Truncation(k_t=0.1)
    k = np.linspace(1e-9, 0.85, N)[None, :]
    profile = TruncatedProfile(
        r=np.logspace(-3, 1, N),
        p_trunc=np.linspace(0.9, 0.0, N),
        rho_trunc=np.hstack([np.ones((1, N - 1)), np.zeros((1, 1))]),
        mass_trunc=np.linspace(0.0, 2.0, N),
        k_t=0.1,
        r_t=10.0,
        mass_t=2.0,
        spatially_finite=True,
        source=one[0].profile,
        source_anisotropy=Anisotropy(),
        model=model_identity(one, truncation, SETTINGS),
        edge_exponent=(2.5,),
        df_k=k,
        df_log_f=np.log10(k),
        df_slope=np.full(k.shape, 1.0 / np.log(10.0)),
        df_taylor=((1.0, 0.0, 1e-8),),
    )
    model = Model(
        components=one,
        truncation=truncation,
        settings=SETTINGS,
        identity=profile.model,
        parent=model_identity(one, None, SETTINGS),
        table=_table(),
        form=_form(),
        profile=profile,
        validation=_report(),
        timing={},
        r_trust={},
        inversion=_inversion(),
    )
    path = tmp_path / "model.h5"
    model.write(path)
    back = read(path).profile
    for name in ("df_k", "df_log_f", "df_slope"):
        assert np.array_equal(getattr(back, name), getattr(profile, name))
    assert back.df_taylor == profile.df_taylor
    # a profile that does not carry its model's DF refuses a child, by name
    bare = dataclasses.replace(
        profile,
        df_k=np.zeros((0, 0)),
        df_log_f=np.zeros((0, 0)),
        df_slope=np.zeros((0, 0)),
    )
    from pianistpy import InputError, compute_df

    with pytest.raises(InputError, match="carries no DF of its model"):
        compute_df(Component(bare, Anisotropy()))
