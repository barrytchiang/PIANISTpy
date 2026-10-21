"""Tests of `compute_df` against closed-form DFs, and of the DF ceiling.

The DF ceiling is the largest DF value the table holds, :math:`10^{300}`.
"""

import numpy as np
import pytest

from pianistpy import Anisotropy, Component, DoublePowerLaw, read
from pianistpy.df import compute_df
from pianistpy.df.analytic import hernquist_df, plummer_df


@pytest.mark.parametrize(
    ("profile", "df_function"),
    [
        (DoublePowerLaw.hernquist(), hernquist_df),
        (DoublePowerLaw.plummer(), plummer_df),
    ],
    ids=["hernquist", "plummer"],
)
def test_compute_df_meets_the_closed_form(profile, df_function):
    df = compute_df(Component(profile, Anisotropy(), name="analytic"))
    k = np.asarray(df.table.k[0])
    want, _ = df_function(k)
    got = 10.0 ** np.asarray(df.table.log_df[0])
    error = np.abs(got / want - 1.0)
    print(f"max relative error {error.max():.3e} at K = {k[np.argmax(error)]:.6g}")
    assert error.max() <= 1e-6


def test_the_rows_as_made_are_kept_and_written(tmp_path):
    df = compute_df(Component(DoublePowerLaw.plummer(), Anisotropy(), name="plummer"))
    rows = df.rows
    assert rows is not None
    assert rows.k.shape == rows.f.shape == rows.ddf.shape
    assert rows.k.shape[0] == 1 and len(rows.data_top) == 1
    # The table holds the rows up to the anchor energy, so every tabulated
    # energy is the energy of a row.
    assert np.all(np.isin(np.asarray(df.table.k[0]), rows.k[0]))
    path = tmp_path / "plummer_df.h5"
    df.write(path)
    back = read(path)
    for name in ("k", "f", "ddf"):
        assert (
            np.asarray(getattr(back.rows, name)).tobytes()
            == np.asarray(getattr(rows, name)).tobytes()
        )
    assert back.rows.data_top == rows.data_top


def test_the_grid_stops_where_the_closed_form_reaches_the_ceiling():
    # The isotropic model with inner slope gamma = 1.99 and outer slope
    # beta = 4 has a central DF that grows as A u^-200.5, with u = P_0 - K
    # the depth below the central potential. The DF exceeds the ceiling 1e300
    # near the center. The table's top node is the last node at which the
    # center's closed form stays at or below the ceiling. Above the top node
    # the closed form describes the DF, and at the top node the table's DF
    # matches the closed form.
    from pianistpy.df.central import LOG10_DF_CEILING, ceiling_limit, center_form

    component = Component(DoublePowerLaw(1.0, 4.0, 1.99), Anisotropy(), name="iso")
    df = compute_df(component)
    profile, anisotropy = component.unit_profile, component.anisotropy
    u_cut = ceiling_limit(profile, anisotropy)
    k = np.asarray(df.table.k[0])
    log_df = np.asarray(df.table.log_df[0])
    top = int(df.table.i_last[0])
    assert 1.0 - k[top] > u_cut
    assert log_df[top] <= LOG10_DF_CEILING
    form = center_form(profile, anisotropy)
    closed = np.log10(form["amplitude"]) + form["kappa"] * np.log10(1.0 - k[top])
    assert log_df[top] == pytest.approx(closed, abs=1e-2)
    assert df.validation.checks["validity"].error == 0.0


def test_a_type_ii_df_above_the_ceiling_is_refused_with_cause():
    # The OM Type II model with R_a = 10, inner slope gamma = 1.99 and outer
    # slope beta = 3 has rows above the ceiling 1e300. A Type II table carries
    # no closed form above its rows, so no closed form covers the energies
    # above a cut. `compute_df` raises AccuracyError, whose message states the
    # ceiling, the first row above the ceiling and that row's energy.
    from pianistpy import AccuracyError

    component = Component(
        DoublePowerLaw(1.0, 3.0, 1.99), Anisotropy(r_a=10.0, kind="II"), name="omii"
    )
    with pytest.raises(AccuracyError, match=r"exceed the DF's ceiling 10\^300"):
        compute_df(component)


def test_a_row_above_the_ceiling_no_closed_form_covers_is_refused():
    # Where no closed form of the center covers the energies above a cut, a
    # row above the ceiling 1e300 raises AccuracyError, whose message states the
    # ceiling, the first row above the ceiling and that row's energy.
    from pianistpy import AccuracyError
    from pianistpy.df.invert import _table

    q = np.linspace(0.1, 0.9, 9)
    f = np.full(q.size, 1.0)
    f[-1] = 1e301
    assert np.isfinite(_table(q, f, None, None, None).log_df).all()
    with pytest.raises(AccuracyError, match=r"ceiling 10\^300, the first row 8 at K"):
        _table(q, f, None, None, None, covered=False)
