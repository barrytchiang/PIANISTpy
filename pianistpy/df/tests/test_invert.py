"""Tests of the inversion from inputs alone, against the closed-form DFs."""

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
    # the table is the rows up to the anchor: every tabulated energy is a row's
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
