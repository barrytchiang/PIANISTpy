"""Tests of a DF from a function: its tabulation on the inversion's grid, its
certificate and its refusal."""

import re

import numpy as np
import pytest

from pianistpy import (
    Anisotropy,
    Component,
    DoublePowerLaw,
    InputError,
    Truncation,
    read,
)
from pianistpy.df import DistributionFunction, compute_df, truncate
from pianistpy.df.analytic import plummer_df
from pianistpy.df.report import NO_SLOPE_COLUMN

PLUMMER = Component(DoublePowerLaw.plummer(), Anisotropy())


def plummer(k):
    return plummer_df(k)[0]


@pytest.fixture(scope="module")
def from_plummer():
    """The isotropic Plummer model's closed-form DF through `from_function`,
    with the energies the function was called on."""
    called = []

    def recorded(k, depth):
        called.append((np.array(k), np.array(depth)))
        return plummer(k)

    recorded.__name__ = "plummer"
    return DistributionFunction.from_function(PLUMMER, recorded), called


def test_the_tabulation_of_the_packages_own_df_is_its_table():
    # f = the package's own DF of a component read at its grid's nodes and at
    # its nodes below the written floor: the function's table has the
    # inversion's log_df at every node they share, and the same nodes below
    component = Component(DoublePowerLaw.hernquist(), Anisotropy())
    own = compute_df(component)
    kb = own.table.k_below[0]
    # the rows first: at an energy both hold, a function has the rows' value
    q = np.concatenate([own.rows.k[0], kb])
    f = np.concatenate([own.rows.f[0], 10.0 ** own.table.log_df_below[0]])
    order = np.argsort(q, kind="stable")
    q, f = q[order], f[order]

    def read_at_the_nodes(k, depth):
        i = np.searchsorted(q, k)
        assert np.array_equal(q[i], k)
        return f[i]

    df = DistributionFunction.from_function(component, read_at_the_nodes)
    a, b = own.table, df.table
    support_a = slice(a.i_first[0], a.i_last[0] + 1)
    support_b = slice(b.i_first[0], b.i_last[0] + 1)
    shared, ia, ib = np.intersect1d(
        a.k[0, support_a], b.k[0, support_b], return_indices=True
    )
    assert shared.size > 0.9 * (a.i_last[0] - a.i_first[0] + 1)
    np.testing.assert_array_equal(
        a.log_df[0, support_a][ia], b.log_df[0, support_b][ib]
    )
    assert kb.size > 0
    np.testing.assert_array_equal(b.k_below, a.k_below)
    np.testing.assert_allclose(
        b.log_df_below[0, :-1], a.log_df_below[0, :-1], rtol=1e-15, atol=0
    )
    assert df.source == "function" and df.function_name == "read_at_the_nodes"
    assert df.inversion is None and df.rows is None
    assert df.identity != own.identity


def test_the_plummer_closed_form_certifies_at_the_goal(from_plummer, tmp_path):
    # the forms certify at the goal; the slope column is not judged, and the
    # report says why
    df, _ = from_plummer
    assert df.validation.tier == "goal"
    assert df.validation.keys == compute_df(PLUMMER).validation.keys
    slope = df.validation.checks["slope_column"]
    assert slope.reason == NO_SLOPE_COLUMN and slope.tier == "not judged"
    assert all(
        df.validation.checks[k].reason is None
        and df.validation.checks[k].tier == "goal"
        for k in df.validation.keys
        if k != "slope_column"
    )
    assert df.is_physical
    path = tmp_path / "plummer_df.h5"
    df.write(path)
    back = read(path)
    assert back.identity == df.identity
    assert (back.source, back.function_name) == ("function", "plummer")
    assert back.validation.checks["slope_column"].reason == NO_SLOPE_COLUMN
    np.testing.assert_array_equal(back.table.log_df, df.table.log_df)


def test_truncation_refuses_a_df_from_a_function(from_plummer):
    df, _ = from_plummer
    for truncation in (None, Truncation(k_t=0.05)):
        with pytest.raises(InputError, match="comes from a function \\(plummer\\)"):
            truncate(df, truncation)


def test_the_function_reads_the_grids_energies_and_depths(from_plummer):
    # called once, on the grid's energies and their depths P_0 - K
    _, called = from_plummer
    assert len(called) == 1
    k, depth = called[0]
    assert np.all(np.diff(k) > 0.0) and k.shape == depth.shape
    assert depth.tobytes() == (1.0 - k).tobytes()
    assert depth[-1] > 0.0


def test_a_function_not_finite_is_refused_where(from_plummer):
    _, called = from_plummer
    q = called[-1][0]
    first = float(q[q > 0.5][0])

    def holed(k, depth):
        return np.where(k > 0.5, np.nan, plummer(k))

    with pytest.raises(
        InputError, match=re.escape(f"not finite at K = {first:.17g}, the first")
    ):
        DistributionFunction.from_function(PLUMMER, holed)
