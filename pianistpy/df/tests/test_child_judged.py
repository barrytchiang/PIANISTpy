"""Tests of the judgment of a model on a tabulated pair.

A Method II child's model is judged by the rule `truncate` follows: a check
beyond the hard limit refuses it unless the settings ask for it anyway
(``write_anyway``), the report carries the tier either way, and a model within
the goal is returned as it is.
"""

import warnings

import numpy as np
import pytest

from pianistpy import Anisotropy, DoublePowerLaw, Settings, Truncation
from pianistpy.df import core
from pianistpy.df.inputs import Component
from pianistpy.exceptions import AccuracyError, AccuracyWarning

CORED = DoublePowerLaw(1.0, 4.0, 0.0)


def _parent(k_t):
    return core.truncate(
        core.compute_df((Component(CORED, Anisotropy()),), Settings()),
        Truncation(k_t=k_t),
    )


def _child(parent, anisotropy, settings):
    df = core.compute_df((Component(parent.profile, anisotropy),), settings)
    return core.truncate(df, None)


@pytest.mark.parametrize(
    ("k_t", "anisotropy"),
    [
        (0.5, Anisotropy(beta_c=-0.5)),
        (0.2, Anisotropy(beta_c=-0.5)),
        (0.5, Anisotropy(beta_c=-0.25, r_a=10.0, kind="I")),
    ],
)
def test_a_child_beyond_the_hard_limit_is_refused_by_name(k_t, anisotropy):
    parent = _parent(k_t)
    with pytest.raises(
        AccuracyError,
        match="does not meet its own checks: the top_form misses it by .*, "
        "beyond the hard limit 0.01",
    ):
        _child(parent, anisotropy, Settings())
    with pytest.warns(AccuracyWarning, match="returned on request, write_anyway"):
        model = _child(parent, anisotropy, Settings(write_anyway=True))
    assert model.validation.tier == "refused"
    assert model.validation.checks["top_form"].level == 3


def test_a_child_within_the_goal_is_returned_as_it_is():
    # at k_t 0.002 the edge approach, read row by row, puts the child at the
    # ceiling (1.3e-6 at r 1012); at 0.01 it is within the goal
    parent = _parent(0.01)
    anisotropy = Anisotropy(r_a=10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", AccuracyWarning)
        model = _child(parent, anisotropy, Settings())
    asked = _child(parent, anisotropy, Settings(write_anyway=True))
    assert model.validation.tier == "goal"
    assert model.validation.to_dict() == asked.validation.to_dict()
    for name in ("k", "log_df", "log_df_slope"):
        a = np.asarray(getattr(model.table, name))
        b = np.asarray(getattr(asked.table, name))
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
    for name in ("r", "p_trunc", "rho_trunc", "mass_trunc"):
        a = np.asarray(getattr(model.profile, name))
        b = np.asarray(getattr(asked.profile, name))
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
