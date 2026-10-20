"""Tests of the central validity on a tabulated pair.

A Method II child whose central anisotropy the parent's inner slope cannot carry
(``gamma < 2 beta_c(0)``) is refused by that name at its input, before any stage
runs, as a Method I model is at `truncate`.
"""

import pytest

from pianistpy import Anisotropy, DoublePowerLaw, Settings, Truncation
from pianistpy.df import core
from pianistpy.df.inputs import Component
from pianistpy.df.tabulated import refusals
from pianistpy.exceptions import UnphysicalModelError


def test_the_input_check_names_the_central_validity():
    with pytest.raises(
        UnphysicalModelError, match="CENTRAL VALIDITY: gamma = 0.00 < 2 beta_c"
    ):
        refusals({}, Anisotropy(), 1.0, Anisotropy(beta_c=0.25), gamma=0.0)


@pytest.mark.parametrize(
    "anisotropy", [Anisotropy(), Anisotropy(r_a=10.0), Anisotropy(beta_c=-0.25)]
)
def test_a_central_anisotropy_the_slope_carries_passes(anisotropy):
    assert refusals({}, Anisotropy(), 1.0, anisotropy, gamma=0.0) is None


def test_a_cored_parent_refuses_a_radially_biased_child_by_name():
    cored = DoublePowerLaw(1.0, 4.0, 0.0)
    parent = core.truncate(
        core.compute_df((Component(cored, Anisotropy()),), Settings()),
        Truncation(k_t=0.5),
    )
    with pytest.raises(UnphysicalModelError, match="CENTRAL VALIDITY"):
        core.compute_df(
            (Component(parent.profile, Anisotropy(beta_c=0.25)),), Settings()
        )
