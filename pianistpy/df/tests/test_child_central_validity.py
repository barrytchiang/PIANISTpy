r"""Tests of the central validity check on a Method II child of a tabulated pair.

A Method II child whose central anisotropy :math:`\beta_{\rm c}(0)` exceeds
half the parent's inner slope :math:`\gamma` (:math:`\gamma < 2\beta_{\rm c}(0)`)
has a DF that is negative at the center. `compute_df` raises
UnphysicalModelError with a message that starts with "CENTRAL VALIDITY" for
such a child, at the check of its inputs and before any stage runs, as
`truncate` does for a Method I model.
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
