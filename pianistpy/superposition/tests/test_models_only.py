"""Tests that every public function that mixes members accepts Models only.

A distribution function with a signed table (``is_physical`` False) is never a
Model. Passed to `superpose`, `moments` or `likelihood` in place of a Model,
the distribution function raises an error that names the reason, and is never
mixed.
"""

import numpy as np
import pytest

from pianistpy import InputError
from pianistpy.df import DistributionFunction
from pianistpy.df.core import model_identity
from pianistpy.df.stages import EnergyGrid
from pianistpy.df.tests.test_results import (
    COMPONENTS,
    SETTINGS,
    N,
    _form,
    _inversion,
    _report,
    _table,
)
from pianistpy.inference import Data, likelihood
from pianistpy.particles.tests.test_stages import _lowered_hernquist
from pianistpy.superposition import moments, superpose
from pianistpy.superposition.tests.test_core import _member


@pytest.fixture(scope="module")
def signed():
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
        timing={},
        inversion=_inversion(),
    )
    assert not df.is_physical
    return df


@pytest.fixture(scope="module")
def member():
    return _member(_lowered_hernquist(), 0.0)


def test_superpose_refuses_a_distribution_function(signed, member):
    with pytest.raises(InputError, match="not a Model"):
        superpose({member: 0.5, signed: 0.5})


def test_moments_refuses_a_distribution_function(signed):
    with pytest.raises(InputError, match="takes a Model"):
        moments(signed, np.array([0.1, 1.0]))


def test_likelihood_refuses_a_distribution_function(signed):
    data = Data(
        observable="sigma_los",
        values=np.array([1.0]),
        r_2d=np.array([1.0]),
        errors=np.array([0.1]),
    )
    with pytest.raises(InputError, match="takes a Model"):
        likelihood(signed, data)
