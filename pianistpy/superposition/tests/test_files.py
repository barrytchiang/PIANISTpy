"""A superposed model and its particles in other units and in the files."""

import numpy as np
import pytest

from pianistpy import Sampler, read
from pianistpy.particles.tests.test_stages import _lowered_hernquist
from pianistpy.superposition import superpose
from pianistpy.superposition.tests.test_core import _member

N = 3000


@pytest.fixture(scope="module")
def superposed():
    parent = _lowered_hernquist()
    return superpose(
        {_member(parent, 0.0): 0.4, _member(parent, -0.2, n_keep=1500): 0.6}
    )


@pytest.fixture(scope="module")
def internal(superposed):
    return Sampler(superposed).sample(N, seed=9)


@pytest.mark.parametrize(
    "units",
    [
        dict(system="model", concentration=10.0),
        dict(system="physical", concentration=10.0, mass_0=1e12, r_0=20000.0),
    ],
)
def test_the_superposed_model_in_other_units(superposed, internal, units):
    # .to() presents the same model: its profiles convert per component with the
    # first component's analytic profile, and the draw is the same particles,
    # converted at the output only
    model = superposed.to(**units)
    profile = superposed.components[0].unit_profile
    factors = model.units.factors(profile)
    rows = model.radial_profiles()
    density = np.asarray(getattr(rows["density"], "value", rows["density"]))
    scale = (
        factors.model_rho
        if units["system"] == "model"
        else factors.physical_rho_msun_pc3
    )
    np.testing.assert_allclose(
        density, superposed.profile.rho_trunc * scale, rtol=1e-15
    )
    particles = Sampler(model).sample(N, seed=9)
    assert np.array_equal(particles.component, internal.component)
    v_scale = factors.model_v if units["system"] == "model" else factors.physical_v_kms
    np.testing.assert_allclose(
        particles.velocity, internal.velocity * v_scale, rtol=1e-15
    )


def test_the_draw_in_its_files(internal, tmp_path):
    # the package's own file keeps the labels; the text adds the component
    # column; GAMER's stream carries the composite, label-free (as Product S)
    internal.write(tmp_path / "p.h5")
    back = read(tmp_path / "p.h5")
    assert np.array_equal(back.component, internal.component)
    assert back.position.tobytes() == internal.position.tobytes()
    assert back.component_names == internal.component_names
    internal.write(tmp_path / "p.txt", layout="text")
    columns = np.loadtxt(tmp_path / "p.txt")
    assert columns.shape == (N, 8)
    assert np.array_equal(columns[:, 7].astype(np.int64), internal.component)
    internal.write(tmp_path / "PAR_IC", layout="gamer")
    stream = np.fromfile(tmp_path / "PAR_IC", dtype=np.float64).reshape(7, N)
    assert stream[1:4].T.tobytes() == internal.position.tobytes()
