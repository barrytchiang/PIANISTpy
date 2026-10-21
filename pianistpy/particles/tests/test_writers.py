"""Tests of the GAMER, Gadget and text particle writers."""

import h5py
import numpy as np
import pytest

from pianistpy import InputError, ResultError
from pianistpy.particles import PARTICLE_WRITERS, Particles, writers


def _particles(n=11, equal=True, components=1):
    rng = np.random.default_rng(3)
    mass = np.full(n, 1.0 / n) if equal else rng.uniform(0.5, 1.5, n) / n
    return Particles(
        position=rng.standard_normal((n, 3)),
        velocity=rng.standard_normal((n, 3)),
        mass=mass,
        component=np.arange(n) % components,
        component_names=tuple(f"c{i}" for i in range(components)),
        unit_system="internal",
        units={"position": "r_0", "velocity": "v_W", "mass": "M_W"},
        identity="a" * 64,
        parent="b" * 64,
    )


def test_the_three_layouts_are_registered():
    assert {"gamer", "gadget", "text"} <= set(PARTICLE_WRITERS)


@pytest.mark.parametrize("order", ["ATT_ID", "ID_ATT"])
@pytest.mark.parametrize("double_precision", [True, False])
@pytest.mark.parametrize("include_mass", [True, False])
def test_gamer_is_the_raw_stream_of_the_attributes(
    tmp_path, order, double_precision, include_mass
):
    particles = _particles()
    path = tmp_path / "PAR_IC"
    particles.write(
        path,
        layout="gamer",
        order=order,
        double_precision=double_precision,
        include_mass=include_mass,
    )
    dtype = np.float64 if double_precision else np.float32
    columns = (
        ([particles.mass] if include_mass else [])
        + [particles.position[:, i] for i in range(3)]
        + [particles.velocity[:, i] for i in range(3)]
    )
    expected = np.stack(columns)
    if order == "ID_ATT":
        expected = expected.T
    back = np.fromfile(path, dtype=dtype).reshape(expected.shape)
    assert np.array_equal(back, expected.astype(dtype))
    assert path.stat().st_size == expected.size * np.dtype(dtype).itemsize


def test_gamer_offset_moves_the_positions_only(tmp_path):
    particles = _particles()
    path = tmp_path / "PAR_IC"
    particles.write(path, layout="gamer", offset=(10.0, 20.0, 30.0))
    back = np.fromfile(path, dtype=np.float64).reshape(7, len(particles))
    assert np.array_equal(back[1:4].T, particles.position + [10.0, 20.0, 30.0])
    assert np.array_equal(back[4:7].T, particles.velocity)


def test_gamer_legacy_is_the_thirteen_float32_blocks(tmp_path):
    particles = _particles()
    n = len(particles)
    path = tmp_path / "PAR_IC_legacy"
    particles.write(path, layout="gamer", attributes="legacy")
    back = np.fromfile(path, dtype=np.float32).reshape(13, n)
    assert np.array_equal(back[0], particles.mass.astype(np.float32))
    assert np.array_equal(back[1:4].T, particles.position.astype(np.float32))
    assert np.array_equal(back[4:7].T, particles.velocity.astype(np.float32))
    assert np.all(back[7] == 1.0) and np.all(back[8] == 1.0)
    assert np.all(back[9:12] == 0.0)
    assert np.array_equal(back[12], np.arange(n, dtype=np.float32))
    assert path.stat().st_size == 13 * 4 * n
    with pytest.raises(InputError, match="legacy layout"):
        particles.write(
            tmp_path / "x", layout="gamer", attributes="legacy", double_precision=False
        )


def test_gamer_legacy_refuses_more_identifiers_than_a_float32_carries(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(writers, "ID_LIMIT_FLOAT32", 8)
    with pytest.raises(InputError, match="identifiers"):
        _particles(n=9).write(tmp_path / "PAR_IC", layout="gamer", attributes="legacy")
    _particles(n=8).write(tmp_path / "PAR_IC_8", layout="gamer", attributes="legacy")


def test_gamer_refuses_an_unknown_layout_and_an_existing_file(tmp_path):
    particles = _particles()
    with pytest.raises(InputError, match="order"):
        particles.write(tmp_path / "x", layout="gamer", order="ROW")
    path = tmp_path / "PAR_IC"
    particles.write(path, layout="gamer")
    with pytest.raises(InputError, match="overwrite"):
        particles.write(path, layout="gamer")
    particles.write(path, layout="gamer", overwrite=True)


@pytest.mark.parametrize("double_precision", [False, True])
def test_gadget_has_the_reference_layout(tmp_path, double_precision):
    particles = _particles()
    n = len(particles)
    path = tmp_path / "ic.hdf5"
    particles.write(path, layout="gadget", double_precision=double_precision)
    dtype = np.float64 if double_precision else np.float32
    with h5py.File(path, "r") as file:
        header = file["Header"].attrs
        assert list(header["NumPart_ThisFile"]) == [0, n, 0, 0, 0, 0]
        assert header["NumPart_Total"].dtype == np.uint32
        assert header["MassTable"][1] == particles.mass[0]
        assert int(header["Flag_DoublePrecision"]) == int(double_precision)
        group = file["PartType1"]
        assert set(group) == {"Coordinates", "Velocities", "ParticleIDs"}
        assert group["Coordinates"].dtype == dtype
        assert np.array_equal(
            group["Coordinates"][...], particles.position.astype(dtype)
        )
        assert np.array_equal(
            group["Velocities"][...], particles.velocity.astype(dtype)
        )
        assert np.array_equal(group["ParticleIDs"][...], np.arange(1, n + 1))


def test_gadget_unequal_masses_are_written_per_particle(tmp_path):
    particles = _particles(equal=False)
    path = tmp_path / "ic.hdf5"
    particles.write(path, layout="gadget", part_type=2)
    with h5py.File(path, "r") as file:
        assert file["Header"].attrs["MassTable"][2] == 0.0
        assert np.array_equal(
            file["PartType2/Masses"][...], particles.mass.astype(np.float32)
        )
    with pytest.raises(InputError, match="part_type"):
        particles.write(tmp_path / "y.hdf5", layout="gadget", part_type=6)


def test_text_has_the_header_and_reads_back_to_its_digits(tmp_path):
    particles = _particles(components=2)
    path = tmp_path / "particles.txt"
    particles.write(path, layout="text")
    lines = path.read_text(encoding="ascii").splitlines()
    head = [line for line in lines if line.startswith("#")]
    assert head[1] == "# unit_system: internal"
    assert head[5] == "# columns: mass x y z vx vy vz component"
    units = "M_W | r_0 | r_0 | r_0 | v_W | v_W | v_W | index into the components"
    assert head[6] == "# units: " + units
    rows = [line for line in lines if not line.startswith("#")]
    assert len(rows) == len(particles)
    # Every text field is 25 characters wide with 14 digits after the decimal
    # point.
    assert rows[0][:25].strip() == f"{particles.mass[0]:.14e}"
    assert rows[0][25:50].strip() == f"{particles.position[0, 0]:.14e}"
    back = np.loadtxt(path)
    expected = np.column_stack([particles.mass, particles.position, particles.velocity])
    assert np.allclose(back[:, :7], expected, rtol=1e-14, atol=0.0)
    assert np.array_equal(back[:, 7].astype(int), particles.component)


def test_text_of_one_component_has_no_component_column(tmp_path):
    path = tmp_path / "particles.txt"
    _particles().write(path, layout="text")
    assert np.loadtxt(path).shape == (11, 7)


def test_a_file_that_does_not_read_back_is_refused(tmp_path, monkeypatch):
    particles = _particles()
    monkeypatch.setattr(
        writers.np, "fromfile", lambda *a, **k: np.zeros(7 * len(particles))
    )
    with pytest.raises(ResultError, match="read back"):
        particles.write(tmp_path / "PAR_IC", layout="gamer")
