"""Tests of the Particles result, the native writer and the registration of
particle writers."""

import numpy as np
import pytest

from pianistpy import InputError, ResultError, read
from pianistpy.io import write_result
from pianistpy.particles import (
    PARTICLE_WRITERS,
    Particles,
    Sampler,
    register_particle_writer,
    write_particles,
)


def _particles(n=7):
    rng = np.random.default_rng(1)
    return Particles(
        position=rng.standard_normal((n, 3)),
        velocity=rng.standard_normal((n, 3)),
        mass=np.full(n, 1.0 / n),
        component=np.array([0, 1] * n)[:n],
        component_names=("halo", "stars"),
        unit_system="internal",
        units={"position": "r_0", "velocity": "v_W", "mass": "M_W"},
        identity="a" * 64,
        parent="b" * 64,
    )


def test_native_round_trip(tmp_path):
    particles = _particles()
    path = tmp_path / "particles.h5"
    particles.write(path)
    back = read(path)
    assert isinstance(back, Particles)
    for name in ("position", "velocity", "mass", "component"):
        assert np.array_equal(getattr(back, name), getattr(particles, name))
    assert back.component_names == particles.component_names
    assert back.unit_system == "internal"
    assert back.units == particles.units
    assert (back.identity, back.parent) == (particles.identity, particles.parent)
    assert len(back) == 7


def test_arrays_are_read_only_and_shapes_are_checked():
    particles = _particles()
    with pytest.raises(ValueError):
        particles.position[0, 0] = 0.0
    with pytest.raises(InputError):
        Particles(
            position=np.zeros((3, 2)),
            velocity=np.zeros((3, 3)),
            mass=np.ones(3),
            component=np.zeros(3, dtype=int),
            component_names=("halo",),
            unit_system="internal",
            units={},
            identity="a",
            parent="b",
        )
    with pytest.raises(InputError):
        Particles(
            position=np.zeros((3, 3)),
            velocity=np.zeros((3, 3)),
            mass=np.ones(3),
            component=np.array([0, 0, 2]),
            component_names=("halo",),
            unit_system="internal",
            units={},
            identity="a",
            parent="b",
        )


def test_a_writer_for_another_layout_is_one_registered_function(tmp_path):
    written = {}

    @register_particle_writer("test-layout")
    def _write(particles, path, scale=1.0):
        written["n"] = len(particles)
        written["scale"] = scale

    try:
        write_particles(_particles(), tmp_path / "x", layout="test-layout", scale=2.0)
        assert written == {"n": 7, "scale": 2.0}
        with pytest.raises(InputError):
            register_particle_writer("test-layout")(_write)
    finally:
        del PARTICLE_WRITERS["test-layout"]
    with pytest.raises(InputError):
        write_particles(_particles(), tmp_path / "x", layout="test-layout")


def test_a_file_of_an_unknown_kind_is_refused(tmp_path):
    path = tmp_path / "other.h5"
    write_result(path, "something_else", "a" * 64, {}, {"k": np.zeros(3)})
    with pytest.raises(ResultError):
        read(path)


def test_a_draw_of_2_31_particles_is_refused_with_cause():
    # Sampler.sample raises InputError before reading the sampler, so the test
    # builds no tables.
    with pytest.raises(InputError, match="is at least 2\\^31"):
        Sampler.sample(None, 2**31)


def test_a_sampler_is_built_from_a_model():
    with pytest.raises(InputError):
        Sampler("not a model")
