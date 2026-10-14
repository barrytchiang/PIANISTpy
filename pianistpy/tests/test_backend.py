"""Tests of how the package sets JAX."""

import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest

import pianistpy as pp
from pianistpy import InputError
from pianistpy.runtime import COMPILATION_CACHE_MAX_SIZE, default_compilation_cache


def test_jax_runs_in_64_bits_once_the_package_is_imported():
    assert jax.config.jax_enable_x64
    assert jnp.asarray(1.0).dtype == jnp.float64


def test_the_compilation_cache_is_on_by_default_and_can_be_moved_or_turned_off(
    tmp_path,
):
    assert jax.config.jax_compilation_cache_dir == default_compilation_cache()
    try:
        assert pp.set_compilation_cache(tmp_path) == str(tmp_path)
        assert jax.config.jax_compilation_cache_dir == str(tmp_path)
        assert jax.config.jax_compilation_cache_max_size == COMPILATION_CACHE_MAX_SIZE
        pp.set_compilation_cache(tmp_path, max_size=None)
        assert jax.config.jax_compilation_cache_max_size == -1
        assert pp.set_compilation_cache(False) is None
        assert jax.config.jax_compilation_cache_dir is None
    finally:
        pp.set_compilation_cache(True)
    assert jax.config.jax_compilation_cache_dir == default_compilation_cache()


def test_the_platform_switch_names_the_cpu_only_mode(monkeypatch):
    before = jax.config.jax_platforms
    try:
        pp.set_platform("cpu")
        assert jax.config.jax_platforms == "cpu"
        pp.set_platform("auto")
        assert jax.config.jax_platforms == ""
        monkeypatch.setenv("PIANISTPY_PLATFORM", "cpu")
        pp.configure_jax()
        assert jax.config.jax_platforms == "cpu"
    finally:
        jax.config.update("jax_platforms", before or "")
    with pytest.raises(InputError):
        pp.set_platform("")


def test_the_cpu_only_mode_runs_on_the_cpu_in_a_fresh_process():
    # The switch acts before the first computation (set_platform's own
    # contract), and this suite's earlier tests have already started the
    # backend, on a GPU where one is usable; so the platform the mode gives
    # is read in a fresh interpreter, where it holds on every machine.
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import pianistpy as pp; pp.set_platform('cpu'); print(pp.platform())",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "cpu"


def test_the_environment_variable_names_the_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("PIANISTPY_COMPILATION_CACHE", str(tmp_path))
    assert default_compilation_cache() == str(tmp_path)
