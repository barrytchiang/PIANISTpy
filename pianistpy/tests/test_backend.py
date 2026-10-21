"""Tests of how the package sets JAX."""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest

import pianistpy as pp
from pianistpy import InputError
from pianistpy.runtime import default_compilation_cache


def test_jax_runs_in_64_bits_once_the_package_is_imported():
    assert jax.config.jax_enable_x64
    assert jnp.asarray(1.0).dtype == jnp.float64


def _cache_setting():
    return (
        jax.config.jax_compilation_cache_dir,
        jax.config.jax_compilation_cache_max_size,
    )


def _restore(setting):
    jax.config.update("jax_compilation_cache_dir", setting[0])
    jax.config.update("jax_compilation_cache_max_size", setting[1])


def test_the_compilation_cache_is_on_by_default_and_can_be_moved_or_turned_off(
    tmp_path,
):
    assert jax.config.jax_compilation_cache_dir == default_compilation_cache()
    before = _cache_setting()
    try:
        assert pp.set_compilation_cache(tmp_path) == str(tmp_path)
        assert jax.config.jax_compilation_cache_dir == str(tmp_path)
        # With no max_size, the folder has no bound, as JAX's own cache folder
        # has none.
        assert jax.config.jax_compilation_cache_max_size == -1
        pp.set_compilation_cache(tmp_path, max_size=2**20)
        assert jax.config.jax_compilation_cache_max_size == 2**20
        assert pp.set_compilation_cache(False) is None
        assert jax.config.jax_compilation_cache_dir is None
    finally:
        _restore(before)


def test_with_the_cache_off_there_is_no_default_folder(monkeypatch, tmp_path):
    # With PIANISTPY_COMPILATION_CACHE set to off, there is no default folder,
    # and a cache turned on again uses the package's folder
    # ~/.cache/pianistpy/jax, never a folder named "off".
    monkeypatch.setenv("PIANISTPY_COMPILATION_CACHE", "off")
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_compilation_cache() is None
    before = _cache_setting()
    try:
        folder = pp.set_compilation_cache(True)
        assert folder == os.path.join(str(tmp_path), ".cache", "pianistpy", "jax")
        assert jax.config.jax_compilation_cache_dir == folder
    finally:
        _restore(before)


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
    # set_platform acts only before JAX's first computation in a process, and
    # earlier tests of the suite have already made JAX choose its hardware
    # (a GPU where one is usable). The CPU-only mode is therefore checked in a
    # new Python process, where set_platform("cpu") runs before any
    # computation on every machine.
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
