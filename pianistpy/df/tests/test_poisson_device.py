"""The Poisson solve runs on the CPU on every platform."""

import os
import subprocess
import sys

import pytest

_SCRIPT = """
import pianistpy as pp
pp.set_platform("gpu")
import jax
try:
    jax.devices("gpu")
except RuntimeError:
    print("NO GPU")
    raise SystemExit(0)
import diffrax
solves = []
_solve = diffrax.diffeqsolve
def spy(*args, **kwargs):
    out = _solve(*args, **kwargs)
    solves.append(out)
    return out
diffrax.diffeqsolve = spy
hernquist = pp.Component(pp.DoublePowerLaw.hernquist(), pp.Anisotropy())
pp.solve(hernquist, pp.Truncation(k_t=0.1))
platforms = {
    d.platform
    for out in solves
    for leaf in jax.tree.leaves(out)
    if isinstance(leaf, jax.Array)
    for d in leaf.devices()
}
print("SOLVES", len(solves), "PLATFORMS", sorted(platforms))
"""


def test_the_solve_comes_back_on_the_cpu_under_the_gpu_platform():
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("JAX_PLATFORMS", "PIANISTPY_PLATFORM")
    }
    run = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        check=False,
        text=True,
        env=env,
        timeout=1800,
    )
    if "NO GPU" in run.stdout or "Unable to initialize backend" in run.stderr:
        pytest.skip("no CUDA GPU reachable")
    assert run.returncode == 0, run.stderr[-2000:]
    line = [s for s in run.stdout.splitlines() if s.startswith("SOLVES")][-1]
    assert not line.startswith("SOLVES 0 ")
    assert line.endswith("PLATFORMS ['cpu']"), line
