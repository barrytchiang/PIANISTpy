"""The factor g(R) of the augmented density, the one definition in the package."""

import numpy as np
import pytest

from pianistpy import Anisotropy

R = np.array([1e-3, 0.3, 1.0, 4.0])


@pytest.mark.parametrize(
    ("anisotropy", "family", "expected"),
    [
        (Anisotropy(), "isotropic", np.ones_like(R)),
        (Anisotropy(beta_c=0.3), "constant", R ** (-0.6)),
        (Anisotropy(r_a=10.0), "OM_I", 1.0 / (1.0 + (R / 10.0) ** 2)),
        (Anisotropy(r_a=10.0, kind="II"), "OM_II", 1.0 / (1.0 - (R / 10.0) ** 2)),
        (
            Anisotropy(beta_c=-0.25, r_a=10.0),
            "COM_I",
            R**0.5 * (1.0 + (R / 10.0) ** 2) ** (-1.25),
        ),
        (
            Anisotropy(beta_c=-0.25, r_a=10.0, kind="II"),
            "COM_II",
            R**0.5 * (1.0 - (R / 10.0) ** 2) ** (-1.25),
        ),
    ],
)
def test_factor_follows_the_paper_case_by_case(anisotropy, family, expected):
    assert anisotropy.family == family
    g = anisotropy.factor(R)
    assert g.shape == R.shape
    assert np.allclose(g, expected, rtol=1e-14, atol=0.0)


def test_factor_in_jax_equals_the_numpy_value():
    jax = pytest.importorskip("jax")
    anisotropy = Anisotropy(beta_c=-0.25, r_a=10.0)
    g_np = anisotropy.factor(R)
    g_jax = np.asarray(anisotropy.factor(jax.numpy.asarray(R), xp=jax.numpy))
    # to the bit on the CPU; elsewhere at rounding, the platform's own
    # arithmetic (3-191: one value of four apart on a GPU, at 2.2e-16
    # relative, one epsilon, which is up to 2 ulp of the value)
    platform = jax.default_backend()
    ulps = 0 if platform == "cpu" else 2
    off = np.abs(g_jax - g_np) / np.spacing(np.abs(g_np))
    assert np.all(off <= ulps), f"on {platform}: {off.max():.0f} ulp, {ulps} allowed"


def test_factor_of_a_constant_model_is_traceable_in_beta_c():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy

    def g(beta_c):
        return Anisotropy(beta_c=float(beta_c)).factor(2.0, xp=jnp)

    # beta_c enters as a Python float of the frozen object: the gradient with
    # respect to it is not through the object (R-47 item 9 is the pytree
    # registration, a later step); the value is checked here.
    assert float(g(0.3)) == pytest.approx(2.0**-0.6, rel=1e-15)
