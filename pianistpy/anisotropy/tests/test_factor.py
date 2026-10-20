"""The factor g(R) of the augmented density, the one definition in the package."""

import numpy as np
import pytest

from pianistpy import Anisotropy, InputError
from pianistpy.anisotropy.factor import inverse_factor

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
    # arithmetic (measured: one value of four apart on a GPU, at 2.2e-16
    # relative, one epsilon, which is up to 2 ulp of the value)
    platform = jax.default_backend()
    ulps = 0 if platform == "cpu" else 2
    off = np.abs(g_jax - g_np) / np.spacing(np.abs(g_np))
    assert np.all(off <= ulps), f"on {platform}: {off.max():.0f} ulp, {ulps} allowed"


@pytest.mark.parametrize("beta_c", [0.0, 0.25])
def test_a_type_ii_factor_is_refused_at_and_beyond_r_a(beta_c):
    anisotropy = Anisotropy(beta_c=beta_c, r_a=10.0, kind="II")
    for r in (10.0, 12.0):
        with pytest.raises(InputError, match="not defined there"):
            anisotropy.factor(np.array([1.0, r]))
    assert np.all(np.isfinite(anisotropy.factor(np.nextafter(10.0, 0.0))))


@pytest.mark.parametrize(
    "anisotropy",
    [
        Anisotropy(beta_c=0.3),
        Anisotropy(r_a=10.0, kind="II"),
        Anisotropy(beta_c=-0.25, r_a=10.0),
        Anisotropy(beta_c=0.25, r_a=10.0, kind="II"),
    ],
)
def test_the_factor_and_its_inverse_are_one_form(anisotropy):
    # g = 1/h, h from inverse_factor (the sources, the edge law and the
    # Poisson solve read h); h' and h'' against central differences of the
    # order below
    (h,) = inverse_factor(R, anisotropy.family, anisotropy.beta_c, anisotropy.r_a)
    assert np.array_equal(anisotropy.factor(R), 1.0 / h)
    x = R[1:]
    _, h_r, h_rr = inverse_factor(
        x, anisotropy.family, anisotropy.beta_c, anisotropy.r_a, 2
    )
    step = 1e-4 * x

    def at(x, k):
        args = (anisotropy.family, anisotropy.beta_c, anisotropy.r_a, 1)
        return inverse_factor(x, *args)[k]

    for k, d in ((0, h_r), (1, h_rr)):
        diff = (at(x + step, k) - at(x - step, k)) / (2 * step)
        assert np.allclose(d, diff, rtol=1e-7, atol=0.0)


def test_factor_of_a_constant_model_is_traceable_in_beta_c():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy

    def g(beta_c):
        return Anisotropy(beta_c=float(beta_c)).factor(2.0, xp=jnp)

    # beta_c enters as a Python float of the frozen object: the gradient with
    # respect to it is not through the object (the anisotropy's pytree
    # form carries it); the value is checked here.
    assert float(g(0.3)) == pytest.approx(2.0**-0.6, rel=1e-15)


_EXACT_CASES = [
    Anisotropy(),
    Anisotropy(beta_c=0.3),
    Anisotropy(beta_c=-0.25),
    Anisotropy(r_a=10.0),
    Anisotropy(r_a=10.0, kind="II"),
    Anisotropy(beta_c=-0.25, r_a=10.0),
    Anisotropy(beta_c=0.25, r_a=10.0),
    Anisotropy(beta_c=-0.25, r_a=10.0, kind="II"),
    Anisotropy(beta_c=0.4, r_a=3.0, kind="II"),
]


@pytest.mark.parametrize("anisotropy", _EXACT_CASES, ids=lambda a: f"{a.family}")
def test_third_derivative_matches_the_exact_expression(anisotropy):
    # h''' against sympy's third derivative of inverse_factor_expr evaluated by
    # mpmath at 30 digits, radii 1e-6 to 1e3 (inside R_a for Type II), the
    # error over max(|h| / R^3, |h'''|). Measured (721 radii): at most 2.7e-13,
    # COM II beta_c 0.4 R_a 3 at R = 2.985 (u = 1 - R^2 / R_a^2 = 0.01, the
    # float u's rounding amplified by u^(s - 3)); every other case 2.4e-15 or less.
    mp = pytest.importorskip("mpmath")
    sympy = pytest.importorskip("sympy")
    from pianistpy.anisotropy.factor import inverse_factor_expr

    a = anisotropy
    ra = a.r_a if np.isfinite(a.r_a) else 1.0
    rs = sympy.symbols("R", positive=True)
    expr = inverse_factor_expr(
        rs, a.family, sympy.Rational(a.beta_c), sympy.Rational(ra)
    )
    h0 = sympy.lambdify(rs, expr, "mpmath")
    h3 = sympy.lambdify(rs, sympy.diff(expr, rs, 3), "mpmath")
    r = np.logspace(-6, 3, 181)
    if a.kind == "II":
        r = np.append(r[r < a.r_a], a.r_a * np.sqrt(1.0 - 1e-2))
    got = np.broadcast_to(
        np.asarray(inverse_factor(r, a.family, a.beta_c, a.r_a, 3)[3], dtype=float),
        r.shape,
    )
    with mp.workdps(30):
        for x, g in zip(r, got, strict=True):
            xm = mp.mpf(float(x))
            ref = h3(xm)
            scale = max(abs(h0(xm)) / xm**3, abs(ref))
            assert float(abs(g - ref) / scale) < 1e-12, x


@pytest.mark.parametrize("beta_c", [-0.25, 0.25, 0.4])
def test_the_com_ii_continuation_is_the_signed_power(beta_c):
    # The continuation branch (continuation=True, beyond R_a) against the
    # independent form sign(u) |u|^(1 - beta_c) R^(2 beta_c) and its mpmath
    # third derivative, relative. Measured: h 3e-31, h''' 8e-23 (30 digits).
    mp = pytest.importorskip("mpmath")
    sympy = pytest.importorskip("sympy")
    from pianistpy.anisotropy.factor import inverse_factor_expr

    ra = 10.0
    rs = sympy.symbols("R", positive=True)
    expr = inverse_factor_expr(
        rs, "COM_II", sympy.Rational(beta_c), sympy.Rational(ra), continuation=True
    )
    h0 = sympy.lambdify(rs, expr, "mpmath")
    h3 = sympy.lambdify(
        rs, sympy.diff(expr, rs, 3).replace(sympy.DiracDelta, lambda *_: 0), "mpmath"
    )
    with mp.workdps(30):
        b = mp.mpf(beta_c)

        def signed(x):
            u = 1 - x**2 / ra**2
            return x ** (2 * b) * mp.sign(u) * abs(u) ** (1 - b)

        for x in np.logspace(np.log10(ra * 1.001), 3, 31):
            xm = mp.mpf(float(x))
            assert abs(h0(xm) / signed(xm) - 1) < mp.mpf("1e-25")
            assert abs(h3(xm) / mp.diff(signed, xm, 3) - 1) < mp.mpf("1e-20")
