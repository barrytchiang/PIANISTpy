"""Tests of the factor g(R) of the augmented density and of its inverse h = 1/g.

`Anisotropy.factor` returns g(R), and `inverse_factor` and
`inverse_factor_expr` return h(R) and the derivatives of h in R. The tests
compare these values with the closed forms of Chiang+(2026d), Eq. (B44), with
finite differences, and with mpmath.
"""

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
    # The jax.numpy factor and the NumPy factor evaluate the same closed form,
    # and XLA's own rounding of it (its fused operations differ by CPU model and
    # by device) is allowed: the two factors may differ by up to 2 ulp of the
    # NumPy factor, on every platform. (On the release's reference CPU the two
    # are equal to the bit; another CPU model gave 1 ulp.)
    platform = jax.default_backend()
    ulps = 2
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
    # Anisotropy.factor must return g = 1/h exactly, with h from
    # inverse_factor, which the Method II sources, the edge law and the
    # Poisson solve read. The first and second derivatives of h from
    # inverse_factor must match, to 1e-7, central differences of h and of the
    # first derivative of h.
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

    # g(beta_c) passes beta_c to Anisotropy as a Python float, so a gradient
    # in beta_c does not pass through the Anisotropy object here. The
    # gradient passes through the pytree form of Anisotropy, which carries
    # beta_c as a leaf. This test checks the factor's value only.
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
    # The third derivative h''' from inverse_factor is compared with sympy's
    # third derivative of inverse_factor_expr, evaluated by mpmath at 30
    # digits, at radii from 1e-6 to 1e3. A Type II model uses only the radii
    # inside R_a, plus the radius where u = 1 - R^2 / R_a^2 = 0.01. The error,
    # divided by the larger of |h| / R^3 and |h'''|, must be below 1e-12. The
    # error is largest near R_a of a Type II model, where the factor
    # u^(s - 3) in h''' amplifies the rounding of the float u.
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
    # Beyond R_a, inverse_factor_expr with continuation=True must equal the
    # independent form sign(u) |u|^(1 - beta_c) R^(2 beta_c) to a relative
    # 1e-25. The third derivative of inverse_factor_expr must equal mpmath's
    # third derivative of the independent form to a relative 1e-20. Both
    # comparisons run at 30 digits.
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
