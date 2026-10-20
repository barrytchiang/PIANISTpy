"""Tests of the angle rule's kernel against an adaptive quadrature, the numpy
rule and the closed forms."""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ellipkm1

from pianistpy.numerics import gauss_legendre_unit
from pianistpy.observables._angle_rule import angle_rule, power_kind


def _nodes(seed, n_nodes, near):
    r"""Node arrays as the rows form them, from ``c1, c2 >= 0`` and c3 > c2^2 (in
    the rows c3 - c2^2 = 2 s / (q A) > 0); `near` of them with c1 within
    :math:`10^{-3}` of c2, where the width lam is small."""
    g = np.random.default_rng(seed)
    c1 = g.uniform(0.0, 2.0, n_nodes)
    c2 = g.uniform(0.0, 2.0, n_nodes)
    c2[:near] = c1[:near] * (1.0 + g.uniform(-1e-3, 1e-3, near))
    c3 = c2 * c2 + g.uniform(0.05, 3.0, n_nodes)
    kap = c1 * c2 + c3 - c2 * c2
    d2 = (c1 - c2) ** 2
    e2 = c2 * c2 - c3
    lam = np.clip(
        np.where(kap > 0.0, np.sqrt(d2 / np.maximum(kap, 1e-300)), np.pi), 1e-15, np.pi
    )
    kpi = c3 - c1 * c2 - c2 * c2
    lpi = np.clip(
        np.where(kpi > 0.0, np.sqrt((c1 + c2) ** 2 / np.maximum(kpi, 1e-300)), np.pi),
        1e-15,
        np.pi,
    )
    return c1, c2, c3, kap, d2, e2, lam, lpi


def _numpy_rule(beta_c, c1, c2, c3, kap, d2, e2, lam, lpi, n):
    """The numpy rule of the rows, its operations in its order, the power by
    the rows' own forms."""
    powm = {
        "sqrt": np.sqrt,
        "rsqrt": lambda p: 1.0 / np.sqrt(p),
        "qrt": lambda p: np.sqrt(np.sqrt(p)),
        "rqrt": lambda p: 1.0 / np.sqrt(np.sqrt(p)),
    }.get(power_kind(beta_c), lambda p: np.exp(-beta_c * np.log(p)))
    xp, wp = gauss_legendre_unit(n)
    total = 0.0
    for lm, to_pi in ((lam, False), (lpi, True)):
        mu = np.arcsinh(0.5 * np.pi / lm)
        ex = np.exp(mu[..., None] * xp)
        ih = 0.5 / ex
        half = 0.5 * ex
        psi = (half - ih) * lm[..., None]
        wpsi = (half + ih) * (lm * mu)[..., None]
        if to_pi:
            p = np.cos(psi) * c2[..., None] + c1[..., None]
            p = p * p + c3[..., None] * np.sin(psi) ** 2
        else:
            h = np.sin(0.5 * psi)
            h = h * h * 2.0
            p = (h * e2[..., None] + 2.0 * kap[..., None]) * h + d2[..., None]
        p = np.maximum(p, 1e-300)
        total = total + np.einsum("...k,...k,k->...", wpsi, powm(p), wp)
    return total


def _adaptive(beta_c, c1, c2, c3, lam, lpi):
    """Lambda by adaptive quadrature, split at pi / 2 and at ten widths from
    each minimum of P."""

    def f(phi):
        return ((c1 - c2 * np.cos(phi)) ** 2 + c3 * np.sin(phi) ** 2) ** (-beta_c)

    a0, a1 = min(10.0 * lam, 0.25 * np.pi), np.pi - min(10.0 * lpi, 0.25 * np.pi)
    cuts = (0.0, a0, 0.5 * np.pi, a1, np.pi)
    return sum(
        quad(f, a, b, epsabs=0.0, epsrel=1e-13, limit=400)[0]
        for a, b in zip(cuts[:-1], cuts[1:], strict=True)
    )


@pytest.mark.parametrize(
    ("beta_c", "seed"),
    [(-0.25, 1), (0.25, 2), (-0.40, 3), (0.30, 4)],
    ids=["COM_I_b-0.25", "COM_I_b+0.25", "constant_b-0.40", "COM_II_b+0.30"],
)
def test_rule_against_quadrature_and_numpy(beta_c, seed):
    r"""The kernel on the canonical forms of the power and a Type II case: the
    adaptive quadrature to :math:`10^{-12}`, the numpy rule at rounding at
    every count."""
    nodes = _nodes(seed, 300, 60)
    lam_32 = angle_rule(beta_c, *nodes, 32)
    ref = np.array(
        [
            _adaptive(beta_c, *(nodes[k][i] for k in (0, 1, 2, 6, 7)))
            for i in range(len(nodes[0]))
        ]
    )
    assert np.max(np.abs(lam_32 / ref - 1.0)) < 1e-12
    for n in (4, 8, 16, 32):
        got = angle_rule(beta_c, *nodes, n)
        assert np.max(np.abs(got / _numpy_rule(beta_c, *nodes, n) - 1.0)) < 4e-15


def test_closed_rule_control_at_one_half():
    r"""At :math:`\beta_{\rm c} = 1/2` the kernel is the closed form's control:
    2 K(m) / sqrt(T)."""
    c1, c2, c3, kap, d2, e2, lam, lpi = _nodes(5, 200, 40)
    e = c3 - c2 * c2
    t = c1 * c1 + c3 + e + 2.0 * np.sqrt(c3 * (c1 * c1 + e))
    closed = 2.0 * ellipkm1(((c1 * c1 - c2 * c2) / t) ** 2) / np.sqrt(t)
    got = angle_rule(0.5, c1, c2, c3, kap, d2, e2, lam, lpi, 32)
    assert power_kind(0.5) == "rsqrt"
    assert np.max(np.abs(got / closed - 1.0)) < 1e-12


def test_isotropic_limit_and_shape():
    r"""At :math:`\beta_{\rm c} = 0` the integrand is 1 and Lambda is pi; the
    shape is the input's and a call shorter than the padding returns only its
    nodes."""
    nodes = _nodes(6, 12, 3)
    got = angle_rule(0.0, *(a.reshape(3, 4) for a in nodes), 16)
    assert got.shape == (3, 4)
    assert np.max(np.abs(got / np.pi - 1.0)) < 1e-14
