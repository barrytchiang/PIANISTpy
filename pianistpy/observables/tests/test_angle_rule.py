r"""Tests of the JAX kernel of the angle rule (`angle_rule`) and of the angle
rule's node count (`angle_count`).

The angle rule returns
:math:`\Lambda = \int_0^\pi P^{-\beta_{\rm c}}\,{\rm d}\phi` with
:math:`P = (c_1 - c_2\cos\phi)^2 + c_3\sin^2\phi` at each node. The tests
compare the kernel's :math:`\Lambda` with an adaptive quadrature, with a numpy
copy of the angle rule, and with the closed forms of :math:`\Lambda` at
:math:`\beta_{\rm c} = 1/2` and at :math:`\beta_{\rm c} = 0`.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ellipkm1

from pianistpy.numerics import gauss_legendre_unit
from pianistpy.observables._angle_rule import angle_rule, power_kind


def _nodes(seed, n_nodes, near):
    r"""Return random node arrays (c1, c2, c3, kap, d2, e2, lam, lpi) of
    length `n_nodes`.

    The random draw has c1 and c2 in [0, 2] and c3 above c2 squared. The first
    `near` nodes have c2 within a relative :math:`10^{-3}` of c1, which makes
    the width lam of the minimum of P at :math:`\phi = 0` small. The arrays
    kap, d2, e2, lam and lpi are formed from c1, c2 and c3 by the formulas of
    `EllipseRoute` in ``_ellipse.py``.
    """
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
    """Return Lambda by a numpy copy of the angle rule at `n` nodes a half.

    The copy performs the kernel's operations in the kernel's order. The
    copy takes the power of P by the closed form that `power_kind` returns for
    `beta_c`, and by exp and log for any other `beta_c`.
    """
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
    """Return Lambda at one node by adaptive quadrature (scipy's quad).

    The interval [0, pi] is split at pi / 2, at the lesser of 10 lam and
    pi / 4 from 0, and at the lesser of 10 lpi and pi / 4 from pi.
    """

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
    r"""Assert that the kernel's Lambda matches the reference values.

    At 32 nodes a half, the kernel's Lambda differs from the adaptive
    quadrature by a relative error below :math:`10^{-12}` at every node. At
    4, 8, 16 and 32 nodes a half, the kernel's Lambda differs from the numpy
    copy of the angle rule by a relative error below
    :math:`4 \times 10^{-15}`. The cases cover
    :math:`\beta_{\rm c} = \pm 1/4`, where the power is taken by nested
    square roots, and :math:`\beta_{\rm c} = -0.4` and :math:`0.3`, where the power
    is taken by exp and log.
    """
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
    r"""Assert that the kernel's Lambda at :math:`\beta_{\rm c} = 1/2` matches
    the closed form :math:`2 K(m) / \sqrt{T}` to a relative :math:`10^{-12}`.

    In the closed form, :math:`K` is the complete elliptic integral of the
    first kind, :math:`T = c_1^2 + c_3 + e + 2\sqrt{c_3 (c_1^2 + e)}` with
    :math:`e = c_3 - c_2^2`, and :math:`1 - m = ((c_1^2 - c_2^2) / T)^2`. The
    test also asserts that `power_kind` takes the power at
    :math:`\beta_{\rm c} = 1/2` by a reciprocal square root.
    """
    c1, c2, c3, kap, d2, e2, lam, lpi = _nodes(5, 200, 40)
    e = c3 - c2 * c2
    t = c1 * c1 + c3 + e + 2.0 * np.sqrt(c3 * (c1 * c1 + e))
    closed = 2.0 * ellipkm1(((c1 * c1 - c2 * c2) / t) ** 2) / np.sqrt(t)
    got = angle_rule(0.5, c1, c2, c3, kap, d2, e2, lam, lpi, 32)
    assert power_kind(0.5) == "rsqrt"
    assert np.max(np.abs(got / closed - 1.0)) < 1e-12


def test_isotropic_limit_and_shape():
    r"""Assert that the kernel's Lambda at :math:`\beta_{\rm c} = 0` equals
    pi to a relative :math:`10^{-14}`, since the integrand is 1.

    The 12 nodes are passed as arrays of shape (3, 4), fewer nodes than the
    kernel's least padded length `BUCKET_MIN`. The test asserts that the
    kernel returns an array of shape (3, 4).
    """
    nodes = _nodes(6, 12, 3)
    got = angle_rule(0.0, *(a.reshape(3, 4) for a in nodes), 16)
    assert got.shape == (3, 4)
    assert np.max(np.abs(got / np.pi - 1.0)) < 1e-14


def test_a_node_off_the_need_grid_takes_the_full_rule():
    """Assert the index `angle_count` returns for a node off its grid.

    The node with c1 = c2 = 0.5 has a root of P at cos(phi) = 1 and gets the
    index len(counts), which selects the full angle rule. The node with
    c1 = 0.5 and c2 = 0.2 gets an index below len(counts). A node with
    c1 = NaN raises AccuracyError.
    """
    from pianistpy.exceptions import AccuracyError
    from pianistpy.observables._angle import angle_count

    counts = (8, 16, 32)
    one = np.ones(2)
    # At c1 == c2 the root of P in cos(phi) lies at cos(phi) = 1, the end of
    # the interval. The root's distance to the end, zero, lies below the first
    # point of angle_count's grid, so angle_count reads no node count from
    # the grid and gives the node the full rule.
    k = angle_count(np.array([0.5, 0.5]), np.array([0.5, 0.2]), 0.1 * one, one, counts)
    assert k[0] == len(counts)
    assert k[1] < len(counts)
    with pytest.raises(AccuracyError, match="not numbers"):
        angle_count(np.array([np.nan]), np.array([0.2]), 0.1 * one[:1], one[:1], counts)
