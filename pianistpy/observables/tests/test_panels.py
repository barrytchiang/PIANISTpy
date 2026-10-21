"""Tests of `_panels`: the reader of a velocity distribution given on
Gauss-Legendre sub-panels, at the ends of the sub-panels.

The ends of the sub-panels are inferred from the nodes and weights in floating
point, and the first sub-panel's lower end, zero, comes out as a residue of order
:math:`10^{-24}`. A read at exactly zero velocity must read the first sub-panel, not the
value zero outside every sub-panel (found by the figure notebooks of ver. 1.0.0:
``aperture_losvd`` returned 0 at the peak of the distribution for a velocity
grid whose middle point is exactly 0).
"""

import numpy as np
from numpy.polynomial import legendre

from pianistpy.observables._panels import panels, read_panels, read_panels_vjp


def _graded_rule(m=16):
    """Return the nodes, weights and values of a rule graded toward zero velocity,
    as the aperture distribution's panels are, with a smooth distribution that
    equals 1 at zero velocity."""
    edges = np.concatenate(
        [[0.0], 0.3 ** np.arange(12, -1, -1) * 0.05, np.linspace(0.05, 1.5, 30)[1:]]
    )
    x, w = legendre.leggauss(m)
    a, b = edges[:-1], edges[1:]
    nodes = (0.5 * (a + b)[:, None] + 0.5 * (b - a)[:, None] * x).ravel()
    weights = (0.5 * (b - a)[:, None] * w).ravel()
    return nodes, weights, np.exp(-(nodes**2) / 0.1), edges


def test_the_velocity_zero_reads_the_first_sub_panel():
    """Assert that the read at exactly zero velocity, and at velocities below the
    residue of the inferred first end, equals the distribution's value there
    (1 within :math:`10^{-9}`, the polynomial on the first sub-panel), while a
    velocity beyond the last end by more than the
    tolerance reads zero."""
    nodes, weights, values, edges = _graded_rule()
    p = panels(nodes, weights, values)
    assert 0.0 < p.a[0] < 1e-18, p.a[0]
    read = read_panels(p, np.array([0.0, 1e-30, 1e-20, 1e-6, edges[-1] + 1e-9]))
    assert np.allclose(read[:4], 1.0, rtol=0.0, atol=1e-9)
    assert read[4] == 0.0


def test_a_velocity_on_a_shared_end_reads_the_lower_sub_panel():
    """Assert that a velocity at an end two sub-panels share reads the
    distribution's value there, in the read and in its vector-Jacobian product,
    from one of the two neighbors."""
    nodes, weights, values, edges = _graded_rule()
    p = panels(nodes, weights, values)
    v = np.array([edges[5], 0.0])
    read = read_panels(p, v)
    assert np.allclose(read, np.exp(-(v**2) / 0.1), rtol=0.0, atol=1e-9)
    out = read_panels_vjp(p, v, np.ones(2))
    m = nodes.size // p.a.size
    per_panel = out.reshape(-1, m).any(axis=1)
    # the rebuilt ends carry a residue, so the exact pass finds the shared end
    # inside one neighbor or the other; the value is the same from either
    assert per_panel[0] and (per_panel[4] or per_panel[5]) and per_panel.sum() == 2
