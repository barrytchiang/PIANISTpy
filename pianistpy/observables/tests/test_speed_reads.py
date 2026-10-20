"""The speed distribution's reads: the unit Gauss-Legendre rule made once per
(panels, nodes), and pdf_v's radii grouped by one sort, the same to the bit."""

import numpy as np
import pytest

from pianistpy import (
    Anisotropy,
    Component,
    DoublePowerLaw,
    Settings,
    Truncation,
    pdf_v,
    solve,
)
from pianistpy.observables._speed import GRADE_MIN, graded_rule

QUICK = Settings(quick=True)


def test_the_graded_rule_from_the_kept_unit_rule_is_the_rule_made_afresh():
    for a, b, n_panels, n_nodes in (
        (0.0, 1.0, 6, 8),
        (0.3, 2.5, 12, 16),
        (1.0, 1.5, 6, 8),
    ):
        x, w = graded_rule(a, b, n_panels, n_nodes)
        h = 0.5 * (b - a) * np.geomspace(GRADE_MIN, 1.0, n_panels)
        edges = np.unique(np.concatenate([[a, b], a + h, b - h]))
        u, uw = np.polynomial.legendre.leggauss(n_nodes)
        lo, hi = edges[:-1], edges[1:]
        half = 0.5 * (hi - lo)
        assert (
            x.tobytes()
            == ((0.5 * (lo + hi))[:, None] + half[:, None] * u).ravel().tobytes()
        )
        assert w.tobytes() == (half[:, None] * uw).ravel().tobytes()


@pytest.fixture(scope="module")
def child():
    parent = solve(Component(DoublePowerLaw.hernquist()), Truncation(k_t=0.05), QUICK)
    return solve(Component(parent.profile, Anisotropy(beta_c=0.25)), settings=QUICK)


def test_pdf_v_at_repeated_radii_reads_as_each_pair_alone(child):
    r = np.array([1.0, 0.3, 1.0, 2.0, 0.3, 1.0])
    v = np.array([0.1, 0.4, 0.25, 0.05, 0.0, 0.6])
    together = np.asarray(pdf_v(child, r, v))
    alone = np.array(
        [float(pdf_v(child, r[j : j + 1], v[j : j + 1])[0]) for j in range(r.size)]
    )
    assert together.tobytes() == alone.tobytes()
