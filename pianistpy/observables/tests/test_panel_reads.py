r"""The table's reading at pairs :math:`(R_{\rm 2D}, V)`: each distinct radius
set up once, the break scan in blocks of radii, the same to the bit as each pair
alone."""

import numpy as np
import pytest

from pianistpy import (
    Anisotropy,
    Component,
    DoublePowerLaw,
    LosTable,
    Settings,
    Truncation,
    solve,
)
from pianistpy.observables._ellipse_model import ellipse_route
from pianistpy.observables._losvd_stages import ellipse_rules, table_stencil

QUICK = Settings(quick=True)


@pytest.fixture(scope="module")
def stencil():
    """A constant-anisotropy table's stencil and its route."""
    parent = solve(Component(DoublePowerLaw.hernquist()), Truncation(k_t=0.05), QUICK)
    table = LosTable(
        solve(Component(parent.profile, Anisotropy(beta_c=0.25)), settings=QUICK)
    )
    route = ellipse_route(table.model, ellipse_rules(QUICK), with_df=False)
    route.vrad_kept, route.vk_kept = table.table.kept.v_rad, table.table.kept.v_k
    return table_stencil(route, table.table), route


def test_pairs_that_repeat_a_radius_read_as_each_pair_alone(stencil):
    st, route = stencil
    r = np.array([0.3, 1.0, 0.3, 2.0, 1.0, 0.3])
    v = np.array([0.1, 0.4, 0.25, 0.05, 0.0, 0.6])
    together = st.at_pairs(route, r, v)
    alone = np.array(
        [st.at_pairs(route, r[j : j + 1], v[j : j + 1])[0] for j in range(r.size)]
    )
    assert together.tobytes() == alone.tobytes()


def test_the_break_scan_in_blocks_is_the_scan_at_once(stencil):
    st, route = stencil
    r = np.geomspace(0.05, 5.0, 40)
    blocked = st._breaks(route, r)
    once = route.breaks_many(r)
    for a, b in zip(blocked, once, strict=True):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
