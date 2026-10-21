r"""Tests of two reads of a `PanelTable` that must match a simpler read to
the bit.

`PanelTable.at_pairs` reads the line-of-sight velocity distribution at pairs
:math:`(R_{\rm 2D}, V)` of projected radius :math:`R_{\rm 2D}` and
line-of-sight velocity :math:`V`. `PanelTable._breaks` computes the break
velocities of the velocity panels at many projected radii in blocks of radii.
"""

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

pytestmark = pytest.mark.slow

QUICK = Settings(quick=True)


@pytest.fixture(scope="module")
def stencil():
    r"""Return a `PanelTable` and an `EllipseRoute` of one model.

    The model has the density profile of a Hernquist model truncated by
    ``Truncation(k_t=0.05)`` and the constant anisotropy
    :math:`\beta_{\rm c} = 0.25`, and is solved with the quick settings. The
    `PanelTable` reads the model's `LosTable`.
    """
    parent = solve(Component(DoublePowerLaw.hernquist()), Truncation(k_t=0.05), QUICK)
    table = LosTable(
        solve(Component(parent.profile, Anisotropy(beta_c=0.25)), settings=QUICK)
    )
    route = ellipse_route(table.model, ellipse_rules(QUICK), with_df=False)
    route.vrad_kept, route.vk_kept = table.table.kept.v_rad, table.table.kept.v_k
    return table_stencil(route, table.table), route


def test_pairs_that_repeat_a_radius_read_as_each_pair_alone(stencil):
    """Assert that `PanelTable.at_pairs` at six pairs, which repeat the
    projected radii 0.3 and 1.0, returns the same bits as six calls of
    `PanelTable.at_pairs` at one pair each."""
    st, route = stencil
    r = np.array([0.3, 1.0, 0.3, 2.0, 1.0, 0.3])
    v = np.array([0.1, 0.4, 0.25, 0.05, 0.0, 0.6])
    together = st.at_pairs(route, r, v)
    alone = np.array(
        [st.at_pairs(route, r[j : j + 1], v[j : j + 1])[0] for j in range(r.size)]
    )
    assert together.tobytes() == alone.tobytes()


def test_the_break_scan_in_blocks_is_the_scan_at_once(stencil):
    """Assert that `PanelTable._breaks` at 40 projected radii from 0.05 to 5
    returns the same bits as one call of `EllipseRoute.breaks_many` at all 40
    radii, for each of the three arrays of break velocities."""
    st, route = stencil
    r = np.geomspace(0.05, 5.0, 40)
    blocked = st._breaks(route, r)
    once = route.breaks_many(r)
    for a, b in zip(blocked, once, strict=True):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()
