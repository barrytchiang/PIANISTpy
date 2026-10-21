"""Tests of the anchor rule of the central form (`pianistpy.df.top`).

The rows are the power law F(k) = (1 - k)^KAPPA with KAPPA = -1.5, so the
local exponent d ln F / d ln u, with u = 1 - k, is KAPPA exactly. The tests
check `kappa_loc` against KAPPA, and check which rule `anchor` applies:
none where the rows reach the ceiling, "data top" where the rows end below
the ceiling, "defect repair" below a band of negative rows, and no rule below
a band of negative rows where the rows do not follow the exponent KAPPA.
"""

import math

import numpy as np

from pianistpy.df.top import anchor, kappa_loc

KAPPA = -1.5
REGULAR = {"branch": "regular", "form": "power"}


def _rows():
    """Return 400 energies k from 0.9 toward 1 and the power law (1 - k)^KAPPA."""
    k = 1.0 - np.logspace(-1, -9, 400)
    return k, (1.0 - k) ** KAPPA


def test_kappa_loc_is_the_exponent():
    k, f = _rows()
    ddf = -KAPPA * (1.0 - k) ** (KAPPA - 1.0)
    assert math.isclose(kappa_loc(k, f, 1.0, k[-1], ddf), KAPPA, rel_tol=1e-12)
    assert math.isclose(kappa_loc(k, f, 1.0, k[-1]), KAPPA, rel_tol=1e-9)


def test_rows_reaching_the_ceiling_stand():
    k, f = _rows()
    assert anchor(k, f, 1.0, KAPPA, REGULAR, 1.0) is None


def test_data_top():
    k, f = _rows()
    top = float(k[300])
    made = anchor(k, f, 1.0, KAPPA, REGULAR, top)
    assert made is not None
    assert made.rule == "data top"
    assert made.k_anchor == top
    assert made.rows == k.size - 301
    assert made.slope_mismatch < 1e-9


def test_defect_repair_below_a_band():
    k, f = _rows()
    f = f.copy()
    f[350:360] = -1.0
    made = anchor(k, f, 1.0, KAPPA, REGULAR, 1.0)
    assert made is not None
    assert made.rule == "defect repair"
    assert made.k_anchor == float(k[349])
    assert made.rows == k.size - 350


def test_no_repair_where_the_form_does_not_hold():
    k, f = _rows()
    f = (1.0 - k) ** -0.5
    f[350:360] = -1.0
    assert anchor(k, f, 1.0, KAPPA, REGULAR, 1.0) is None
