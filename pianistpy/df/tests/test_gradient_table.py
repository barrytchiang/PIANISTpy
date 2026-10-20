r"""Tests of the DF table's derivative in the anisotropy parameters.

The table (`invert._table_pieces`) at fixed grid and discrete choices,
against the forward's own central differences in :math:`\beta_{\rm c}`
(Richardson over two steps), on a Hernquist member whose table has every
piece but the tail: the rows, the nodes below the written floor and the form
above the anchor.
"""

import numpy as np
import pytest

from pianistpy import platform
from pianistpy.anisotropy import Anisotropy
from pianistpy.df import invert
from pianistpy.df.gradient import (
    _with,
    inversion_nodes,
    make_inversion_table,
    table_jacobian,
)
from pianistpy.df.inputs import Component
from pianistpy.df.pair import potential_density_pair
from pianistpy.exceptions import InputError
from pianistpy.profiles import DoublePowerLaw
from pianistpy.settings import Settings

HERNQUIST = DoublePowerLaw(1.0, 4.0, 1.0)
# the transpose held to the bit is a property of a fixed reduction order,
# which a GPU does not promise; there the accuracy grid judges
CPU_ONLY = pytest.mark.skipif(
    platform() != "cpu",
    reason="deterministic recomputation, a property of a fixed reduction order; "
    "on a GPU the accuracy grid judges",
)


def _forward_table(c, q):
    pair = potential_density_pair((c,))
    f, ddf, _, aux, floor = invert._rows(c, Settings(), pair, q)
    f, ddf = np.asarray(f), np.asarray(ddf)
    p = invert._table_pieces(c, Settings(), q, f, ddf, aux, floor)
    t, patch = p["table"], p["patch"]
    return dict(
        k=np.asarray(t.k[0]),
        log_df=np.asarray(t.log_df[0]),
        log_df_slope=np.asarray(t.log_df_slope[0]),
        k_below=np.asarray(t.k_below[0]),
        log_df_below=np.asarray(t.log_df_below[0]),
        log_df_slope_below=np.asarray(t.log_df_slope_below[0]),
        form_df=np.log10(p["form"].central_df(q[q > patch.k_anchor])),
    )


@pytest.fixture(scope="module")
def table():
    c = Component(HERNQUIST, Anisotropy(beta_c=0.25))
    values, jac, record = table_jacobian(c, Settings())
    return c, values, jac, record


def test_the_table_against_the_forward_differences(table):
    # the forward's differences at b +- h and +- h/2, Richardson, h = 2e-3, on
    # the same grid; the rows compared where the differences resolve them
    # (from 1e-6 to 0.99: at the floor the forward varies with beta_c by a few
    # 1e-8 of F off any smooth curve, near the center its slopes by more)
    c, values, jac, _ = table
    q = inversion_nodes(c, Settings())
    h = 2e-3
    diff = {}
    for hh in (h, h / 2):
        up = _forward_table(_with(c, 0.25 + hh, c.anisotropy.r_a), q)
        dn = _forward_table(_with(c, 0.25 - hh, c.anisotropy.r_a), q)
        assert np.array_equal(up["k"], values["k"])
        assert np.array_equal(up["k_below"], values["k_below"])
        diff[hh] = {k: (up[k] - dn[k]) / (2 * hh) for k in up if not k.startswith("k")}
    rich = {k: (4 * diff[h / 2][k] - diff[h][k]) / 3 for k in diff[h]}
    d = jac["beta_c"]
    k_t = values["k"]
    mid = (k_t >= 1e-6) & (k_t <= 0.99)
    # (the log slope's derivative reaches 1e2 near 0.99: 1e-6 of it, or 1e-6)
    for key in ("log_df", "log_df_slope"):
        np.testing.assert_allclose(d[key][mid], rich[key][mid], rtol=1e-6, atol=1e-6)
    for key in ("log_df_below", "log_df_slope_below"):
        np.testing.assert_allclose(d[key], rich[key], rtol=1e-6, atol=1e-6)
    # the form above the anchor, as the DF it gives on the grid's nodes there
    kap, a_a, bb, dd = values["central"]
    dk, da, db, ddd = d["central"]
    above = q[q > k_t[-1]]
    u_a = 1.0 - k_t[-1]
    rr = (1.0 - above) / u_a
    form = a_a * rr**kap + bb + dd * (1.0 - above)
    d_form = da * rr**kap + a_a * rr**kap * np.log(rr) * dk + db + ddd * (1.0 - above)
    np.testing.assert_allclose(
        d_form / (form * np.log(10.0)), rich["form_df"], rtol=1e-6, atol=0
    )


@CPU_ONLY
def test_the_table_vjp_is_the_transpose(table):
    import jax

    c, values, jac, _ = table
    rows, params, values2 = make_inversion_table(c, Settings())
    out, vjp = jax.vjp(rows, np.array([0.25]))
    assert params == ("beta_c",)
    for k in out:
        np.testing.assert_array_equal(np.asarray(out[k]), values2[k])
    rng = np.random.default_rng(2)
    v = {k: np.nan_to_num(rng.normal(size=np.shape(out[k]))) for k in out}
    (g,) = vjp(v)
    want = sum(float(np.nansum(v[k] * jac["beta_c"][k])) for k in out)
    assert float(g[0]) == pytest.approx(want, rel=1e-12)


def test_a_critical_center_is_refused():
    c = Component(DoublePowerLaw(0.5, 4.0, 1.0), Anisotropy(beta_c=0.5))
    with pytest.raises(InputError, match="critical band"):
        table_jacobian(c, Settings())
