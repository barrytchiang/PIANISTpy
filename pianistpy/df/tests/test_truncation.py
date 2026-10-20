"""Tests of the truncation's source grid."""

import numpy as np
import pytest

from pianistpy.df.truncation import CEILING_GAP, K_FLOOR, q_fast_grid
from pianistpy.settings import Settings

RUN = Settings().truncation


@pytest.mark.parametrize("q_max", [1.0, 0.95, 0.05, 1e-3])
def test_the_grid_spans_the_floor_to_the_ceiling_gap(q_max):
    q = q_fast_grid(q_max, RUN)
    assert q[0] == K_FLOOR
    assert q[-1] <= q_max - CEILING_GAP
    assert q[-1] == pytest.approx(q_max - CEILING_GAP, rel=0.0, abs=1e-15)
    assert np.all(np.diff(q) > 0.0)
    assert q.size <= RUN["n_source_rows"]


@pytest.mark.parametrize("q_max", [1.0, 0.05])
def test_the_tails_are_logarithmic(q_max):
    q = q_fast_grid(q_max, RUN)
    n_tail = min(RUN["n_source_tail_rows"], RUN["n_source_rows"] // 3)
    low = q[: n_tail // 2]
    assert np.allclose(np.diff(np.log(low)), np.log(low[1] / low[0]), rtol=1e-9)
    gap = q_max - q[-n_tail // 2 :]
    assert np.allclose(np.diff(np.log(gap)), np.log(gap[1] / gap[0]), rtol=1e-4)


def _pieces(q_max):
    """The grid's three pieces before the joins' near-duplicates are dropped."""
    tail_ext = min(RUN["source_tail_extent"], 0.01 * q_max)
    n_tail = min(RUN["n_source_tail_rows"], RUN["n_source_rows"] // 3)
    floor_ = RUN["source_tail_floor"] * q_max
    lo = max(tail_ext * q_max, floor_)
    hi = q_max - max(tail_ext, floor_)
    lin = np.linspace(lo, hi, RUN["n_source_rows"] - 2 * n_tail, endpoint=False)[1:]
    omq = np.logspace(
        np.log10(max(tail_ext, floor_)),
        np.log10(CEILING_GAP),
        n_tail,
    )
    low = np.logspace(np.log10(K_FLOOR), np.log10(lo), n_tail)
    return np.unique(np.concatenate([lin, q_max * (1.0 - omq / q_max), low]))


@pytest.mark.parametrize(
    ("q_max", "n_dropped"), [(1.0, 0), (0.95, 0), (0.05, 1), (1e-3, 1)]
)
def test_only_near_duplicates_at_the_joins_are_dropped(q_max, n_dropped):
    q = q_fast_grid(q_max, RUN)
    merged = _pieces(q_max)
    assert np.all(np.isin(q, merged))
    dropped = np.setdiff1d(merged, q)
    assert dropped.size == n_dropped
    d = np.diff(merged)
    for x in dropped:
        k = int(np.searchsorted(merged, x))
        assert d[k - 1] < 0.05 * np.median(d)


def test_a_lowered_df_negative_beyond_rounding_is_refused():
    # release audit 317 A3: F(K + K_t) - F(K_t) below zero by more than the
    # rounding of the difference is a refusal by name, never a clip; F here
    # rises to K = 2/3 and falls after it, so the DF lowered at K_t = 0.7 is
    # negative above its edge
    from pianistpy import UnphysicalModelError
    from pianistpy.df.truncation import truncated_source

    q = np.geomspace(1e-6, 0.999, 400)
    with pytest.raises(UnphysicalModelError, match="NEGATIVE TRUNCATED DF"):
        truncated_source(
            q_table=q,
            df_table=q**2 * (1.0 - q),
            ddf_dq=None,
            df_has_neg=False,
            k_t=0.7,
            p_center=1.0,
            beta_c=0.0,
            model_type="isotropic",
            settings=Settings().truncation,
        )


def test_a_non_finite_center_form_is_refused_with_its_values():
    # release audit 317 A3: the center's form above the table, where it is NaN,
    # is refused by name and carries what was made, never set to 0 (beyond the
    # range of a double it is +inf, the table's own top band)
    from pianistpy import AccuracyError
    from pianistpy.df.truncated import _extend

    q = np.linspace(0.1, 0.5, 50)
    with pytest.raises(AccuracyError, match="NaN or -inf") as refused:
        _extend(q, q.copy(), 1.0, np.nan, 0.5, -1.0, 0.0, 0.0, ode_rtol=1e-10)
    assert set(refused.value.result) == {"k", "f"}
