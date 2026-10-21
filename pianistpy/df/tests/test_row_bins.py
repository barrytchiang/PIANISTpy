r"""Tests of the row-count bins of a truncated profile.

The fill of the profile's rows (`pianistpy.df.fill`) and the row-density check
(`pianistpy.df.profile_checks`) both build their bins with
`pianistpy.df.truncation.row_bin_edges`. The tests check that the check's bins
equal the fill's bins on the README's quick-start model, and that the
quick-start model's validation report has tier "silent".
"""

import numpy as np
import pytest

import pianistpy as pp


@pytest.fixture(scope="module")
def quick_start():
    """Return the README quick-start model and the recorded bin calls.

    The model is built by the README's own calls. Every call of
    `row_bin_edges` from the fill module or from the check module is recorded
    as a pair of the caller's name ("fill" or "check") and the returned bins.
    """
    import pianistpy.df.fill as fill
    import pianistpy.df.profile_checks as profile_checks
    from pianistpy.df.truncation import row_bin_edges

    calls = []

    def recorded(name):
        def edges(*args, **kwargs):
            bins = row_bin_edges(*args, **kwargs)
            calls.append((name, bins))
            return bins

        return edges

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(fill, "row_bin_edges", recorded("fill"))
        mp.setattr(profile_checks, "row_bin_edges", recorded("check"))
        halo = pp.Component(pp.DoublePowerLaw.nfw(), pp.Anisotropy(beta_c=0.25))
        df = pp.compute_df(halo)
        model = pp.truncate(df, pp.Truncation(mass_t=1.0, concentration=10))
    return model, calls


def test_the_readme_quick_start_truncates_silent(quick_start):
    model, _ = quick_start
    assert model.validation.tier == "silent"
    assert model.validation.checks["row_density"].error == 0.0
    # The README's particle draw, with 1000 particles and no file written.
    particles = pp.Sampler(model).sample(1000, seed=1)
    assert len(particles) == 1000


def test_the_check_reads_the_fills_bins(quick_start):
    _, calls = quick_start
    fills = [bins for name, bins in calls if name == "fill"]
    checks = [bins for name, bins in calls if name == "check"]
    assert fills and checks
    fill, check = fills[-1], checks[-1]
    assert check.min_per_bin == fill.min_per_bin
    # The depth top u_top of the fill's bins is read from the solve's rows, and
    # the depth top u_top of the check's bins from the profile's rows, so the two
    # u_top values may differ. The bin edges of the fill and of the check must
    # still be equal.
    for side in (fill, check):
        assert side.u_top <= side.psi_top  # psi_top is ROW_BINS_TOP times P_0.
    np.testing.assert_allclose(check.psi_edges, fill.psi_edges, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(check.u_edges, fill.u_edges)
