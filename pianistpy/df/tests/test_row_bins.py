"""Tests of the row-count bins: the fill and the check read one definition.

The README's quick start once truncated at tier "refused": the check counted
the depth's top bin only up to a tenth of P_0, while the fill filled it up to
a depth of 1e-1.
"""

import numpy as np
import pytest

import pianistpy as pp


@pytest.fixture(scope="module")
def quick_start():
    """The README quick start's model, by the README's own calls, with every
    call of the bins recorded by the module that made it."""
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


def test_the_readme_quick_start_truncates_at_the_goal(quick_start):
    model, _ = quick_start
    assert model.validation.tier == "goal"
    assert model.validation.checks["row_density"].error == 0.0
    # the README's draw, with fewer particles and no file
    particles = pp.Sampler(model).sample(1000, seed=1)
    assert len(particles) == 1000


def test_the_check_reads_the_fills_bins(quick_start):
    _, calls = quick_start
    fills = [bins for name, bins in calls if name == "fill"]
    checks = [bins for name, bins in calls if name == "check"]
    assert fills and checks
    fill, check = fills[-1], checks[-1]
    assert check.min_per_bin == fill.min_per_bin
    assert check.u_top == fill.u_top
    np.testing.assert_allclose(check.psi_edges, fill.psi_edges, rtol=1e-12, atol=0)
    np.testing.assert_array_equal(check.u_edges, fill.u_edges)
