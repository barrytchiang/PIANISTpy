import numpy as np

from pianistpy.anisotropy import Anisotropy
from pianistpy.df.grid import inversion_grid
from pianistpy.profiles import DoublePowerLaw
from pianistpy.settings import Settings

SETTINGS = Settings(inversion={"n_grid_rows": 200, "n_grid_tail_rows": 300})


def test_hernquist_isotropic_grid():
    # The isotropic Hernquist grid has the standard zone bounds. The grid is
    # strictly increasing from its anchor at K = 1e-10 and ends at the ceiling
    # K = 1 - 1e-9, with no node above the ceiling.
    q, zones = inversion_grid(DoublePowerLaw(1.0, 4.0, 1.0), Anisotropy(), SETTINGS)
    assert zones.a_hi == 0.03 and zones.ab_hi == 0.375
    assert np.isclose(q[0], 1e-10, rtol=1e-12) and np.all(np.diff(q) > 0.0)
    assert q[-1] == 1.0 - 1e-9
