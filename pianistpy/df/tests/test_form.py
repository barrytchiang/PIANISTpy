import numpy as np

from pianistpy.anisotropy import Anisotropy
from pianistpy.df.form import FORMS, inversion_form
from pianistpy.df.inputs import Component
from pianistpy.df.stages import EnergyGrid, PotentialDensityPair
from pianistpy.profiles import DoublePowerLaw


def test_unbounded_outer_end_rows():
    # The COM Type I model with beta_c = -0.75 and outer density slope 3.5 has
    # an auxiliary density rho_aux ~ P^1.5 at the outer end. The power 1.5 is
    # below the floor order m = 2, so the outer end is unbounded. The rows at
    # energies of at least 10 times the table's first energy take the general
    # form, and the rows at lower energies keep the surface-term form.
    component = Component(
        DoublePowerLaw(1.0, 3.5, 1.0), Anisotropy(beta_c=-0.75, r_a=10.0, kind="I")
    )
    k = np.geomspace(1e-10, 0.99, 50)[None, :]
    pair = PotentialDensityPair(
        components=(component,),
        is_analytic=True,
        p_0=1.0,
        r=np.zeros(0),
        p=np.zeros(0),
        rho=np.zeros((1, 0)),
    )
    grid = EnergyGrid(
        k=k, r_of_k=np.full(k.shape, np.nan), k_floor=(1e-10,), k_ceiling=(0.99,)
    )
    form, (record,) = inversion_form(pair, grid)
    assert record.form == "integrated_by_parts+derivative_outside_below_floor"
    assert (record.order_j, record.kernel_exponent) == (2, 0.25)
    assert record.outer_end == "unbounded" and record.outer_power == 1.5
    assert record.outer_from == 10.0 * k[0, 0]
    rows = form.form[0]
    above = k[0] >= record.outer_from
    assert above.any() and not above.all()
    assert np.all(rows[above] == FORMS["general"])
    assert np.all(rows[~above] == FORMS["surface"])
    assert form.k_interior == (None,)
