r"""Potential-density pair from which a DF is inverted.

`potential_density_pair` returns the pair for the inversion. The pair of an
untruncated model is analytic: the inversion evaluates the closed forms of
the components' profiles, with the central potential
equal to 1, the internal unit of the potential. The pair of a Method II
model is tabulated: the radii, potentials and densities of the parent's
truncated profile, whose central potential is
:math:`1 - \mathcal{K}_{\rm t}`, with :math:`\mathcal{K}_{\rm t}` the
parent's truncation energy.
"""

__all__ = ["potential_density_pair"]

from collections.abc import Sequence

import numpy as np

from ..exceptions import InputError
from ..profiles import DoublePowerLaw, TruncatedProfile
from .inputs import Component
from .stages import PotentialDensityPair


def potential_density_pair(
    components: Sequence[Component],
    parent: TruncatedProfile | None = None,
) -> PotentialDensityPair:
    r"""Return the total potential and the component densities of a model.

    Parameters
    ----------
    components : sequence of Component
        The components of the model.
    parent : TruncatedProfile, optional
        Truncated profile to invert, for a Method II model. Default:
        ``None``. With ``None``, the call uses the first component's profile
        where that profile is a `TruncatedProfile`, and the analytic pair of
        the untruncated components otherwise.

    Returns
    -------
    pair : PotentialDensityPair
        The analytic pair (``is_analytic`` true, central potential 1, empty
        tables) or the tabulated pair of the parent (``is_analytic`` false,
        central potential :math:`1 - \mathcal{K}_{\rm t}`, the parent's
        radii, potentials and densities).

    Raises
    ------
    InputError
        If no truncated profile is given or found and a component's profile
        is not a `DoublePowerLaw`.
    """
    components = tuple(components)
    if parent is None and isinstance(components[0].profile, TruncatedProfile):
        # A first component whose profile is a TruncatedProfile gets the
        # tabulated pair of that TruncatedProfile.
        parent = components[0].profile
    if parent is None:
        if not all(isinstance(c.profile, DoublePowerLaw) for c in components):
            raise InputError(
                "An analytic pair takes components with a closed-form profile; "
                "a truncated profile is inverted as a parent."
            )
        return PotentialDensityPair(
            components=components,
            is_analytic=True,
            p_0=1.0,  # The internal unit of the potential.
            r=np.zeros(0),
            p=np.zeros(0),
            rho=np.zeros((len(components), 0)),
        )
    return PotentialDensityPair(
        components=components,
        is_analytic=False,
        p_0=1.0 - float(parent.k_t),
        r=np.asarray(parent.r, dtype=np.float64),
        p=np.asarray(parent.p_trunc, dtype=np.float64),
        rho=np.asarray(parent.rho_trunc, dtype=np.float64),
    )
