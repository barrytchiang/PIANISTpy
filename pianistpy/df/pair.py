r"""The potential-density pair a DF is inverted from.

An untruncated model's pair is analytic: its components' closed forms, in
the internal unit of the potential (the central value 1), evaluated where
they are needed. A Method II model's pair is tabulated: the rows of its
parent's truncated profile, whose central potential is :math:`1 - \mathcal{K}_{\rm t}`.
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
    """Return the total potential and the component densities.

    Parameters
    ----------
    components : sequence of Component
        The components of the model.
    parent : TruncatedProfile, optional
        The truncated profile to invert, for Method II. If omitted, it is the
        profile of a component built on one; otherwise the pair is the
        analytic one of the untruncated components.

    Returns
    -------
    pair : PotentialDensityPair

    Raises
    ------
    InputError
        If an untruncated component's profile has no closed form.
    """
    components = tuple(components)
    if parent is None and isinstance(components[0].profile, TruncatedProfile):
        # a component on another model's truncated profile: the pair is
        # tabulated, read from the component
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
            p_0=1.0,  # the internal unit of the potential
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
