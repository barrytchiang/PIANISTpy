"""Distribution functions in closed form, and the models built from them.

The isotropic Hernquist model and the isotropic Plummer model have a
distribution function (DF) in closed form: `hernquist_df` and `plummer_df`
return each DF and its derivative. The closed forms are references
independent of the numerical inversion. `analytic_model` builds a complete
`Model` from either closed form, for tests and examples that need a model
without an inversion.
"""

__all__ = ["ANALYTIC_MODELS", "analytic_model", "hernquist_df", "plummer_df"]

import math

import numpy as np
from scipy.special import beta as beta_function
from scipy.special import betainc

from .._version import __version__
from ..anisotropy import Anisotropy
from ..exceptions import InputError
from ..identity import identity_string
from ..profiles import DoublePowerLaw, TruncatedProfile
from ..settings import Settings
from ..typing import Array, ArrayLike
from ..validation import ValidationReport
from .inputs import Component
from .results import Model
from .stages import AsymptoticForm, DfTable
from .truncation import K_FLOOR

# The prefactor of the density integral of an isotropic DF:
# rho(P) = 4 pi sqrt(2) INT_0^P F(K) (P - K)^(1/2) dK.
ISOTROPIC_DENSITY_PREFACTOR = 4.0 * math.pi * math.sqrt(2.0)
# The amplitude A of the Hernquist DF at the center, F -> A (1 - K)^(-5/2).
HERNQUIST_CENTRAL_AMPLITUDE = 3.0 / (4.0 * math.sqrt(2.0) * math.pi)
# The amplitude A of the Plummer DF F = A K^(7/2), from the density 3 P^5
# through the density integral.
PLUMMER_AMPLITUDE = 3.0 / (ISOTROPIC_DENSITY_PREFACTOR * beta_function(4.5, 1.5))

ANALYTIC_MODELS = ("hernquist", "plummer")
# The DF table of an analytic model runs from the energy K_FLOOR to the depth
# U_TOP below the central potential 1. The profile of an analytic model runs
# from the radius R_MIN to the radius R_MAX, in internal units.
U_TOP = 1e-9
R_MIN, R_MAX = 1e-6, 1e6
# The remaining fields of the asymptotic form of a one-component DF whose
# tail below the table is a pure power law, and whose center above the table
# has no logarithm and no second singular term.
_NO_FURTHER_TERM: dict = dict(
    central_log_coefficient=(math.nan,),
    central_next_form=("none",),
    central_next_exponent=(math.nan,),
    central_next_coefficient=(math.nan,),
    tail_log_scale=(1.0,),
    below_floor=("tail",),
    below_floor_coefficients=((),),
    # The closed-form table needs no anchor, because the closed-form table
    # reaches the depth U_TOP below the center.
    central_anchor=(math.nan,),
    central_anchor_amplitude=(math.nan,),
    central_regular=((math.nan, math.nan),),
)


def hernquist_df(k: ArrayLike) -> tuple[Array, Array]:
    r"""Return the DF of the isotropic Hernquist model and its derivative.

    Parameters
    ----------
    k : array-like
        Energy :math:`\mathcal{K}` in internal units, with
        :math:`0 \leq \mathcal{K} < 1`.

    Returns
    -------
    df : array
        The DF :math:`F(\mathcal{K})`, in internal units, with the shape of
        `k`.
    ddf_dk : array
        The derivative :math:`\mathrm{d}F/\mathrm{d}\mathcal{K}`, in internal
        units, with the shape of `k`.

    Notes
    -----
    In internal units the Hernquist model has the potential
    :math:`P = 1/(1+R)` and the density :math:`\varrho = 2/[R(1+R)^3]`. The
    DF of Hernquist (1990) is written as

    .. math::

        F(\mathcal{K}) = \frac{3}{4\sqrt{2}\,\pi}\,
            \frac{I_\mathcal{K}(5/2, 5/2)}{(1-\mathcal{K})^{5/2}} ,

    with :math:`I` the regularized incomplete Beta function. Every term of
    the incomplete-Beta form of :math:`F` and of
    :math:`\mathrm{d}F/\mathrm{d}\mathcal{K}` is positive, so neither
    :math:`F` nor :math:`\mathrm{d}F/\mathrm{d}\mathcal{K}` loses precision
    toward :math:`\mathcal{K} = 0`. The textbook form of the
    Hernquist DF subtracts nearly equal terms toward :math:`\mathcal{K} = 0`.

    Examples
    --------
    >>> df, ddf_dk = hernquist_df(0.5)
    >>> round(float(df), 12), round(float(ddf_dk), 12)
    (0.477464829276, 4.008463084656)
    """
    k = np.asarray(k, dtype=np.float64)
    u = 1.0 - k
    # The regularized incomplete Beta function I_x(5/2, 5/2) satisfies
    # I_x = 1 - I_(1-x). Above K = 0.5 the value is formed from the
    # complement, so betainc is always evaluated where betainc is small.
    fraction = np.where(k <= 0.5, betainc(2.5, 2.5, k), 1.0 - betainc(2.5, 2.5, u))
    df = HERNQUIST_CENTRAL_AMPLITUDE * fraction * u**-2.5
    ddf_dk = HERNQUIST_CENTRAL_AMPLITUDE * (
        k**1.5 / (beta_function(2.5, 2.5) * u) + 2.5 * fraction * u**-3.5
    )
    return df, ddf_dk


def plummer_df(k: ArrayLike) -> tuple[Array, Array]:
    r"""Return the DF of the isotropic Plummer model and its derivative.

    Parameters
    ----------
    k : array-like
        Energy :math:`\mathcal{K}` in internal units, with
        :math:`0 \leq \mathcal{K} \leq 1`.

    Returns
    -------
    df : array
        The DF :math:`F(\mathcal{K})`, in internal units, with the shape of
        `k`.
    ddf_dk : array
        The derivative :math:`\mathrm{d}F/\mathrm{d}\mathcal{K}`, in internal
        units, with the shape of `k`.

    Notes
    -----
    In internal units the Plummer model has the potential
    :math:`P = (1+R^2)^{-1/2}` and the density :math:`\varrho = 3 P^5`. The
    Plummer model is a polytrope, with the DF
    :math:`F = A\,\mathcal{K}^{7/2}` and
    :math:`A = 3/[4\pi\sqrt{2}\,B(9/2, 3/2)]`, :math:`B` the Beta
    function.

    Examples
    --------
    >>> df, ddf_dk = plummer_df(1.0)
    >>> round(float(df), 12), round(float(ddf_dk), 12)
    (1.965117148288, 6.877910019009)
    """
    k = np.asarray(k, dtype=np.float64)
    return PLUMMER_AMPLITUDE * k**3.5, 3.5 * PLUMMER_AMPLITUDE * k**2.5


def analytic_model(name: str, n_nodes: int = 2001) -> Model:
    r"""Return the untruncated isotropic model of a profile with a closed-form DF.

    The model is filled from closed forms only. The DF table comes from
    `hernquist_df` or `plummer_df`, and the model's profile comes from the
    `DoublePowerLaw` density profile. The model is untruncated and therefore
    not spatially finite. The model's validation report holds no checks.

    Parameters
    ----------
    name : {"hernquist", "plummer"}
        The profile: the Hernquist profile or the Plummer profile.
    n_nodes : int, optional
        The number of energies of the DF table, and the number of radii of
        the profile, at least 5. Default: ``2001``. The energies are uniform
        in :math:`\ln \mathcal{K}` from ``K_FLOOR`` (:math:`10^{-10}`) to
        0.5, and uniform in the logarithm of the depth
        :math:`1 - \mathcal{K}` from 0.5 to ``U_TOP`` (:math:`10^{-9}`).
        The radii are uniform in :math:`\ln R` from ``R_MIN``
        (:math:`10^{-6}`) to ``R_MAX`` (:math:`10^{6}`), in internal units.

    Returns
    -------
    model : Model
        The untruncated model, with one component, the default settings
        (`Settings`), the closed-form DF table, and the asymptotic form of
        the DF at the center and below the table.

    Raises
    ------
    InputError
        If ``name`` is neither "hernquist" nor "plummer".
    InputError
        If ``n_nodes`` is below 5.

    Examples
    --------
    >>> model = analytic_model("plummer", n_nodes=101)
    >>> model.spatially_finite, model.validation.tier
    (False, 'silent')
    """
    if name not in ANALYTIC_MODELS:
        raise InputError(
            f"The models with a closed-form DF are {list(ANALYTIC_MODELS)}; "
            f"got {name!r}."
        )
    if n_nodes < 5:
        raise InputError(f"n_nodes must be at least 5; got {n_nodes}.")
    if name == "hernquist":
        profile = DoublePowerLaw.hernquist()
        df_function = hernquist_df
        # The Hernquist DF tends to A u^(-5/2) at the center, with u = 1 - K
        # and A = HERNQUIST_CENTRAL_AMPLITUDE.
        # At the outer end the Hernquist DF tends to
        # (64 / (5 sqrt(2) pi^2)) K^(5/2), from the density 2 P^4 at small P.
        form = AsymptoticForm(
            p_0=1.0,
            central_label=("regular",),
            central_exponent=(-2.5,),
            central_amplitude=(HERNQUIST_CENTRAL_AMPLITUDE,),
            tail_exponent=(-2.5,),
            tail_amplitude=(
                2.0 / (ISOTROPIC_DENSITY_PREFACTOR * beta_function(3.5, 1.5)),
            ),
            tail_log_exponent=(0.0,),
            central_form=("power",),
            **_NO_FURTHER_TERM,
        )
    else:
        profile = DoublePowerLaw.plummer()
        df_function = plummer_df
        form = AsymptoticForm(
            p_0=1.0,
            central_label=("critical_analytic",),
            central_exponent=(0.0,),
            central_amplitude=(math.nan,),
            tail_exponent=(-3.5,),
            tail_amplitude=(PLUMMER_AMPLITUDE,),
            tail_log_exponent=(0.0,),
            central_form=("analytic",),
            **_NO_FURTHER_TERM,
        )
    # The energies are uniform in ln K up to K = 0.5, and uniform in the
    # logarithm of the depth 1 - K above K = 0.5, so that the nodes resolve
    # both the tail at small K and the center at small depth.
    n_lower = n_nodes // 2 + 1
    k_lower = np.geomspace(K_FLOOR, 0.5, n_lower)
    u_upper = np.geomspace(0.5, U_TOP, n_nodes - n_lower + 1)[1:]
    k = np.concatenate([k_lower, 1.0 - u_upper])
    df, ddf_dk = df_function(k)
    table = DfTable(
        k=k[np.newaxis, :],
        log_df=np.log10(df)[np.newaxis, :],
        log_df_slope=(k * ddf_dk / (df * math.log(10.0)))[np.newaxis, :],
        i_first=(0,),
        i_last=(n_nodes - 1,),
        slopes_exact=(True,),
    )
    settings = Settings()
    inputs = {"analytic_model": name, "n_nodes": int(n_nodes)}
    identity = identity_string(inputs, settings.to_dict(), __version__)
    r = np.geomspace(R_MIN, R_MAX, n_nodes)
    truncated_profile = TruncatedProfile(
        r=r,
        p_trunc=np.asarray(profile.p(r)),
        rho_trunc=np.asarray(profile.rho(r))[np.newaxis, :],
        mass_trunc=np.asarray(profile.mass(r)),
        k_t=0.0,
        r_t=math.inf,
        mass_t=float(profile.mass_total),
        spatially_finite=False,
        source=profile,
        source_anisotropy=Anisotropy(),
        model=identity,
        edge_exponent=(),
    )
    return Model(
        components=(Component(profile, Anisotropy(), name=name),),
        truncation=None,
        settings=settings,
        identity=identity,
        parent=identity_string(
            dict(inputs, kind="distribution_function"), settings.to_dict(), __version__
        ),
        table=table,
        form=form,
        profile=truncated_profile,
        validation=ValidationReport(keys=(), checks={}),
        timing={},
        # A closed-form model meets every accuracy bound at every radius, so the
        # radius of trust is infinite for the "silent" and "recorded" tiers.
        # A spatially finite model is trusted to its edge and states no
        # radius of trust.
        r_trust={}
        if truncated_profile.spatially_finite
        else {"silent": math.inf, "recorded": math.inf},
        inversion=None,  # A closed-form model has no inversion record.
    )
