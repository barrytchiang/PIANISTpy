r"""The inverse :math:`h = 1/g` of the anisotropy factor and its derivatives.

The inverse factor is :math:`h = R^{2 \beta_{\rm c}} u^s`, with :math:`R` the
radius, :math:`\beta_{\rm c}` the central anisotropy,
:math:`u = 1 + R^2 / R_{\rm a}^2` for Type I and
:math:`u = 1 - R^2 / R_{\rm a}^2` for Type II, and :math:`R_{\rm a}` the
anisotropy radius. The exponent :math:`s` is 1 for an OM model,
:math:`1 - \beta_{\rm c}` for a COM model, and 0 for the isotropic and the
constant-anisotropy models. Every use of the factor in the package reads
:math:`h` from this module: `Anisotropy.factor` (as :math:`g = 1/h`), the
Method II sources, the edge law, the density of the Poisson solve, and the
symbolic derivatives of the Method I inversion (through
`inverse_factor_expr`).

A Type II model is defined only inside :math:`R_{\rm a}`, where
:math:`u > 0`. At a radius equal to or larger than :math:`R_{\rm a}`,
`inverse_factor` raises InputError. `inverse_factor_expr` does not check the
radius. The symbolic derivatives of the Method I inversion of an untruncated
Type II model pass ``continuation=True`` to `inverse_factor_expr`, which
then returns the continuation
:math:`R^{2 \beta_{\rm c}} \, {\rm sign}(u) \, |u|^{1 - \beta_{\rm c}}` of a
COM Type II factor beyond :math:`R_{\rm a}`. `inverse_factor` does not check
a radius traced by JAX. Every caller that traces the radius evaluates
:math:`h` only at radii inside :math:`R_{\rm a}`.
"""

__all__ = ["factor_exponents", "inverse_factor", "inverse_factor_expr"]

from typing import Any

import jax
import numpy as np

from ..exceptions import InputError

_TYPE_II = ("OM_II", "COM_II")


def factor_exponents(family: str, beta_c: float) -> tuple[float, float]:
    r"""Return the sign of :math:`R^2 / R_{\rm a}^2` in :math:`u`, and :math:`s`.

    The sign is -1 for a Type II family and +1 otherwise. The exponent
    :math:`s` of :math:`h = R^{2 \beta_{\rm c}} u^s` is 1 for an OM family,
    :math:`1 - \beta_{\rm c}` for a COM family, and 0 for the isotropic and
    the constant families.
    """
    sign = -1.0 if family in _TYPE_II else 1.0
    if family in ("OM_I", "OM_II"):
        s = 1.0
    elif family in ("COM_I", "COM_II"):
        s = 1.0 - beta_c
    else:
        s = 0.0
    return sign, s


def inverse_factor(
    r: Any, family: str, beta_c: float, r_a: float, order: int = 0
) -> tuple[Any, ...]:
    r"""Return :math:`h = R^{2 \beta_{\rm c}} u^s` and its derivatives up to ``order``.

    Parameters
    ----------
    r : float or array
        Radius :math:`R`, in internal units: a float, a NumPy array or a JAX
        array.
    family : str
        Family of the anisotropy, as `Anisotropy.family`.
    beta_c : float
        Central anisotropy :math:`\beta_{\rm c}`; 0 for an OM model.
    r_a : float
        Anisotropy radius :math:`R_{\rm a}`, in internal units; ``math.inf``
        for a model without an anisotropy radius.
    order : int, optional
        Highest derivative in :math:`R` returned, from 0 to 3. The third
        derivative is formed by sympy. Default: ``0``, which means
        :math:`h` alone.

    Returns
    -------
    h : tuple
        The tuple (h, h_R, h_RR, h_RRR) of :math:`h` and its first three
        derivatives in :math:`R`, cut after the derivative of order
        ``order``.

    Raises
    ------
    InputError
        If ``order`` is negative or larger than 3.
    InputError
        If the model is Type II and a radius in ``r`` is equal to or larger
        than :math:`R_{\rm a}`, where :math:`h` is not defined. A radius
        traced by JAX is not checked.
    """
    if not 0 <= order <= 3:
        raise InputError(f"Derivatives of the factor of order {order}: up to 3.")
    b, ra = beta_c, r_a
    sgn, s_om = factor_exponents(family, b)
    u = 1.0 + sgn * (r / ra) ** 2
    if family in _TYPE_II and not isinstance(u, jax.core.Tracer):
        outside = ~(np.asarray(u) > 0.0)
        if np.any(outside):
            r_out = np.broadcast_to(np.asarray(r, dtype=np.float64), outside.shape)
            raise InputError(
                f"A Type II model's radius R = {float(r_out[outside].flat[0])!r} lies "
                f"at or beyond R_a = {float(ra)!r}: the anisotropy factor is not "
                "defined there."
            )
    h = r ** (2 * b) * u**s_om
    if order == 0:
        return (h,)
    du = sgn * 2.0 * r / ra**2
    ddu = sgn * 2.0 / ra**2
    h_r = (
        2 * b * r ** (2 * b - 1) * u**s_om + r ** (2 * b) * s_om * u ** (s_om - 1) * du
    )
    if order == 1:
        return h, h_r
    h_rr = (
        2 * b * (2 * b - 1) * r ** (2 * b - 2) * u**s_om
        + 2 * (2 * b * r ** (2 * b - 1)) * s_om * u ** (s_om - 1) * du
        + r ** (2 * b)
        * (s_om * (s_om - 1) * u ** (s_om - 2) * du**2 + s_om * u ** (s_om - 1) * ddu)
    )
    if order == 2:
        return h, h_r, h_rr
    import sympy

    r_sym = sympy.symbols("R", positive=True)
    h_rrr = np.asarray(
        sympy.lambdify(
            r_sym,
            sympy.diff(
                r_sym ** (2 * b) * (1 + sgn * r_sym**2 / ra**2) ** s_om, r_sym, 3
            ),
            "numpy",
        )(r),
        dtype=np.float64,
    )
    return h, h_r, h_rr, h_rrr


def inverse_factor_expr(
    r: Any, family: str, beta_c: Any, r_a: Any, *, continuation: bool = False
) -> Any:
    r"""Return :math:`h = R^{2 \beta_{\rm c}} u^s` as an exact sympy expression.

    Parameters
    ----------
    r : sympy expression
        Radius :math:`R`.
    family : str
        Family of the anisotropy, as `Anisotropy.family`.
    beta_c : sympy expression
        Central anisotropy :math:`\beta_{\rm c}`.
    r_a : sympy expression
        Anisotropy radius :math:`R_{\rm a}`. Not read for the isotropic and
        the constant families.
    continuation : bool, optional
        If True and the family is ``"COM_II"``, the factor
        :math:`u^{1 - \beta_{\rm c}}` is replaced by its continuation
        :math:`{\rm sign}(u) \, |u|^{1 - \beta_{\rm c}}`, which is defined
        beyond :math:`R_{\rm a}`. Only the symbolic derivatives of the
        Method I inversion set ``continuation=True``. Default: ``False``,
        which means :math:`u^{1 - \beta_{\rm c}}`.

    Returns
    -------
    h : sympy expression
        The inverse factor :math:`h` in terms of `r`, `beta_c` and `r_a`.
    """
    import sympy as sp

    h = r ** (2 * beta_c)
    if family not in ("OM_I", "OM_II", "COM_I", "COM_II"):
        return h
    u = 1 - r**2 / r_a**2 if family in _TYPE_II else 1 + r**2 / r_a**2
    if family.startswith("OM"):
        return h * u
    if family == "COM_II" and continuation:
        return h * u * sp.exp(-beta_c * sp.log(u**2) / 2)
    return h * u ** (1 - beta_c)
