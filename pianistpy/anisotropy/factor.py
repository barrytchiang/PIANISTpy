r"""The inverse of the anisotropy factor, h = 1/g, and its R-derivatives.

``h = R^(2 beta_c) u^s`` with ``u = 1 +- R^2 / R_a^2`` (+ Type I, - Type II)
and ``s = 1`` (OM), ``1 - beta_c`` (COM) or 0 (isotropic and constant). Every
copy of the factor in the package reads it here: `Anisotropy.factor`
(g = 1/h), the Method II sources, the edge law, the Poisson solve's density
and the symbolic derivatives of the Method I inversion (`inverse_factor_expr`).

A Type II model is defined inside :math:`R_{\rm a}` only (u > 0). A radius at
or beyond it is refused by name, except in the symbolic form of the Method I
inversion of an untruncated Type II model, which asks for the continuation
sign(u) abs(u)^s by its explicit argument. A traced radius (under JAX) is not
checked: its callers read on the model's support.
"""

__all__ = ["factor_exponents", "inverse_factor", "inverse_factor_expr"]

from typing import Any

import jax
import numpy as np

from ..exceptions import InputError

_TYPE_II = ("OM_II", "COM_II")


def factor_exponents(family: str, beta_c: float) -> tuple[float, float]:
    """The sign of ``R^2 / R_a^2`` in u and the exponent s of
    ``h = R^(2 beta_c) u^s``."""
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
    r"""Return ``h = R^(2 beta_c) u^s`` and its R-derivatives up to ``order``.

    Parameters
    ----------
    r : float or array
        Radius, in internal units: a float, a NumPy array or a JAX array.
    family : str
        The anisotropy's family (`Anisotropy.family`).
    beta_c : float
        The central anisotropy (0 for OM).
    r_a : float
        The anisotropy radius (infinite where there is none).
    order : int, optional
        The highest R-derivative returned, 0 to 3; the third by sympy.

    Returns
    -------
    h : tuple
        (h, h_R, h_RR, h_RRR) up to ``order``.

    Raises
    ------
    InputError
        For a Type II model at a radius at or beyond :math:`R_{\rm a}`, where
        the factor is not defined; for an order above 3.
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
    r"""Return ``h = R^(2 beta_c) u^s`` as a sympy expression, in exact terms.

    Parameters
    ----------
    r, beta_c, r_a : sympy expressions
        The radius, the central anisotropy and the anisotropy radius.
    family : str
        The anisotropy's family.
    continuation : bool, optional
        For a COM Type II model, the continuation
        ``sign(u) abs(u)^(1 - beta_c)`` beyond :math:`R_{\rm a}` in place of
        ``u^(1 - beta_c)``: the Method I inversion of an untruncated Type II
        model asks for it, and no other caller.
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
