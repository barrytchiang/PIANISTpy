r"""The inner series of the double-power-law density, its enclosed mass and its depth.

The density :math:`\rho = C r^{-\gamma} (1 + r^{\alpha})^{e}`, with
:math:`e = (\gamma - \beta) / \alpha`, :math:`\alpha` the transition slope,
:math:`\beta` the outer slope and :math:`\gamma` the inner slope, is expanded
near the center as :math:`C \sum_k b_k r^{k \alpha - \gamma}`, with :math:`b_k`
the binomial coefficient of :math:`e` over :math:`k`. `inner_density_series`
is the density's closed form, and `inner_mass_series` and `inner_depth_series`
the enclosed mass and the depth below the central potential of the first four
terms, summed term by term. The Poisson solver lays the innermost rows of a
model's profile with these forms (Notes Section T:3.4), and `TruncatedProfile`
continues its columns inside its first node with them. All quantities are in
internal units.
"""

__all__ = [
    "inner_density_series",
    "inner_depth_series",
    "inner_depth_terms",
    "inner_mass_series",
]

import numpy as np

from ..exceptions import InputError


def inner_density_series(r, c, gamma, alpha, beta):
    r"""Return the double-power-law density in closed form.

    The density is :math:`\rho(r) = C r^{-\gamma} (1 + r^{\alpha})^{e}`
    with :math:`e = (\gamma - \beta) / \alpha`, at the user-specified 3D
    radii ``r`` in internal units. The argument ``c`` is the amplitude
    :math:`C`, and ``gamma``, ``alpha`` and ``beta`` are :math:`\gamma`,
    :math:`\alpha` and :math:`\beta`.

    Raises
    ------
    InputError
        If :math:`\gamma` is not below 2.
    """
    if not gamma < 2.0:
        raise InputError(f"gamma = {gamma:.4f} >= 2: the enclosed mass diverges")
    r = np.asarray(r, dtype=np.float64)
    e = (gamma - beta) / alpha
    return c * (r ** (-gamma)) * (1.0 + r**alpha) ** e


def inner_mass_series(r, c, gamma, alpha, beta):
    r"""Return the enclosed mass of the first four terms of the inner density series.

    The density :math:`C r^{-\gamma} (1 + r^{\alpha})^{e}`, with
    :math:`e = (\gamma - \beta) / \alpha`, is expanded as
    :math:`C \sum_k b_k r^{k \alpha - \gamma}`, with :math:`b_k` the binomial
    coefficient of :math:`e` over :math:`k`. The enclosed mass of the terms
    :math:`k = 0` to 3 is
    :math:`m(r) = C \sum_k b_k r^{3 - \gamma + k \alpha} / (3 - \gamma + k \alpha)`,
    without the factor :math:`4 \pi`, in internal units.

    Raises
    ------
    InputError
        If :math:`\gamma` is not below 3.
    """
    if not gamma < 3.0:
        raise InputError(f"gamma = {gamma:.4f} >= 3: the mass diverges at the origin")
    r = np.asarray(r, dtype=np.float64)
    g, a = float(gamma), float(alpha)
    e = (g - float(beta)) / a
    acc = np.zeros_like(r)
    bk = 1.0
    for k in range(4):
        if k > 0:
            bk *= (e - (k - 1)) / k
        acc = acc + bk * r ** (3.0 - g + k * a) / (3.0 - g + k * a)
    return c * acc


def inner_depth_series(r, c, gamma, alpha, beta):
    r"""Return the depth of the first four terms of the inner density series.

    The depth below the central potential :math:`P_0` is
    :math:`u(r) = P_0 - \Phi(r)`, the integral of :math:`m(r') / r'^2` from
    0 to :math:`r`, with :math:`m` the enclosed mass of `inner_mass_series`:
    :math:`u(r) = C \sum_k b_k r^{2 - \gamma + k \alpha} /
    [(2 - \gamma + k \alpha) (3 - \gamma + k \alpha)]` for :math:`k = 0` to
    3, in internal units.

    Raises
    ------
    InputError
        If :math:`\gamma` is not below 2.
    """
    if not gamma < 2.0:
        raise InputError(f"gamma = {gamma:.4f} >= 2: the depth diverges at the origin")
    r = np.asarray(r, dtype=np.float64)
    g, a = float(gamma), float(alpha)
    e = (g - float(beta)) / a
    acc = np.zeros_like(r)
    bk = 1.0
    for k in range(4):
        if k > 0:
            bk *= (e - (k - 1)) / k
        acc = acc + bk * r ** (2.0 - g + k * a) / (
            (2.0 - g + k * a) * (3.0 - g + k * a)
        )
    return c * acc


def inner_depth_terms(c, gamma, alpha, beta, deficit=0.0, deficit_slope=0.0):
    r"""Return the coefficients and the exponents of the depth series' five terms.

    `inner_depth_series` is :math:`u(r) = \sum_k c_k r^{p_k}` for
    :math:`k = 0` to 3, with :math:`p_k = 2 - \gamma + k \alpha` and
    :math:`c_k = C b_k / [p_k (p_k + 1)]`. The fifth term is the depth of the
    central deficit's first-order term, :math:`-\delta_1 r^{2 - w} /
    [(3 - w)(2 - w)]` for a missing density :math:`\delta_1 r^{-w}`, with
    ``deficit`` the sum :math:`\delta_1` over the components and
    ``deficit_slope`` the exponent :math:`w = 2 \beta_{\rm c}`
    (`TruncatedProfile`); the fifth coefficient is zero where no deficit
    is carried. The two arrays of five values let a JAX function evaluate the
    series from traced arguments.

    Raises
    ------
    InputError
        If :math:`\gamma` is not below 2, or if ``deficit_slope`` is not
        below 2.
    """
    if not gamma < 2.0:
        raise InputError(f"gamma = {gamma:.4f} >= 2: the depth diverges at the origin")
    w = float(deficit_slope)
    if not w < 2.0:
        raise InputError(
            f"the deficit's slope {w:.4f} is not below 2: its depth diverges at "
            f"the origin"
        )
    g, a = float(gamma), float(alpha)
    e = (g - float(beta)) / a
    coefficients, exponents = np.zeros(5), np.zeros(5)
    bk = 1.0
    for k in range(4):
        if k > 0:
            bk *= (e - (k - 1)) / k
        p_k = 2.0 - g + k * a
        exponents[k] = p_k
        coefficients[k] = float(c) * bk / (p_k * (p_k + 1.0))
    exponents[4] = 2.0 - w
    coefficients[4] = -float(deficit) / ((3.0 - w) * (2.0 - w))
    return coefficients, exponents
