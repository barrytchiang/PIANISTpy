"""Shape-preserving cubic Hermite interpolation (PCHIP) in JAX.

The interpolant of the truncation's lookups: jit and vmap safe, with the node
arrays taken as closure constants at the call sites, so that the slopes are
traced once per kernel.

* ``order="cubic"`` evaluates the piecewise cubic Hermite polynomial with
  Fritsch-Carlson slopes (shape preserving: no overshoot), or with the slopes
  ``m`` the caller gives in the interpolation space. PCHIP slopes are second
  order at best and zero at a local extremum; exact slopes make the
  interpolant fourth order.
* ``order="linear"`` is ``jnp.interp`` in linear x and f, whatever ``logx``,
  ``logf`` and ``m``.
* ``logx``, ``logf``: interpolate in the logarithm of x, of f (f > 0). Every
  argument of a logarithm is clamped at ``floor`` first; a fill value of 0
  with ``logf`` comes back as ``floor``.
* Out of range (cubic): ``left`` below the first node, ``right`` above the
  last; a query on an end node is interpolated.
"""

__all__ = ["hermite_eval", "interp1d", "pchip_slopes"]

import jax.numpy as jnp

from ..exceptions import InputError


def pchip_slopes(xp, fp):
    """Return the Fritsch-Carlson slopes at the nodes.

    The convention is SciPy's ``PchipInterpolator``: the weighted harmonic
    mean of the two secants where they have one sign and zero otherwise, and
    the shape-preserving three-point formula at each end.

    Parameters
    ----------
    xp, fp : array, shape (n,)
        Increasing nodes and their values, n >= 3.

    Returns
    -------
    m : array, shape (n,)

    Raises
    ------
    InputError
        If there are fewer than three nodes.
    """
    if xp.shape[0] < 3:
        raise InputError(
            f"PCHIP slopes need >= 3 nodes, got {xp.shape[0]} "
            "(pass m or use order='linear')"
        )
    h = xp[1:] - xp[:-1]
    d = (fp[1:] - fp[:-1]) / h
    hk, hk1 = h[:-1], h[1:]
    dk, dk1 = d[:-1], d[1:]
    w1 = 2.0 * hk1 + hk
    w2 = hk1 + 2.0 * hk
    same = (dk * dk1) > 0.0
    safe_dk = jnp.where(same, dk, 1.0)
    safe_dk1 = jnp.where(same, dk1, 1.0)
    m_int = jnp.where(same, (w1 + w2) / (w1 / safe_dk + w2 / safe_dk1), 0.0)

    def _end(h0, h1, d0, d1):
        m = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        m = jnp.where(jnp.sign(m) != jnp.sign(d0), 0.0, m)
        m = jnp.where(
            (jnp.sign(d0) != jnp.sign(d1)) & (jnp.abs(m) > 3.0 * jnp.abs(d0)),
            3.0 * d0,
            m,
        )
        return m

    m0 = _end(h[0], h[1], d[0], d[1])
    mN = _end(h[-1], h[-2], d[-1], d[-2])
    return jnp.concatenate([m0[None], m_int, mN[None]])


def hermite_eval(x, xp, fp, m, left, right):
    """Return the cubic Hermite interpolant of (xp, fp, m) at x.

    Below the first node it returns ``left``, above the last ``right``.
    """
    i = jnp.clip(jnp.searchsorted(xp, x, side="right") - 1, 0, xp.shape[0] - 2)
    x0, x1 = xp[i], xp[i + 1]
    h = x1 - x0
    t = (x - x0) / h
    t2, t3 = t * t, t * t * t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    y = h00 * fp[i] + h10 * h * m[i] + h01 * fp[i + 1] + h11 * h * m[i + 1]
    y = jnp.where(x < xp[0], left, y)
    y = jnp.where(x > xp[-1], right, y)
    return y


def interp1d(
    x, xp, fp, left, right, order="cubic", logx=False, logf=False, m=None, floor=1e-300
):
    """Return the interpolant of (xp, fp) at x, as ``jnp.interp`` with the
    choices of the module docstring.

    Parameters
    ----------
    x : array
        Queries.
    xp, fp : array, shape (n,)
        Increasing nodes and their values.
    left, right : float
        Values below the first node and above the last, in f.
    order : {"cubic", "linear"}
    logx, logf : bool
        Interpolate in log x, in log f.
    m : array, shape (n,), optional
        Slopes at the nodes in the interpolation space; PCHIP's where None.
    floor : float
        Clamp before every logarithm.
    """
    if order == "linear":
        return jnp.interp(x, xp, fp, left=left, right=right)
    xq = jnp.log(jnp.maximum(x, floor)) if logx else x
    xg = jnp.log(jnp.maximum(xp, floor)) if logx else xp
    fg = jnp.log(jnp.maximum(fp, floor)) if logf else fp
    slopes = pchip_slopes(xg, fg) if m is None else m
    lf = jnp.log(jnp.maximum(left, floor)) if logf else left
    rf = jnp.log(jnp.maximum(right, floor)) if logf else right
    y = hermite_eval(xq, xg, fg, slopes, lf, rf)
    return jnp.exp(y) if logf else y
