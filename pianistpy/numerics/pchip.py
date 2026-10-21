"""Shape-preserving piecewise cubic Hermite interpolation (PCHIP) in JAX.

`interp1d` is the interpolant that the truncation's lookups use. The functions
of this module run under ``jax.jit`` and ``jax.vmap``. The call sites pass
the node arrays as closure constants, so that JAX traces the PCHIP slopes
once per compiled kernel. The options of `interp1d` are:

* ``order="cubic"`` evaluates the piecewise cubic Hermite polynomial. The
  slopes at the nodes are the Fritsch-Carlson slopes, which preserve the
  data's monotonicity (no overshoot), unless the caller passes slopes ``m``
  in the interpolation space. The Fritsch-Carlson slopes are accurate to
  second order at best and are zero at a local extremum of the data. Exact
  slopes ``m`` make the interpolant accurate to fourth order.
* ``order="linear"`` is ``jnp.interp`` in linear x and linear f, and ignores
  ``logx``, ``logf`` and ``m``.
* ``logx`` and ``logf`` select interpolation in the logarithm of x and in the
  logarithm of f (which requires f > 0). Every argument of a logarithm is
  first clamped from below at ``floor``. With ``logf``, a ``left`` or
  ``right`` of 0 therefore comes back as ``floor``.
* Outside the nodes, the cubic order returns ``left`` below the first node
  and ``right`` above the last node. A query on an end node is interpolated.
"""

__all__ = ["hermite_eval", "interp1d", "pchip_slopes"]

import jax.numpy as jnp

from ..exceptions import InputError


def pchip_slopes(xp, fp):
    """Return the Fritsch-Carlson slopes at the nodes.

    The slopes follow the convention of SciPy's ``PchipInterpolator``. At an
    interior node, the slope is the weighted harmonic mean of the two
    neighboring secants when both secants have the same sign, and zero
    otherwise. At each end node, the slope is the shape-preserving
    three-point formula.

    Parameters
    ----------
    xp : array, shape (n,)
        Nodes, strictly increasing, with n of 3 or more.
    fp : array, shape (n,)
        Values at the nodes.

    Returns
    -------
    m : array, shape (n,)
        Slopes df/dx at the nodes.

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


def hermite_eval(x, xp, fp, m, left, right, n=None):
    """Return the cubic Hermite interpolant of nodes xp, values fp and slopes m at x.

    The interpolant is ``left`` below the first node and ``right`` above the
    last node. When ``n`` is given, only the first ``n`` nodes are read, and
    the remaining entries of ``xp``, ``fp`` and ``m`` are padding that repeats
    the last node. The interpolant with padding then equals the interpolant
    of the first ``n`` nodes alone.
    """
    top = xp.shape[0] - 2 if n is None else n - 2
    i = jnp.clip(jnp.searchsorted(xp, x, side="right") - 1, 0, top)
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
    x,
    xp,
    fp,
    left,
    right,
    order="cubic",
    logx=False,
    logf=False,
    m=None,
    floor=1e-300,
    n=None,
):
    """Return the interpolant of nodes xp and values fp at x.

    The call takes the arguments of ``jnp.interp`` plus the options that the
    module docstring describes.

    Parameters
    ----------
    x : array
        Query points.
    xp : array, shape (n,)
        Nodes, strictly increasing.
    fp : array, shape (n,)
        Values at the nodes.
    left : float
        Value of f below the first node.
    right : float
        Value of f above the last node.
    order : {"cubic", "linear"}, optional
        Piecewise cubic Hermite or piecewise linear interpolation. Default:
        ``"cubic"``.
    logx : bool, optional
        If True, interpolate in log x. Default: ``False``.
    logf : bool, optional
        If True, interpolate in log f. Default: ``False``.
    m : array, shape (n,), optional
        Slopes at the nodes in the interpolation space. Default: ``None``,
        which means the PCHIP slopes of `pchip_slopes`.
    floor : float, optional
        Lower clamp of every argument of a logarithm. Default: ``1e-300``.
    n : int or traced int, optional
        Number of nodes that are read; the remaining entries of ``xp`` are
        padding that repeats the last node, as in `hermite_eval`. Allowed
        only with ``order="cubic"`` and slopes ``m``. Default: ``None``,
        which means every node is read.

    Returns
    -------
    f : array
        The interpolant at ``x``, with the shape of ``x``.

    Raises
    ------
    InputError
        If ``n`` is given without slopes ``m`` or with ``order="linear"``.
    InputError
        If ``m`` is None, the order is cubic, and there are fewer than
        three nodes.
    """
    if n is not None and (m is None or order != "cubic"):
        raise InputError("padded nodes (n) are read by the cubic order with m only")
    if order == "linear":
        return jnp.interp(x, xp, fp, left=left, right=right)
    xq = jnp.log(jnp.maximum(x, floor)) if logx else x
    xg = jnp.log(jnp.maximum(xp, floor)) if logx else xp
    fg = jnp.log(jnp.maximum(fp, floor)) if logf else fp
    slopes = pchip_slopes(xg, fg) if m is None else m
    lf = jnp.log(jnp.maximum(left, floor)) if logf else left
    rf = jnp.log(jnp.maximum(right, floor)) if logf else right
    y = hermite_eval(xq, xg, fg, slopes, lf, rf, n)
    return jnp.exp(y) if logf else y
