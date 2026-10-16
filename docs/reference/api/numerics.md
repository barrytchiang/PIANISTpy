# Numerics (pianistpy.numerics)

Source: https://pianistpy.com/docs/reference/api/numerics.html

# Numerics (pianistpy.numerics)

Shared numerical leaves: quadrature, interpolation and special functions.

pianistpy.numerics.bell_partitions(order)

Return the terms of the complete exponential Bell polynomial.

Parameters:

order (int) – Order of the polynomial, at least 1.

Returns:

terms – One entry per partition of order into positive integers: its
coefficient and the parts, in decreasing order. The polynomial is
the sum over the entries of the coefficient times the product of the
variables indexed by the parts.

Return type:

tuple of (int, tuple of int)

Examples

```
>>> bell_partitions(3)
((1, (3,)), (3, (2, 1)), (1, (1, 1, 1)))
>>> sum(coefficient for coefficient, _ in bell_partitions(4))
15

```

pianistpy.numerics.derivative_from_log(log_derivatives, values)

Return a derivative of a positive function from those of its logarithm.

Parameters:

log_derivatives (sequence of array, length N) – The derivatives of orders 1 to N of the logarithm of the function,
each at the same points.

values (array) – The function itself at those points.

Returns:

derivative – The derivative of order N of the function.

Return type:

array

Notes

With \(S = \ln f\), the chain rule of Faà di Bruno gives

\[\frac{\mathrm{d}^N f}{\mathrm{d}x^N}
    = f\,Y_N\!\left(S', S'', \dots, S^{(N)}\right),\]

where \(Y_N\) is the complete exponential Bell polynomial.

Examples

```
>>> import numpy as np
>>> x = np.array([0.5, 1.0])
>>> f = np.exp(3.0 * x)
>>> ones, zeros = np.ones_like(x), np.zeros_like(x)
>>> bool(np.allclose(derivative_from_log([3 * ones, zeros], f), 9.0 * f))
True

```

pianistpy.numerics.gauss_legendre(n_nodes)

Return the Gauss-Legendre nodes and weights on [-1, 1].

The nodes of a given count are computed once per session and returned as
read-only arrays, so that no caller can change them for another.

Parameters:

n_nodes (int) – Number of nodes, at least 1.

Returns:

nodes (array, shape (n_nodes,)) – Nodes in increasing order.

weights (array, shape (n_nodes,)) – Weights; they sum to 2.

Return type:

tuple[Any, Any]

Examples

```
>>> nodes, weights = gauss_legendre(2)
>>> float(weights.sum())
2.0

```

pianistpy.numerics.gauss_legendre_unit(n_nodes)

Return the Gauss-Legendre nodes and weights on [0, 1].

Parameters:

n_nodes (int) – Number of nodes, at least 1.

Returns:

nodes (array, shape (n_nodes,)) – Nodes in increasing order.

weights (array, shape (n_nodes,)) – Weights; they sum to 1.

Return type:

tuple[Any, Any]

pianistpy.numerics.hermite(x, x_nodes, f_nodes, df_dx_nodes)

Return the cubic Hermite interpolant and its derivative.

Between two nodes the interpolant is the cubic that takes the values and
the slopes given at both. Outside the nodes it continues as the straight
line through the end node with the end slope.

Parameters:

x (array-like) – Points of evaluation.

x_nodes (array, shape (n,)) – Nodes, strictly increasing.

f_nodes (array, shape (n,)) – Values at the nodes.

df_dx_nodes (array, shape (n,)) – Slopes at the nodes.

Returns:

f (array) – The interpolant at x.

df_dx (array) – Its derivative at x.

Return type:

tuple[Any, Any]

Examples

A cubic is reproduced exactly from its values and slopes at two nodes:

```
>>> import numpy as np
>>> x_nodes = np.array([0.0, 2.0])
>>> f, df_dx = hermite(1.0, x_nodes, x_nodes**3, 3.0 * x_nodes**2)
>>> float(f), float(df_dx)
(1.0, 3.0)

```

pianistpy.numerics.hermite_inverse(f, x_nodes, f_nodes, df_dx_nodes)

Return where an increasing cubic Hermite interpolant takes given values.

The root is bracketed by the two nodes whose values enclose the target,
or by the straight continuation beyond the end nodes, and found by
bisection. Newton steps then refine it; a step that would leave the
bracket is not taken, so a locally steep interpolant cannot throw a root
out of its interval.

Parameters:

f (array-like) – Values to invert for.

x_nodes (array, shape (n,)) – Nodes, strictly increasing.

f_nodes (array, shape (n,)) – Values at the nodes, increasing.

df_dx_nodes (array, shape (n,)) – Slopes at the nodes, positive at the two end nodes.

Returns:

x (array) – Points at which the interpolant equals f.

residual (float) – Largest absolute difference between the interpolant at x and f.

Return type:

tuple[Any, float]

Examples

```
>>> import numpy as np
>>> x_nodes = np.array([0.0, 1.0, 2.0])
>>> x, residual = hermite_inverse(1.5, x_nodes, 2.0 * x_nodes, np.full(3, 2.0))
>>> float(x), residual
(0.75, 0.0)

```

     On this page
