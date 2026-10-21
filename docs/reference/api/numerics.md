# Numerics

Source: https://pianistpy.com/docs/reference/api/numerics

Module `pianistpy.numerics`

Gauss-Legendre quadrature and cubic Hermite interpolation.

A user calls gauss_legendre or gauss_legendre_unit for the nodes
\(x_j\) and weights \(w_j\) of the \(n\)-point Gauss-Legendre
rule, which integrates a polynomial of degree up to \(2n - 1\) exactly:

\[\int_{-1}^{1} f(x)\,{\rm d}x \approx \sum_{j=1}^{n} w_j f(x_j).\]

A user calls hermite for the piecewise cubic interpolant that takes given
values and slopes at given nodes, and hermite_inverse for the points where
an increasing piecewise cubic Hermite interpolant takes given values.

pianistpy.numerics.gauss_legendre(*n_nodes*)

Return the nodes and weights of the Gauss-Legendre rule on `[-1, 1]`.

The nodes and weights of each node count are computed once per Python
session and then reused. The arrays are read-only, so that no caller can
change the nodes or weights that another caller receives.

Parameters:

**n_nodes** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – Number of nodes \(n\). n_nodes must be 1 or larger.

Returns:

- **nodes** (*array, shape (n_nodes,)*) – Nodes \(x_j\) in increasing order, read-only.

- **weights** (*array, shape (n_nodes,)*) – Weights \(w_j\), read-only. The weights sum to 2.

Raises:

[**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If n_nodes is smaller than 1.

Return type:

[tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[*Any*](https://docs.python.org/3/library/typing.html#typing.Any), [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)]

See also

[`gauss_legendre_unit`](https://pianistpy.com/docs/reference/api/numerics#pianistpy.numerics.gauss_legendre_unit)
The Gauss-Legendre rule mapped to `[0, 1]`.

Notes

The \(n\)-point rule integrates every polynomial of degree up to
\(2n - 1\) exactly:

\[\int_{-1}^{1} f(x)\,{\rm d}x \approx \sum_{j=1}^{n} w_j f(x_j).\]

Examples

The 2-point rule integrates \(x^2\) over `[-1, 1]` to 2/3:

```python
>>> from pianistpy.numerics import gauss_legendre
>>> nodes, weights = gauss_legendre(2)
>>> float(weights.sum())
2.0
>>> round(float(weights @ nodes**2), 15)
0.666666666666667
```

pianistpy.numerics.gauss_legendre_unit(*n_nodes*)

Return the nodes and weights of the Gauss-Legendre rule on `[0, 1]`.

The nodes and weights are those of gauss_legendre mapped from
`[-1, 1]` to `[0, 1]`. The nodes and weights of each node count are
computed once per Python session and then reused, as read-only arrays.

Parameters:

**n_nodes** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) – Number of nodes \(n\). n_nodes must be 1 or larger.

Returns:

- **nodes** (*array, shape (n_nodes,)*) – Nodes \(t_j = (x_j + 1)/2\) in increasing order, read-only, with
\(x_j\) the nodes of gauss_legendre.

- **weights** (*array, shape (n_nodes,)*) – Weights \(w_j/2\), read-only, with \(w_j\) the weights of
gauss_legendre. The weights sum to 1.

Raises:

[**InputError**](https://pianistpy.com/docs/reference/api/exceptions#pianistpy.exceptions.InputError) – If n_nodes is smaller than 1.

Return type:

[tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[*Any*](https://docs.python.org/3/library/typing.html#typing.Any), [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)]

See also

[`gauss_legendre`](https://pianistpy.com/docs/reference/api/numerics#pianistpy.numerics.gauss_legendre)
The Gauss-Legendre rule on `[-1, 1]`.

Notes

The \(n\)-point rule integrates every polynomial of degree up to
\(2n - 1\) exactly:

\[\int_{0}^{1} f(t)\,{\rm d}t \approx \sum_{j=1}^{n} \frac{w_j}{2}\,
f(t_j).\]

Examples

The 1-point rule on `[0, 1]` has its node at 1/2 and its weight 1:

```python
>>> from pianistpy.numerics import gauss_legendre_unit
>>> nodes, weights = gauss_legendre_unit(1)
>>> nodes.tolist(), weights.tolist()
([0.5], [1.0])
```

pianistpy.numerics.hermite(*x*, *x_nodes*, *f_nodes*, *df_dx_nodes*)

Return the cubic Hermite interpolant and its derivative at given points.

Between two neighboring nodes, the interpolant is the cubic polynomial
that takes the given values and the given slopes at both nodes. Below
the first node and above the last node, the interpolant is the straight
line through the end node with the slope given at the end node. Between
the first and the last node, the interpolant reproduces every cubic
polynomial exactly.

Parameters:

- **x** ([*float*](https://docs.python.org/3/builtins/functions.html#float)* or **array-like*) – Points \(x\) at which the interpolant is evaluated, in the unit
of x_nodes.

- **x_nodes** (*array**, **shape** (**n**,**)*) – Nodes \(x_i\), strictly increasing, with \(n \ge 2\).

- **f_nodes** (*array**, **shape** (**n**,**)*) – Values \(f_i\) of the interpolant at the nodes, in the unit of
\(f\).

- **df_dx_nodes** (*array**, **shape** (**n**,**)*) – Slopes \(f'_i\) of the interpolant at the nodes, in the unit of
f_nodes per unit of x_nodes.

Returns:

- **f** (*array*) – The interpolant at x, with the shape of x.

- **df_dx** (*array*) – The derivative of the interpolant in \(x\) at x, with the
shape of x.

Return type:

[tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[*Any*](https://docs.python.org/3/library/typing.html#typing.Any), [*Any*](https://docs.python.org/3/library/typing.html#typing.Any)]

See also

[`hermite_inverse`](https://pianistpy.com/docs/reference/api/numerics#pianistpy.numerics.hermite_inverse)
The points where an increasing interpolant takes given values.

Notes

On the interval \(x_i \le x \le x_{i+1}\), with
\(h = x_{i+1} - x_i\) and \(t = (x - x_i)/h\), the interpolant
is

\[f(x) = (2t^3 - 3t^2 + 1) f_i + (t^3 - 2t^2 + t)\,h f'_i
+ (-2t^3 + 3t^2) f_{i+1} + (t^3 - t^2)\,h f'_{i+1}.\]

Examples

The cubic \(x^3\) is reproduced exactly from its values and slopes
at the two nodes 0 and 2:

```python
>>> import numpy as np
>>> from pianistpy.numerics import hermite
>>> x_nodes = np.array([0.0, 2.0])
>>> f, df_dx = hermite(1.0, x_nodes, x_nodes**3, 3.0 * x_nodes**2)
>>> float(f), float(df_dx)
(1.0, 3.0)
```

pianistpy.numerics.hermite_inverse(*f*, *x_nodes*, *f_nodes*, *df_dx_nodes*)

Return the points where an increasing cubic Hermite interpolant takes values f.

The interpolant is the cubic Hermite interpolant that hermite
evaluates. Each point is found between the two neighboring nodes whose
values enclose the given value, or on the straight line beyond an end
node for a value outside the values at the nodes. For a value between
the values at two neighboring nodes, the returned point lies between
the two nodes, or outside the two nodes by less than \(10^{-12}\)
plus \(2^{-60}\) times the distance between the two nodes, in the
unit of x_nodes.

Parameters:

- **f** ([*float*](https://docs.python.org/3/builtins/functions.html#float)* or **array-like*) – Values of the interpolant for which the points are sought, in the
unit of f_nodes.

- **x_nodes** (*array**, **shape** (**n**,**)*) – Nodes, strictly increasing, with n of 2 or more.

- **f_nodes** (*array**, **shape** (**n**,**)*) – Values of the interpolant at the nodes, increasing.

- **df_dx_nodes** (*array**, **shape** (**n**,**)*) – Slopes of the interpolant at the nodes, in the unit of f_nodes
per unit of x_nodes. The slopes at the first and the last node
must be positive.

Returns:

- **x** (*array*) – Points at which the interpolant equals f, with the shape of f,
in the unit of x_nodes.

- **residual** (*float*) – Largest absolute difference between the interpolant at x and f,
in the unit of f_nodes.

Return type:

[tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[*Any*](https://docs.python.org/3/library/typing.html#typing.Any), [float](https://docs.python.org/3/builtins/functions.html#float)]

See also

[`hermite`](https://pianistpy.com/docs/reference/api/numerics#pianistpy.numerics.hermite)
The interpolant and its derivative.

Examples

The interpolant of the line \(f = 2x\) takes the value 1.5 at
\(x = 0.75\):

```python
>>> import numpy as np
>>> from pianistpy.numerics import hermite_inverse
>>> x_nodes = np.array([0.0, 1.0, 2.0])
>>> x, residual = hermite_inverse(1.5, x_nodes, 2.0 * x_nodes, np.full(3, 2.0))
>>> float(x), residual
(0.75, 0.0)
```
