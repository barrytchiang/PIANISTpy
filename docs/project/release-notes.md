# Release notes

Source: https://pianistpy.com/docs/project/release-notes

## Ver. 1.0.0

The initial public release of PIANISTpy, described in [Chiang+(2026d)](https://pianistpy.com/docs/project/citing).

### Known limitations

- Spherical, non-rotating systems with a double-power-law input density profile.

- One density profile per potential ([Upcoming features](https://pianistpy.com/docs/project/upcoming-features)).

- Gradients in the anisotropy parameters are available only for Method II models, whose input density profile
is the truncated density profile of another model. A Method I truncated model raises
`InputError`. Separately, for Cuddeford-Osipkov-Merritt (COM) Type II models, the
distribution function and every observable have no gradient in the anisotropy parameters
([End-to-end differentiability](https://pianistpy.com/docs/guide/differentiability)).

- With `Settings(check_rows=True)`, the gradient of `losvd` and `aperture_losvd` recomputes each row of
the line-of-sight table that the gradient reads and requires the recomputed row to equal the stored row bit for
bit, and a JAX copy of the row’s computation to reproduce the row to \(10^{-12}\) of the row’s largest
value. On a GPU, whose order of summation does not promise the same bits, the comparison can fail a row and the
gradient raises `ResultError` instead of returning an inexact value ([Differentiable observables](https://pianistpy.com/docs/guide/differentiable)). The
default, `check_rows=False`, recomputes and compares no row.

- The derivative of `losvd` and `aperture_losvd` through the line-of-sight table takes minutes to hours
([Differentiable observables](https://pianistpy.com/docs/guide/differentiable)).

Dependencies: the standard and the minimum versions are listed on [Install](https://pianistpy.com/docs/start/install).
