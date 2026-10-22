# Velocity anisotropy

Source: https://pianistpy.com/docs/guide/anisotropy

`pianistpy.Anisotropy`, `pianistpy.Component`

One class, `Anisotropy(beta_c, r_a, kind)`, covers every analytical anisotropy model of [Chiang+(2026d)](https://arxiv.org/abs/2610.28758):

- the default, `Anisotropy()`, is isotropic;

- a nonzero central anisotropy `beta_c` alone, with \(\beta_{\rm c} < 1\), gives constant anisotropy;

- a finite anisotropy radius `r_a` gives the Osipkov-Merritt model (with `beta_c = 0`) or the
Cuddeford-Osipkov-Merritt (COM) model (with a nonzero `beta_c`). `kind="I"` (the default) is radially biased,
`kind="II"` tangentially biased.

The anisotropy model is chosen by exact comparison of the parameters. `beta_c=0.0` gives the isotropic model, and
`beta_c=1e-12` gives a constant-anisotropy model. The attribute `Anisotropy.model` gives the anisotropy model:
`"isotropic"`, `"constant"`, `"om"` or `"com"`. Results are continuous across \(\beta_{\rm c} = 0\) and
across an infinite \(R_{\rm a}\). A model with \(\beta_{\rm c} = 10^{-12}\) agrees with the isotropic model
within the accuracy of the settings ([Accuracy](https://pianistpy.com/docs/guide/accuracy)).

```python
pp.Anisotropy()                           # isotropic
pp.Anisotropy(beta_c=0.25)                # constant anisotropy
pp.Anisotropy(r_a=10.0, kind="I")         # Osipkov-Merritt, radially biased
pp.Anisotropy(beta_c=-0.25, r_a=10.0)     # COM, Type I by default
```

`Component(profile, anisotropy)` joins a density profile and an anisotropy. A `Component` is the input of
`compute_df()` and `solve()`. In ver. 1.0.0, `compute_df` and `solve` take one
component. Several components in one potential are among the [Upcoming features](https://pianistpy.com/docs/project/upcoming-features).
