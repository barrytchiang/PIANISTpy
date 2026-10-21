# Upcoming features

Source: https://pianistpy.com/docs/project/upcoming-features

The features listed below are planned or under development for releases after ver. 1.0.0.

- Generalization of the self-consistent models to systems of several components that share one truncated
potential, and to tracer populations in a fixed potential.

- The test problem of the isolated \(N\)-body runs of [Chiang+(2026d)](https://pianistpy.com/docs/project/citing) as a
[GAMER](https://doi.org/10.1093/mnras/sty2586) test problem, with its analysis and plotting scripts.

- `losvd` and `aperture_losvd` on a `Model` that compute only the part of the line-of-sight table that the
requested radii or apertures need, instead of the whole table.

- Faster derivative evaluations of `losvd` and `aperture_losvd` through parallelization.

- A GPU-accelerated first build of the line-of-sight table.

- The likelihood on data sets of tens of thousands of projected radii or apertures, such as samples of individual
stars or integral-field spaxels.
