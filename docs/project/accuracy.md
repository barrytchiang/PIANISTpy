# Accuracy by release

Source: https://pianistpy.com/docs/project/accuracy

The table below gives, for each release, the counts of the accuracy grid that the
[Accuracy tables](https://pianistpy.com/docs/reference/accuracy) page describes. Each row gives the number of grid cells in each tier
(silent, recorded, warned, refused), the number of cells where the package raises an exception instead of
returning a result (the column “Refused with an error”), and the number of cells excluded because the model violates
the Cusp Slope–Central Anisotropy theorem ([An & Evans 2006b](https://doi.org/10.1086/501040)), which requires
the inner slope of the density profile, \(\gamma\), to be at least twice the central velocity anisotropy,
\(\beta_{\rm c}\) (i.e., \(\gamma \ge 2\beta_{\rm c}\)). Each row also links the full grid as CSV files, one for the default settings
and one for the quick settings, and the list of the grid’s models.

Release

Silent

Recorded

Warned

Refused (above \(10^{-2}\))

Refused with an error

Excluded (Cusp Slope-Central Anisotropy theorem)

Files

Ver. 1.0.0

380

59

24

12

83

19

[`default`](https://pianistpy.com/docs/_downloads/6f15053f3a7f4c7d100c986afa508c4e/grid_default.csv), [`quick`](https://pianistpy.com/docs/_downloads/2e7555df8c0babbc2a009c85f6173c54/grid_quick.csv), [`models`](https://pianistpy.com/docs/_downloads/962e66e37a969157210bdebc01173211/members.csv)
