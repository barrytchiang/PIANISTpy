Accuracy by release
===================

The table below gives, for each release, the counts of the accuracy grid that the
:doc:`Accuracy tables <../reference/accuracy>` page describes. Each row gives the number of grid cells in each tier
(silent, recorded, warned, refused), the number of cells where the package raises an exception instead of
returning a result (the column "Refused with an error"), and the number of cells excluded because the model violates
the Cusp Slope–Central Anisotropy theorem (`An & Evans 2006b <https://doi.org/10.1086/501040>`__), which requires
:math:`\gamma \ge 2\beta_{\rm c}`. Each row also links the full grid as CSV files, one for the default settings
and one for the quick settings, and the list of the grid's models.

.. include:: _accuracy_releases.rst
