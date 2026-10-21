:html_theme.sidebar_secondary.remove: true

Accuracy tables
===============

The accuracy tables below record the relative error of extensive "roundtrip" tests over a grid of systematically
varying input density profiles and anisotropy models. In each test, the package first
numerically computes the distribution function inverted from the input density profile :math:`\varrho`; the density
profile :math:`\varrho_{\rm inv}` numerically evaluated from that distribution function is then compared against
the analytical input (the truth). Each cell reports the relative error
:math:`|(\varrho_{\rm inv} - \varrho)/\varrho|` over :math:`10^{-8} < R < 10^{8}`, with :math:`R` in units of the
profile's scale radius. The column heads give the anisotropy models: OM is
Osipkov-Merritt and COM is Cuddeford-Osipkov-Merritt, of Type I or Type II (:doc:`../guide/anisotropy`). A cell
shows one of three outcomes:

- the error's **maximum** (bold) and **median** (gray), and the result's **tier** (:doc:`Accuracy <../guide/accuracy>`); a
  cell without a tier is silent. The tier comes from every check of the result, so a tier can be higher than the
  cell's error alone gives.
- **error** and the exception's class, where the package raises an exception instead of returning a result:
  :class:`~pianistpy.UnphysicalModelError` where no physical equilibrium exists for the model's parameters,
  :class:`~pianistpy.InputError` where the parameters are outside what the package supports, and
  :class:`~pianistpy.AccuracyError` where the package cannot compute the model to the required accuracy. The
  exception's full message is in the cell's tooltip and in the CSV.
- **excluded**, where the model violates the Cusp Slope–Central Anisotropy theorem
  (`An & Evans 2006b <https://doi.org/10.1086/501040>`__), which requires :math:`\gamma \ge 2\beta_{\rm c}`. The model is still
  computed and recorded in the CSV.

The distribution function of an untruncated Type II model is defined only at energies above
:math:`P(R_{\rm a})`, the relative potential at the anisotropy radius :math:`R_{\rm a}`. The cell of such a model
gives the tier that the package reports at energies above :math:`P(R_{\rm a})`, with the note "support above
:math:`P(R_{\rm a})`". Below :math:`P(R_{\rm a})`, the density :math:`\varrho_{\rm inv}` does not measure the
distribution function's accuracy, so the cell gives no error values.

A † marks a cell whose outcome differs from the outcome that the CSV's ``expected outcome`` column states for the
model. The tables use the default settings level. The
downloadable files below hold every radial range the grid reads and the radius of each maximum, for the default
and the quick settings level. The counts of each release are on :doc:`../project/accuracy`.

.. include:: _accuracy_tables.rst

The downloadable files
----------------------

.. include:: _accuracy_files.rst
