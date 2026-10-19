:html_theme.sidebar_secondary.remove: true

Accuracy tables
===============

The accuracy grid is a fixed set of models, four tables of rows by columns, run unchanged at every release. Each
model's distribution function is inverted and its density :math:`\varrho_{\rm inv}` computed back from it; the
roundtrip error :math:`|(\varrho_{\rm inv} - \varrho)/\varrho|`, against the input profile :math:`\varrho`, is read at
513 radii, and a cell gives that error over :math:`10^{-4} < R < 10^{5}` (:math:`R` in scale radii):

- its **maximum**, then its **median**, then the **tier** the result itself reports (goal: :math:`10^{-6}` or less;
  ceiling: :math:`10^{-4}` or less; warning: :math:`10^{-2}` or less; refused: beyond, see :doc:`../guide/accuracy`);
- **refused** and the refusal's name, where the package refuses the model (the full message is in the cell's
  tooltip and in the CSV);
- **excluded** where the anisotropy breaks the cusp bound :math:`\gamma \ge 2\beta_{\rm c}` (computed, and recorded in the
  CSV). A Type II model without a truncation has its distribution function only above the energy :math:`P(R_{\rm a})`: its
  cell gives the tier the package reports over that support and the note "support above :math:`P(R_{\rm a})`", not the
  density's roundtrip, which is no measure there.

A † marks a cell whose outcome is not the one the grid expects. The tables are at the default tier; the CSV holds
every range the grid reads, the radius of each maximum and the compute times. The tables of earlier releases are
on :doc:`../project/accuracy`.

.. include:: _accuracy_tables.rst
