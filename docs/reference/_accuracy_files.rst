The grid of ver. 1.0.0 is published as three files, each of which opens with a comment block that repeats the definitions below (``#`` lines, which ``pandas.read_csv(path, comment='#')`` skips):

- :download:`grid_default.csv <../_extra/accuracy/1.0.0/grid_default.csv>`: the grid with the default settings level. One row is one model of the grid.
- :download:`grid_quick.csv <../_extra/accuracy/1.0.0/grid_quick.csv>`: the same grid with the quick settings level, in the same columns.
- :download:`members.csv <../_extra/accuracy/1.0.0/members.csv>`: the list of the grid's models with the place of each in the tables above and the outcome the grid expects of it. One row is one model.

The columns of the grid files, in their order:

``model``
   the model's key in the grid: its table (T1 the main grid, T2 anisotropy radius, T3 truncation energy, T4 limiting cases, the sections of the Accuracy tables page), its profile, its anisotropy model and its truncation
``settings level``
   the settings level of the run: "default" or "quick" (the two tested levels of pianistpy.Settings)
``truncation method``
   1 where the distribution function is inverted from the model's own density (Method I, with or without a truncation); 2 where the model is a Method II model, the distribution function inverted from the truncated density of a Method I model
``anisotropy model``
   the velocity anisotropy model: "iso" (isotropic), "constant" (constant anisotropy :math:`\beta_{\rm c}`), "OM_I" and "OM_II" (Osipkov-Merritt of Type I and Type II), "COM_I" and "COM_II" (Cuddeford-Osipkov-Merritt of Type I and Type II)
``alpha``
   the transition sharpness of the double power law
``beta``
   the outer slope of the double power law
``gamma``
   the inner slope of the double power law
``beta_c``
   the central velocity anisotropy; 0 for the isotropic and the Osipkov-Merritt models
``r_a``
   the anisotropy radius in scale radii; blank for the isotropic and the constant-anisotropy models
``k_t``
   the truncation energy as a fraction of the central potential; blank for an untruncated model
``source model``
   for a Method II model, the Method I model whose truncated density it is built on (its profile, anisotropy model and truncation energy); blank otherwise
``expected outcome``
   what the grid expects of the model: "value" (a result is served), "value: not physical" (a result is served for a model that is not a physical equilibrium), "value: the Method I model's DF within :math:`10^{-6}`, read by the package's rule" (a Method II model whose distribution function equals its Method I model's), or a refusal with the error the package raises (its kind, then the opening words of its message)
``physical``
   whether the model is a physical equilibrium by the grid's own criteria: True, False, or blank where the model is refused before the criteria apply; a result can be served for a model that is not physical
``outcome``
   what the package did: "value" (the model was computed and the error of the density computed back from its distribution function measured); "refused: the package's tier (checks)" (the model was computed and its validation report is in the tier refused, with the checks beyond :math:`10^{-2}` listed); "refusal: <error>: <message>" (the package raised that error; no model exists)
``outcome as expected``
   True where the outcome is the expected outcome, False where it is not (the Accuracy tables page marks such a cell)
``tier``
   the tier of the model's validation report: silent, every check at most :math:`10^{-6}`; recorded, at most :math:`10^{-4}`; warned, at most :math:`10^{-2}`; refused, beyond :math:`10^{-2}`; blank where the model is refused before a report exists
``tier of the distribution function``
   the tier of the distribution function's own report, read as the model's tier (the model's tier is at least this one, since a model's report carries its distribution function's checks)
``lowest energy of the support``
   for a Type II model without a truncation, the energy :math:`P(R_{\rm a})`, as a fraction of the central potential, above which its distribution function is defined; the errors are read over that support only; blank otherwise
``median relative error, 1e-4 < R < 1e4``
   the median over the radii :math:`10^{-4} < R < 10^{4}` (:math:`R` in scale radii) of the relative error of the density computed back from the distribution function against the model's density; blank where no result exists
``maximum relative error, 1e-4 < R < 1e4``
   the maximum of the same relative error over the radii :math:`10^{-4} < R < 10^{4}`
``radius of the maximum, 1e-4 < R < 1e4``
   the radius, in scale radii, at which the maximum over :math:`10^{-4} < R < 10^{4}` is taken
``number of radii, 1e-4 < R < 1e4``
   the number of radii over which the error is read in :math:`10^{-4} < R < 10^{4}`
``radii with a non-finite error, 1e-4 < R < 1e4``
   the number of those radii at which the computed density is not finite; 0 where every radius is finite
``median relative error, 1e-4 < R < 1e5``
   the median over the radii :math:`10^{-4} < R < 10^{5}` (:math:`R` in scale radii) of the relative error of the density computed back from the distribution function against the model's density; blank where no result exists
``maximum relative error, 1e-4 < R < 1e5``
   the maximum of the same relative error over the radii :math:`10^{-4} < R < 10^{5}`
``radius of the maximum, 1e-4 < R < 1e5``
   the radius, in scale radii, at which the maximum over :math:`10^{-4} < R < 10^{5}` is taken
``number of radii, 1e-4 < R < 1e5``
   the number of radii over which the error is read in :math:`10^{-4} < R < 10^{5}`
``radii with a non-finite error, 1e-4 < R < 1e5``
   the number of those radii at which the computed density is not finite; 0 where every radius is finite
``median relative error, 1e-8 < R < 1e8``
   the median over the radii :math:`10^{-8} < R < 10^{8}` (:math:`R` in scale radii) of the relative error of the density computed back from the distribution function against the model's density; blank where no result exists
``maximum relative error, 1e-8 < R < 1e8``
   the maximum of the same relative error over the radii :math:`10^{-8} < R < 10^{8}`
``radius of the maximum, 1e-8 < R < 1e8``
   the radius, in scale radii, at which the maximum over :math:`10^{-8} < R < 10^{8}` is taken
``number of radii, 1e-8 < R < 1e8``
   the number of radii over which the error is read in :math:`10^{-8} < R < 10^{8}`
``radii with a non-finite error, 1e-8 < R < 1e8``
   the number of those radii at which the computed density is not finite; 0 where every radius is finite
``refusal of the distribution function``
   the package's message where it refused to compute the model's distribution function; blank where it did not
``refusal of the truncation``
   the package's message where it refused a truncation of the model: for a truncated model, its own truncation; for an untruncated model, the grid's probe that asks the package to truncate it (a profile whose total mass diverges, outer slope at most 3, and a Type II model whose support ends at the anisotropy's domain edge are refused by design); blank where nothing was refused
``refusal of the density check``
   the package's message where the density could not be computed back from the distribution function (a distribution function that is negative somewhere has no logarithm); blank otherwise

The columns of the model list, in their order:

``model``
   the model's key, as in the grid files
``table``
   the table of the Accuracy tables page the model is in: 1 the main grid, 2 anisotropy radius, 3 truncation energy, 4 limiting cases (the page's sections)
``row``
   the key of the model's row in that table, as the page labels it
``column``
   the key of the model's column in that table, as the page labels it
``alpha``
   the transition sharpness of the double power law
``beta``
   the outer slope of the double power law
``gamma``
   the inner slope of the double power law
``anisotropy model``
   the velocity anisotropy model, as in the grid files
``beta_c``
   the central velocity anisotropy
``r_a``
   the anisotropy radius in scale radii; blank where the model has none
``truncation method``
   1 or 2, as in the grid files
``k_t``
   the truncation energy as a fraction of the central potential; blank for an untruncated model
``source model``
   for a Method II model, the Method I model it is built on; blank otherwise
``expected outcome``
   what the grid expects of the model, as in the grid files
