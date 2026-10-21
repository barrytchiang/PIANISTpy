Models
======

A model (:class:`~pianistpy.Model`) is an equilibrium of one or more components, untruncated or truncated in phase
space. Each component is a density profile with a velocity anisotropy, and the model holds the distribution
function of each component. A model is built in the four main steps below.

.. toctree::
   :maxdepth: 1

   Density profiles (DoublePowerLaw) <profiles>
   Velocity anisotropy (Anisotropy, Component) <anisotropy>
   Distribution functions (compute_df) <df>
   Truncation in phase space (truncate, solve) <truncation>
