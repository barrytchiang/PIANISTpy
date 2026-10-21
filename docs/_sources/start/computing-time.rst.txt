.. role:: nogpu

Representative computing time
=============================

The tables below give the computing time of each step of one PIANISTpy workflow, from a density profile to the
observables. The example truncates an isotropic NFW halo by Method I. Method II then builds a model with a constant velocity
anisotropy of 0.2 on the truncated density of that Method I model (:doc:`Truncation in phase space <../guide/truncation>`). Each time in the tables is the time of a repeated
call. The first call of a session takes longer due to JAX compilation, as does the first aperture call on a
table (14.7 s on spock's CPU). The times were measured with the default settings on two machines. The first machine
is the `spock GPU cluster <https://github.com/calab-ntu/gpu-cluster/wiki/System-Overview#spock>`__ of
`NTU CALab <https://calab-ntu.github.io/>`__ (PI: `Hsi-Yu Schive <https://www.phys.ntu.edu.tw/enphysics/hyschive.html>`__),
with an AMD Ryzen Threadripper PRO 5975WX CPU (32 cores, Linux) and an NVIDIA GeForce RTX 3080 Ti GPU. The second
machine is a laptop with an Intel Core Ultra 9 185H CPU (16 cores, Windows 11). The steps that run in parallel
processes used 28 processes on spock and 12 processes on the laptop, set by :func:`~pianistpy.set_cores`.

.. At each release: the release's runs replace every number (docs/tools/install_chain.py --table), and the
   first aperture call below is read again from the legs; entries under 0.1 s are given in ms.

Building a model
----------------

.. list-table::
   :header-rows: 1
   :widths: 46 18 18 18
   :class: chain-table

   * - Stage
     - spock, CPU
     - spock, CPU + GPU
     - Laptop, CPU
   * - 1\. Untruncated DF: ``compute_df`` (NFW, isotropic)
     - 6.17 s
     - 5.43 s
     - 12.7 s
   * - 2\. Method I truncation :math:`(\mathcal{K}_{\rm t} = 0.05)`
     - 4.34 s
     - :nogpu:`no GPU acceleration`
     - 11.2 s
   * - | 3\. Method II model at constant :math:`\beta_{\rm c} = 0.2`
       | (``compute_df``, ``truncate``)
     - 3.41 s
     - :nogpu:`no GPU acceleration`
     - 10.1 s
   * - 4\. ``Sampler``: the velocity table
     - 6.49 s
     - 0.897 s
     - 7.39 s
   * - 5\. Drawing :math:`10^7` particles
     - 7.08 s
     - 1.91 s
     - 12.9 s
   * - 6\. ``LosTable``: the first build †
     - 13 min
     - :nogpu:`no GPU acceleration`
     - 62 min
   * - | Peak memory over the whole run
       | (main process, largest parallel process)
     - 4.20 / 0.92 GiB
     - 3.49 / 0.93 GiB
     - 2.50 / 0.53 GiB

† A ``LosTable`` is built once per model, and every later call given the same ``LosTable`` reads it without a
rebuild. Every observable in the table "Observables from the line-of-sight table" reads a ``LosTable``. ``losvd`` and ``aperture_losvd`` called on a
model build a ``LosTable`` first. Each row of a ``LosTable`` is the line-of-sight velocity distribution at one
projected radius. The integral of each row agrees with the surface density, and the second moment of each row agrees
with the projected dispersion, each to a relative error of at most :math:`10^{-6}`. For a distribution read between
two rows, the estimated error is at most :math:`10^{-8}` relative in the second moment, and at most
:math:`10^{-7}` of the row's largest value at each velocity. A GPU-accelerated build, and builds of
only the part of a ``LosTable`` that a call needs, are listed on the
:doc:`Upcoming features <../project/upcoming-features>` page.

Observables from the distribution function
------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 46 18 18 18
   :class: chain-table

   * - Stage
     - spock, CPU
     - spock, CPU + GPU
     - Laptop, CPU
   * - ``sigma_r`` evaluated at 50 3D radii
     - 0.648 ms
     - :nogpu:`no GPU acceleration`
     - 0.736 ms
   * - ``sigma_t`` evaluated at 50 3D radii
     - 0.610 ms
     - :nogpu:`no GPU acceleration`
     - 0.659 ms
   * - ``sigma_los`` evaluated at 50 projected radii
     - 4.95 s
     - :nogpu:`no GPU acceleration`
     - 8.09 s
   * - ``sigma_los_aperture`` over a circle of radius 1
     - 4.94 s
     - :nogpu:`no GPU acceleration`
     - 7.68 s
   * - ``pdf_v`` evaluated at 10 3D radii, at the velocity :math:`V = 0.3`
     - 0.836 s
     - :nogpu:`no GPU acceleration`
     - 1.69 s
   * - ``jax.grad`` of ``sigma_los`` at :math:`R_{\rm 2D} = 1`
     - 5.43 ms
     - 6.36 ms
     - 9.12 ms

Observables from the line-of-sight table
----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 46 18 18 18
   :class: chain-table

   * - Stage
     - spock, CPU
     - spock, CPU + GPU
     - Laptop, CPU
   * - | ``losvd`` at :math:`R_{\rm 2D} = 1`, evaluated at
       | 201 line-of-sight velocities
     - 2.69 ms
     - :nogpu:`no GPU acceleration`
     - 3.57 ms
   * - | ``aperture_losvd`` over a circle of radius 1,
       | evaluated at 201 line-of-sight velocities
     - 12.7 s
     - :nogpu:`no GPU acceleration`
     - 32.8 s
   * - ``sigma_los_aperture`` over a circle of radius 1
     - 12.7 s
     - :nogpu:`no GPU acceleration`
     - 29.5 s
   * - Gauss-Hermite moments evaluated at 10 projected radii
     - 55.2 ms
     - :nogpu:`no GPU acceleration`
     - 77.4 ms
