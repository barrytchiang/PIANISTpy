Install
=======

.. code-block:: bash

   pip install pianistpy

PIANISTpy needs Python 3.11 or later and runs on Linux, macOS and Windows.

To check an installation, run the package's tests:

.. code-block:: bash

   pip install "pianistpy[test]"
   pytest --pyargs pianistpy

For a shorter check, ``pytest --pyargs pianistpy -m "not slow"`` skips the slowest tests, which carry the marker
``slow``.

Dependencies
------------

The dependencies are installed by pip with the package. The Standard column of the table gives the version of each
dependency that the release was tested with. The Minimum column gives the oldest version of each dependency, and the
minimum versions together pass the full test suite.

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Package
     - Standard (tested with)
     - Minimum (oldest that passes the suite)
   * - Python
     - 3.11
     - 3.11 (the oldest NumPy 2.3 supports)
   * - NumPy
     - 2.4
     - 2.3.0
   * - SciPy
     - 1.17
     - 1.15.0
   * - h5py
     - 3.15
     - 3.11.0
   * - Astropy
     - 7.2
     - 6.1.0
   * - mpmath
     - 1.3
     - 1.1.0
   * - SymPy
     - 1.14
     - 1.2
   * - JAX
     - 0.9.2
     - 0.6.0
   * - Diffrax
     - 0.7.2
     - 0.7.1
   * - Optimistix
     - 0.1.0
     - 0.1.0

By default, pip installs the newest version of each dependency that PIANISTpy allows. The newest allowed version can
differ from the Standard version, the version of the tested environment. To reproduce the tested environment, install
each dependency at its Standard version, for example ``pip install "jax==0.9.2"``. In ver. 1.0.0, PIANISTpy requires a
JAX version below 0.10.2.

CPU and GPU
-----------

PIANISTpy runs on the CPU with the default installation. With a CUDA build of JAX installed, PIANISTpy runs
every step that has a GPU implementation on the GPU, with no setting in PIANISTpy. The
:doc:`Representative computing time <computing-time>` page shows which steps run on the GPU. JAX's CUDA builds are
for Linux only, so macOS runs the CPU-only mode. A native Windows Python also runs the CPU-only mode. On Windows,
PIANISTpy reaches the GPU through the Windows Subsystem for Linux or a Linux virtual machine. Results on the GPU and
on the CPU agree to better than :math:`10^{-10}` in relative terms, and every accuracy check holds on the GPU and on
the CPU. To
choose the CPU or the GPU explicitly, see :doc:`Runtime options <../guide/runtime>`.