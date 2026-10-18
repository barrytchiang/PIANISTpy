# Install

Source: https://pianistpy.com/docs/start/install.html

# Install

```
pip install pianistpy

```

PIANISTpy needs Python 3.10 or later and runs on Linux, macOS and Windows.

## Requirements

The dependencies are installed by pip with the package. Two lists are kept with every release: the versions the
release was developed and tested with, and the oldest versions that pass the full test suite.

Package

Standard (tested with)

Minimum (oldest that passes the suite)

Python

3.11

3.10 (the oldest with a JAX 0.6 wheel; the suite is run on 3.11)

NumPy

2.4

2.3.0

SciPy

1.17

1.15.0

h5py

3.15

3.11.0

Astropy

7.2

6.1.0

mpmath

1.4

1.1.0

SymPy

1.14

1.2

JAX

0.9.2

0.6.0

Diffrax

0.7.2

0.7.1

Optimistix

0.1.0

0.1.0

The minimum list is measured one package at a time with the others at their newest, and then all the minimum
versions together as one environment, which passes the suite; the standard list is the environment of the
release. Astropy below 7.2 needs NumPy below 2.4 (older Astropy calls a function NumPy 2.4 removed).

## JAX and the GPU

The package runs on the CPU everywhere and takes a GPU where the installed JAX supports one; nothing is set by
the user. JAX’s CUDA builds are for Linux: on Windows the GPU is reached through the Windows Subsystem for Linux
(or a Linux virtual machine) with the Linux wheels, and a native Windows Python runs the CPU-only mode. A GPU and
a CPU give results that differ at rounding on the kernels; the accuracy goal holds on both, and each file
records the backend that made it. To force one or the other, see Runtime options.

Only the main process uses the card: the worker processes of the observables’ pools run on the CPU. A card in the
Exclusive_Process compute mode (nvidia-smi --query-gpu=compute_mode --format=csv shows it), common on shared
clusters, serves one process, so where another process already holds it the package cannot reach it, and JAX’s
error begins Unable to initialize backend 'cuda'. Free the card, share it through an MPS daemon
(nvidia-cuda-mps-control -d), or run in the CPU-only mode.

A card speeds up the sampler and nothing else. The sampler’s speed table builds in 0.6 to 6.6 s on the card against
4.6 to 40 s on 8 cores, and drawing is faster on the card from about ten thousand particles, three to five times
from a hundred thousand. The DF’s Poisson solve is a serial ODE: where the package runs on a card, so does the
solve, and each step costs about a millisecond of launch latency, 50 to 200 times the CPU’s wall, so a model is
solved fastest in the CPU-only mode. The rows of the line-of-sight velocity table run in the worker pool on the CPU
whatever the platform, so a card does not shorten the table’s build (the two columns below differ by the main
process’s share and the node’s load), and reading a built table (the LOSVD, pdf_v) takes the CPU’s wall. At the
quick tier the table’s build ran out of the card’s 12 GB, so the quick tier’s observables were measured on the CPU
only. The walls below are from one 32-core node (AMD Threadripper PRO 5975WX) and one RTX 3080 Ti. The solve’s are
the medians of five reruns of 37 solves of 21 truncated models, in two groups by their step count. The speed
table’s are the steady builds of six members (512 to 1,021 rows) at the default tier, on 8 of the node’s cores. The
sampling’s and the observables’ are on an NFW model (beta_c 0.25, concentration 10) and a truncated Hernquist model
(Osipkov-Merritt Type I, beta_c -0.25, R_a 10, K_t 0.05); each is the median of three calls after a warm one, except
the table’s build, which is the first and the second build in one process.

Stage

CPU (32 cores)

RTX 3080 Ti

DF solve, one solve (478 to 1574 steps)

0.0036 to 0.025 s

0.66 to 1.55 s

The sampler’s speed table, default tier (8 cores)

4.6 to 40 s

0.63 to 6.6 s

Sampling, 1e3 particles

0.014 to 0.019 s

0.019 to 0.025 s

Sampling, 1e4 particles

0.024 to 0.031 s

0.019 to 0.026 s

Sampling, 1e5 particles

0.103 to 0.110 s

0.029 to 0.034 s

Sampling, 1e6 particles

0.736 to 0.768 s

0.162 to 0.211 s

Line-of-sight velocity table, default tier

1246 to 2298 s

1014 to 3163 s

Line-of-sight velocity table, quick tier

238 to 1251 s

out of the card’s memory

LOSVD, 64 velocities at 16 radii, default tier

0.46 to 0.48 s

0.42 to 0.43 s

LOSVD, 64 velocities at 1024 radii, default tier

30.4 to 31.1 s

28.4 to 28.9 s

pdf_v, 64 velocities at 16 radii, default tier

9.2 to 24.4 s

9.2 to 24.3 s

pdf_v, 64 velocities at 1024 radii, default tier

595 to 1565 s

591 to 1568 s

LOSVD, 64 velocities at 16 to 1024 radii, quick tier

0.13 to 9.4 s

not measured

pdf_v, 64 velocities at 16 to 1024 radii, quick tier

4.2 to 709 s

not measured

     On this page
