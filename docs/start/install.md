# Install

Source: https://pianistpy.com/docs/start/install

# Install

```
pip install pianistpy

```

PIANISTpy needs Python 3.11 or later and runs on Linux, macOS and Windows.

The test suite is installed with the package. To run it on an installation:

```
pip install "pianistpy[test]"
pytest --pyargs pianistpy

```

## Dependencies

The dependencies are installed by pip with the package. Two lists are kept with every release: the versions the
release was developed and tested with, and the oldest versions of the listed dependencies that pass the full test
suite together.

Package

Standard (tested with)

Minimum (oldest that passes the suite)

Python

3.11

3.11 (the oldest NumPy 2.3 supports)

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

1.3

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

This standard list is the systematically tested environment, which can differ from the default package versions
that pip installs by taking the newest version of each dependency that PIANISTpy allows. To reproduce the tested
environment, manual specifications of each dependent package version during installation may be required. JAX
0.10.2 and later are not allowed by PIANISTpy, because one test of the suite fails on 0.10.2.

## CPU and GPU

PIANISTpy runs on the CPU with the default installation. With a CUDA build of JAX installed, it automatically
becomes GPU-enabled for the GPU-implemented features, with no manual setting needed in PIANISTpy. JAX’s CUDA builds
are for Linux only, so macOS runs the CPU-only mode. A native Windows Python also runs the CPU-only mode; on
Windows the GPU is reached through the Windows Subsystem for Linux (or a Linux virtual machine). Results on the
GPU and the CPU agree to better than \(10^{-10}\) in relative terms, and the accuracy goal holds on both. To
choose the CPU or the GPU explicitly, see Runtime options.

In ver. 1.0.0, only the particle sampler (its speed table and the draw) uses the GPU.

     On this page
