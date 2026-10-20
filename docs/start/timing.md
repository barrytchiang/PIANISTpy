# Representative computing time

Source: https://pianistpy.com/docs/start/timing

# Representative computing time

The table below times one chain of calls, from a density profile to the observables, built as in Chiang+(2026d): an isotropic NFW parent truncated by Method I, and a child at constant \(\beta_{\rm c} = 0.2\) built
from it by Method II. Each row is the median of three calls after one warm call, except row 6, the table’s
first build. The inputs are in internal units and every setting is at its default. The first two columns are one AMD Ryzen
Threadripper PRO 5975WX (64 threads, 256 GB, Linux), on its CPU alone and then with its NVIDIA GeForce RTX 3080 Ti;
the third is an Intel Core Ultra 9 185H (22 threads, 32 GB, Windows 11), CPU only. These are trial runs on the
development tree; ver. 1.0.0 replaces them with runs on its own code.

Stage

Threadripper PRO 5975WX, CPU (28 workers)

Threadripper PRO 5975WX + RTX 3080 Ti (28 workers)

Core Ultra 9 185H, CPU (16 workers)

1. Parent DF: compute_df (NFW, isotropic)

6.21 s

5.48 s

12.2 s

2. Method I truncation (\(\mathcal{K}_{\rm t} = 0.05\))

4.58 s

4.30 s

9.9 s

3. Child at constant \(\beta_{\rm c} = 0.2\), Method II (compute_df, truncate)

3.48 s

3.36 s

9.1 s

4. Sampler: the speed table

6.59 s

0.914 s

14.5 s

5. Drawing \(10^7\) particles

7.08 s

1.79 s

13.0 s

6. LosTable: the first build

1050 s

1050 s

5335 s

On the model

sigma_r at 50 radii

3.71 s

3.52 s

24.9 s

sigma_t at 50 radii

3.68 s

3.52 s

33.7 s

sigma_los at 50 projected radii

8.75 s

8.20 s

53.4 s

sigma_los_aperture over a circle of radius 1

8.74 s

8.18 s

31.9 s

From row 6’s table

losvd at \(R = 1\), 201 velocities

0.0209 s

0.0190 s

0.043 s

aperture_losvd over a circle of radius 1, 201 velocities

12.2 s

12.5 s

36.9 s

sigma_los_aperture over a circle of radius 1

12.6 s

12.5 s

38.1 s

Gauss-Hermite moments at 10 radii

0.0615 s

0.0619 s

0.095 s

pdf_v at 10 radii, \(V = 0.3\)

5.18 s

5.09 s

27.5 s

Speed distribution at \(r = 1\)

3.87 s

3.77 s

25.3 s

jax.grad of sigma_los at \(R = 1\)

0.00433 s

0.00520 s

0.018 s

Peak memory (main process, largest worker)

3.84 / 0.94 GiB

3.42 / 0.94 GiB

2.59 / 0.49 GiB

Gray cells: the stage’s core runs on the CPU on every platform (the Poisson solve in rows 2 and 3, the table’s rows
in the worker pool in row 6), and the number is the stage’s wall-clock time with the GPU present. On the model, sigma_r and sigma_t each run the model’s
moments solve, and sigma_los and sigma_los_aperture each run that solve and the projection; nothing is kept
on the model between calls, so each row is the cost of one call. The aperture rows from the
table time a later call on the same aperture. The first call on a new aperture also finds the distribution
function’s extrema at the radii the table has not met, once per aperture, and the first aperture call on a table
starts its worker pool: on the Threadripper’s CPU the first call took 31.0 s.

On the Core Ultra 9 185H the table’s build (row 6) takes about five times the Threadripper’s wall-clock time. The rows
on the model run the moments solve serially on one core, and there the wall-clock time of identical calls differs by up to half
from one call to the next, so no ratio between the machines is given for them.
