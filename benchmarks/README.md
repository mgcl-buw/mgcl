# mgcl Benchmarks
## `bench_ignore_tol.cpp` (2024-03)
Tests `Problem::solve` having `ignoreTol` set to `true` vs. `false` for solving the 4th order periodic problem.

Run with e.g.: `./benchmarks benchIgnoreTol --grids 4,8,16,32,64`

## `bench_steps.cpp` (2022-09, 2024-03)
Runs parts of the v-cycle and prints its timings. Tested steps of the v-cycle are:
- Restriction
- Prolongation
- Jacobi
- Residual
- Ghost update of Cuboid

There are benchmarks measuring time of the function calls using Nanobench.

In 2024-03 a benchmark was added that collects the kernel runtimes using the OpenCL Profiling feature.
