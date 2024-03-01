# mgcl Benchmarks
## `bench_ignore_tol.cpp` (2024-03-01)
Tests `Problem::solve` having `ignoreTol` set to `true` vs. `false` for solving the 4th order periodic problem.

Run with e.g.: `./benchmarks benchIgnoreTol --grids 4,8,16,32,64`

### Results 2024-03-01
After creating `dRsq` buffer in `Level::init`.

Summary: ignoreTrue always faster. Bigger difference for smaller grids.

Problem parameters:
- tol: 1e-07
- nu1: 2
- nu2: 2
- omega: 0.8
- v-cycle iterations: 10

Laptop
| relative |  ms/op |   op/s | err% | total | benchmark              |
| -------: | -----: | -----: | ---: | ----: | :--------------------- |
|   100.0% |   5.17 | 193.25 | 0.5% |  1.19 | `ignoreFalse_4x4x4`    |
|   140.9% |   3.67 | 272.20 | 0.6% |  1.20 | `ignoreTrue__4x4x4`    |
|   100.0% |   7.69 | 130.03 | 1.1% |  1.18 | `ignoreFalse_8x8x8`    |
|   139.8% |   5.50 | 181.73 | 0.6% |  1.22 | `ignoreTrue__8x8x8`    |
|   100.0% |  15.83 |  63.17 | 0.4% |  1.23 | `ignoreFalse_16x16x16` |
|   123.2% |  12.85 |  77.85 | 0.4% |  1.19 | `ignoreTrue__16x16x16` |
|   100.0% | 106.52 |   9.39 | 0.2% |  1.18 | `ignoreFalse_32x32x32` |
|   103.7% | 102.77 |   9.73 | 0.3% |  1.13 | `ignoreTrue__32x32x32` |
|   100.0% | 963.53 |   1.04 | 0.1% | 10.60 | `ignoreFalse_64x64x64` |
|   101.1% | 953.07 |   1.05 | 0.1% | 10.48 | `ignoreTrue__64x64x64` |

Pleiades
| relative |  ms/op |   op/s | err% | total | benchmark                 |
| -------: | -----: | -----: | ---: | ----: | :------------------------ |
|   100.0% |   5.67 | 176.33 | 0.2% |  1.18 | `ignoreFalse_4x4x4`       |
|   138.4% |   4.10 | 244.11 | 0.6% |  1.22 | `ignoreTrue__4x4x4`       |
|   100.0% |   8.22 | 121.65 | 0.1% |  1.18 | `ignoreFalse_8x8x8`       |
|   136.4% |   6.03 | 165.97 | 0.3% |  1.21 | `ignoreTrue__8x8x8`       |
|   100.0% |  11.90 |  84.01 | 0.3% |  1.24 | `ignoreFalse_16x16x16`    |
|   132.5% |   8.98 | 111.32 | 0.1% |  1.20 | `ignoreTrue__16x16x16`    |
|   100.0% |  20.76 |  48.16 | 0.1% |  1.25 | `ignoreFalse_32x32x32`    |
|   121.6% |  17.07 |  58.57 | 0.2% |  1.21 | `ignoreTrue__32x32x32`    |
|   100.0% |  66.14 |  15.12 | 0.7% |  1.27 | `ignoreFalse_64x64x64`    |
|   108.5% |  60.94 |  16.41 | 0.8% |  1.35 | `ignoreTrue__64x64x64`    |
|   100.0% | 719.24 |   1.39 | 0.1% |  7.97 | `ignoreFalse_128x128x128` |
|   100.1% | 718.77 |   1.39 | 0.2% |  7.90 | `ignoreTrue__128x128x128` |