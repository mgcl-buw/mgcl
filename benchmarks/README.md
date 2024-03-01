# mgcl Benchmarks
## `bench_ignore_tol.cpp` (2024-03-01)
Tests `Problem::solve` having `ignoreTol` set to `true` vs. `false`.

Run with e.g.: `./benchmarks benchIgnoreTol --grids 4,8,16,32,64`

### Results 2024-03-01
After creating `dRsq` buffer in `Level::init`.

On Laptop: ignoreTrue always faster. Bigger difference for smaller grids.
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