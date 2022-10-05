# mgcl

A multigrid implementation using OpenCL for solving PDEs.

## Basic usage
```
#include "Problem.hpp"
#include "Cuboid.hpp"

int N = 64;
mgcl::Cuboid v(N, N, N);
mgcl::Cuboid f(N, N, N);
f.fillRandom();

// default stencil is Laplace 7p
mgcl::Problem p(N, N, N, f, v);
p.setUseOpenCL(true);
p.solve();
// solution is now in v
```

## Reusing an existing OpenCL environment
```
cl_context context;
cl_command_queue commands;
cl_device_id deviceId;

// ... set up your environment ...

mgcl::Problem p(N, N, N, f, v);
p.reuseOpenCL(context, commands, deviceId);
p.solve();
// solution is now in v
```

## Using a varying stencil that differs for each grid point
```
#include "Problem.hpp"
#include "Cuboid.hpp"
#include "Hypercube.hpp"

int N = 64;
mgcl::Cuboid v(N, N, N);
mgcl::Cuboid f(N, N, N);
f.fillRandom();

mgcl::Problem p(N, N, N, f, v);

// Retrieve a 4d Hypercube in which stencil values are stored.
// Size of 4th dimension differs depending on the stencil in use, e.g.
// 7 for 7p varying stencil,
// 19 for 19p varying stencil and
// 27 for 27p varying stencil.
mgcl::Hypercube4d stencilValues = p.getStencilValues();

// ... fill stencil values e.g. in a for-loop using stencilValues[i][j][k][pos] = ...

p.solve();
// solution is now in v
```

## Dependencies
- [cmake](https://cmake.org/) for the build system
- [catch2](https://github.com/catchorg/Catch2) for Unit Tests
- [nanobench](https://github.com/martinus/nanobench) for benchmarks

