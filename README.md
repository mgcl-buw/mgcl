# mgcl

A multigrid implementation using OpenCL for sovling PDEs.

## Usage
```
#include "Problem.hpp"
#include "Cuboid.hpp"

int N = 64;
mgcl::Cuboid v(N, N, N);
mgcl::Cuboid f(N, N, N);
f.fillRandom();

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
p2.reuseOpenCL(context, commands, deviceId);
p.solve();
// solution is now in v
```

## Dependencies
- [cmake](https://cmake.org/) for the build system
- [catch2](https://github.com/catchorg/Catch2) for Unit Tests
- [nanobench](https://github.com/martinus/nanobench) for benchmarks

