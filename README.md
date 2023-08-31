# mgcl

A multigrid implementation using OpenCL for solving elliptic PDEs with varying coefficients.

**Important: This library is work in progress and neither feature-complete nor bug-free (probably)!**

Currently only unit cube shaped domains are supported.

## Basic usage
```
#include "problem.hpp"
#include "cuboid.hpp"

int N = 64;
mgcl::Cuboid v(N, N, N);
mgcl::Cuboid f(N, N, N);
f.fillRandom();

// default stencil is Laplace 7p with periodic bcs
mgcl::Problem p(N, N, N, f, v);
p.setUseOpenCL(true);
p.solve();
// solution is now in v
```

## Using Dirichlet bcs
```
#include "problem.hpp"
#include "cuboid.hpp"

int N = 64;
int gh_in = 1;
int Ngh = N + 2 * gh_in;
mgcl::Cuboid v(Ngh, Ngh, Ngh);
mgcl::Cuboid f(Ngh, Ngh, Ngh);
f.fillRandom();

// Define boundary values in halo region of v, e.g. 1.0 for left side.
// If gh_in > 1, only those halo cells sitting directly at the boundary are read.
for (int j = 0; j < Ngh; j++)
    for (int k = 0; k < Ngh; k++)
        v[0][j][k] = 1.0;

// default stencil is Laplace 7p with periodic bcs
mgcl::Problem p(N, N, N, f, v);
p.setUseOpenCL(true);
p.setGhostsIn(gh_in);
p.setBc(mgcl::BC::DIRICHLET);
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
#include "problem.hpp"
#include "cuboid.hpp"
#include "stencil.hpp"

int N = 64;
mgcl::Cuboid v(N, N, N);
mgcl::Cuboid f(N, N, N);
f.fillRandom();

mgcl::Problem p(N, N, N, f, v);

// Retrieve a 6d VaryingStencil in which stencil values are stored.
// First three dimensions match grid size + ghost size, last three
// dimensions are 3.
p.setStencilType(mgcl::MGCL_VARYING);
auto &s = *p.getStencilValues();

double h2inv = N * N;

// i.e. fill with 7-point Laplace
for (int i = 0; i < s.getDim1gh(); i++)
    for (int j = 0; j < s.getDim2gh(); j++)
        for (int k = 0; k < s.getDim3gh(); k++)
        {
            // 7-point Laplace
            s[i][j][k][0][1][1] = h2inv * -1.0;
            s[i][j][k][1][0][1] = h2inv * -1.0;
            s[i][j][k][1][1][0] = h2inv * -1.0;
            s[i][j][k][1][1][1] = h2inv * 6.0;
            s[i][j][k][1][1][2] = h2inv * -1.0;
            s[i][j][k][1][2][1] = h2inv * -1.0;
            s[i][j][k][2][1][1] = h2inv * -1.0;
        }

p.solve();
// solution is now in v
```

## Using MPI
To use MPI configure CMake with `-DMGCL_USE_MPI`. A custom communicator with attached topology information must be set using `Problem::setMpiComm`. See exmaples/example_mpi.cpp for more information.

There is a threshold that prevents coarse levels to be calculated using MPI. It can be set by using `Problem::setMpiLevelThreshold` but it can only go as high as there are still at least 8 grid points (atm) available on each process for each direction.

## Dependencies
- [cmake](https://cmake.org/) for the build system
- [catch2](https://github.com/catchorg/Catch2) for Unit Tests
- [nanobench](https://github.com/martinus/nanobench) for benchmarks
- [Backward-cpp](https://github.com/bombela/backward-cpp) for prettier Stacktraces (only used for Tests)

## Build

Optional useful CMake options:
- `-DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/openmpi` (if mpi is not found by default)
- `-DBACKWARD_HAS_DWARF=ON` (for tests targets)
- `-DCMAKE_CXX_COMPILER_LAUNCHER=ccache` (needs to have ccache installed on your system)

Not used anymore:
- `-DCMAKE_CXX_INCLUDE_WHAT_YOU_USE=include-what-you-use;-Xiwyu;--cxx17ns` (currently hard-coded only for src and for Debug type)
- `-DCMAKE_TOOLCHAIN_FILE=<vcpkgInstallDir>/scripts/buildsystems/vcpkg.cmake` (currently vcpkg is not supported)

For a simple build:
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release <options>
cmake --build .
```

For a development build enabling switching between build types without recompiling all dependencies, creating separate build directories for Debug and Release is suggested. In VSCode you can e.g. add in the settings:

```
"cmake.buildDirectory": "${workspaceFolder}/build/${buildType}"
```
