cmake_minimum_required(VERSION 3.17)
project(MGCL VERSION 0.1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# find_package(CUDAToolkit 8.0 REQUIRED)
find_package(OpenCL REQUIRED)

set(SOURCES_LIB 
cuboid.hpp
cuboid.cpp
ghostscl.cpp
jacobicl.cpp
mgcl.hpp
problem.hpp
problem.cpp
level.hpp
level.cpp
opencl_helper.hpp
opencl_helper.cpp
multigrid_engine.hpp
multigrid_engine.cpp
prolongation.cpp
restriction.cpp
stencil.hpp
stencil.cpp
hypercube.hpp
hypercube.cpp
)

add_library(mgcl ${SOURCES_LIB})
# target_link_libraries(main PRIVATE CUDA::OpenCL)
# target_link_libraries(main OpenCL::OpenCL)
# target_link_options(main PRIVATE -lOpenCL)
target_link_libraries(mgcl PRIVATE "-lOpenCL")

# Uncomment this to get Debug output in mgcl
# target_compile_definitions(mgcl PRIVATE MGCL_DEBUG)

# copy kernel file to build directory
add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/mgcl.cl"
  COMMAND ${CMAKE_COMMAND} -E copy
  "${CMAKE_CURRENT_SOURCE_DIR}/mgcl.cl" "${CMAKE_BINARY_DIR}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mgcl.cl"
  )

# Custom target for activate the custom command above
add_custom_target(copy_kernels DEPENDS "${CMAKE_BINARY_DIR}/mgcl.cl")

# Whenever target 'mgcl' is requested, the custom target will be evaluated.
add_dependencies(mgcl copy_kernels)

Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.1.0 # or a later release
  GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(Catch2)

# These tests can use the Catch2-provided main
set(SOURCES_TEST 
test/test_cuboid.cpp
test/test_problem_conf.cpp
test/test_problem_solve.cpp
test/test_opencl_helper.cpp
test/test_utility.hpp
test/test_utility.cpp
test/test_test_utility.cpp
test/test_ghosts.cpp
test/test_restriction.cpp
test/test_prolongation.cpp
test/test_residual.cpp
test/test_jacobi.cpp
test/test_level.cpp
test/test_stencil.cpp
test/test_hypercube.cpp
)

add_executable(tests ${SOURCES_TEST})
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(tests PRIVATE mgcl)

# pmg configuration for comparison in benchmarks
set (SOURCES_PMG
thirdparty/pmg/cuboid.c
thirdparty/pmg/cuboid.h
thirdparty/pmg/galerkin.c
thirdparty/pmg/galerkin.h
thirdparty/pmg/ghosts.c
thirdparty/pmg/ghosts.h
thirdparty/pmg/interpolate.c
thirdparty/pmg/interpolate.h
thirdparty/pmg/jacobi.c
thirdparty/pmg/jacobi.h
thirdparty/pmg/lueqf.c
thirdparty/pmg/lueqf.h
thirdparty/pmg/mg.c
thirdparty/pmg/mg.h
thirdparty/pmg/rectangle.c
thirdparty/pmg/rectangle.h
thirdparty/pmg/restrict.c
thirdparty/pmg/restrict.h
thirdparty/pmg/stencil.c
thirdparty/pmg/stencil.h
)

add_library(pmg ${SOURCES_PMG})

find_package(MPI)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
add_definitions(-DOMPI_SKIP_MPICXX)
target_link_libraries(pmg PRIVATE MPI::MPI_C)

# Nanobench configuration
FetchContent_Declare(
    nanobench
    GIT_REPOSITORY https://github.com/martinus/nanobench.git
    GIT_TAG v4.3.7
    GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(nanobench)

set(SOURCES_NANOBENCH 
test/test_utility.hpp
test/test_utility.cpp
benchmarks/bench_solve.cpp
benchmarks/bench_init.cpp
benchmarks/bench_steps.cpp
)

add_executable(benchmarks ${SOURCES_NANOBENCH})
# target_include_directories(benchmarks_nanobench PRIVATE include/)
target_link_libraries(benchmarks PRIVATE nanobench)
target_link_libraries(benchmarks PRIVATE Catch2::Catch2WithMain)
target_link_libraries(benchmarks PRIVATE mgcl)
target_link_libraries(benchmarks PRIVATE pmg)

