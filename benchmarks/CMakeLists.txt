# ==========================
Include(FetchContent)

# Nanobench configuration
FetchContent_Declare(
    nanobench
    GIT_REPOSITORY https://github.com/martinus/nanobench.git
    GIT_TAG v4.3.11
    GIT_SHALLOW TRUE
    FIND_PACKAGE_ARGS # try to find using find_package first
)

FetchContent_MakeAvailable(nanobench)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.3.2 # or a later release
  GIT_SHALLOW TRUE
  FIND_PACKAGE_ARGS # try to find using find_package first
)

FetchContent_MakeAvailable(Catch2)

# don't use iwyu for Catch2
set_property(
  TARGET Catch2
  PROPERTY CXX_INCLUDE_WHAT_YOU_USE
)
set_property(
  TARGET Catch2WithMain
  PROPERTY CXX_INCLUDE_WHAT_YOU_USE
)

# don't use iwyu for nanobench
set_property(
  TARGET nanobench
  PROPERTY CXX_INCLUDE_WHAT_YOU_USE
)

set(SOURCES_BENCH
../test/test_utility.hpp
../test/test_utility.cpp
bench_solve.cpp
bench_solve_catch2.cpp
bench_init.cpp
bench_steps.cpp
bench_stencil.cpp
bench_galerkin.cpp
bench_stencil_arithmetic.cpp
bench_stencil_arithmetic_kernels.cpp
)

add_executable(benchmarks ${SOURCES_BENCH})

# don't use iwyu for benchmarks
set_property(
  TARGET benchmarks
  PROPERTY CXX_INCLUDE_WHAT_YOU_USE
)

find_package(MPI REQUIRED)
# target_include_directories(benchmarks SYSTEM PRIVATE ${MPI_INCLUDE_PATH})
add_definitions(-DOMPI_SKIP_MPICXX)
target_link_libraries(benchmarks PRIVATE MPI::MPI_C)

# target_include_directories(benchmarks_nanobench PRIVATE include/)
target_link_libraries(benchmarks PRIVATE nanobench)
target_link_libraries(benchmarks PRIVATE Catch2::Catch2WithMain)
target_link_libraries(benchmarks PRIVATE mgcl)
target_link_libraries(benchmarks PRIVATE pmg)
target_link_libraries(benchmarks PRIVATE mgcl_c)

# copy kernel file to build directory
add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/kernel_optimizations.cl"
  COMMAND ${CMAKE_COMMAND} -E copy
  "${CMAKE_CURRENT_SOURCE_DIR}/kernel_optimizations.cl" "${CMAKE_BINARY_DIR}"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernel_optimizations.cl"
)

# Custom target for activate the custom command above
add_custom_target(copy_kernel_optimizations DEPENDS "${CMAKE_BINARY_DIR}/kernel_optimizations.cl")

# Whenever target 'benchmarks' is requested, the custom target will be evaluated.
add_dependencies(benchmarks copy_kernel_optimizations)
