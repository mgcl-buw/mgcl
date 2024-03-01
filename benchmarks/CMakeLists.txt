cpmaddpackage("gh:catchorg/Catch2#${MGCL_catch2_version}")
cpmaddpackage("gh:martinus/nanobench#${MGCL_nanobench_version}")

set(SOURCES_BENCH
    ../test/test_utility.hpp
    ../test/test_utility.cpp
    ../test/ocl_wrapper.hpp
    ../test/ocl_wrapper.cpp
    bench_solve.cpp
    bench_solve_catch2.cpp
    bench_init.cpp
    bench_steps.cpp
    bench_stencil.cpp
    bench_galerkin.cpp
    bench_stencil_arithmetic.cpp
    bench_stencil_arithmetic_kernels.cpp
    bench_sum.cpp
    bench_jacobi_multiple_iters.cpp
    bench_vcycle_mpi.cpp
    bench_steps_execution_params.cpp
    bench_ignore_tol.cpp
    cli_args.hpp
    catch2_main_benchs.cpp)

add_executable(benchmarks ${SOURCES_BENCH})

find_package(MPI REQUIRED)
# target_include_directories(benchmarks SYSTEM PRIVATE ${MPI_INCLUDE_PATH})
# add_definitions(-DOMPI_SKIP_MPICXX)
target_link_libraries(benchmarks PRIVATE MPI::MPI_CXX)

# target_include_directories(benchmarks_nanobench PRIVATE include/)
target_link_libraries(benchmarks PRIVATE nanobench::nanobench)
target_link_libraries(benchmarks PRIVATE Catch2::Catch2)
target_link_libraries(benchmarks PRIVATE mgcl)
target_link_libraries(benchmarks PRIVATE pmg)
target_link_libraries(benchmarks PRIVATE mgcl_c)

# copy kernel file to build directory
add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/benchmarks/kernel_optimizations.cl"
  COMMAND
    ${CMAKE_COMMAND} -E copy
    "${CMAKE_CURRENT_SOURCE_DIR}/kernel_optimizations.cl"
    "${CMAKE_BINARY_DIR}/benchmarks"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kernel_optimizations.cl")

# Custom target for activate the custom command above
add_custom_target(
  copy_kernel_optimizations
  DEPENDS "${CMAKE_BINARY_DIR}/benchmarks/kernel_optimizations.cl")

# Whenever target 'benchmarks' is requested, the custom target will be
# evaluated.
add_dependencies(benchmarks copy_kernel_optimizations)
