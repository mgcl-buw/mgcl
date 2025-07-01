cpmaddpackage("gh:catchorg/Catch2#${MGCL_catch2_version}")
cpmaddpackage("gh:martinus/nanobench#${MGCL_nanobench_version}")

set(SOURCES_BENCH
    ../test/test_utility.hpp
    ../test/test_utility.cpp
    ../test/ocl_wrapper.hpp
    ../test/ocl_wrapper.cpp
    bench_solve.cpp
    bench_init.cpp
    bench_steps.cpp
    bench_stencil.cpp
    bench_galerkin.cpp
    bench_galerkin_optimized.cpp
    bench_stencil_arithmetic.cpp
    bench_stencil_arithmetic_kernels.cpp
    bench_sum.cpp
    bench_jacobi_multiple_iters.cpp
    bench_vcycle_mpi.cpp
    bench_steps_execution_params.cpp
    bench_ignore_tol.cpp
    bench_ghost_update.cpp
    bench_ghost_update_galerkin.cpp
    bench_data_transfer.cpp
    bench_mpi_send_planes.cpp
    bench_fill_buffer.cpp
    bench_utils.hpp # TODO refactor this
    bench_util.hpp
    bench_various_ndim_kernels.cpp
    bench_residual_fixed_stencil.cpp
    bench_residual_varying_stencil.cpp
    bench_cuboid_blockstencil.cpp
    bench_jacobi_temporal_blocking.cpp
    bench_residual_blockstencil.cpp
    bench_borderplanes_blockstencil.cpp
    bench_jacobi_blockstencil.cpp
    bench_residual_varying_stencil_inner_vs_boundary.cpp
    bench_borderplanes.cpp
    bench_jacobi_parallel_ghost_update_mpi.cpp
    cli_args.hpp
    catch2_main_benchs.cpp)

add_executable(benchmarks ${SOURCES_BENCH})

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2 -mfma" MGCL_HAVE_AVX2)
if (MGCL_HAVE_AVX2)
  add_compile_definitions(MGCL_HAVE_AVX2)
  target_compile_options(benchmarks PRIVATE -mavx2 -mfma)
endif()

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

# Remove comments to use address sanitizer (ASAN). ASAN is not compatible with OpenCL, see
# https://stackoverflow.com/a/68027496/4108363. A workaround is to set
# ASAN_OPTIONS=protect_shadow_gap=0 when calling the executable. 
# target_compile_options(benchmarks PUBLIC -fsanitize=address -fno-omit-frame-pointer)
# target_link_options(benchmarks PUBLIC -fsanitize=address)

# Copy kernel files to build dir
include(../cmake/util.cmake)
mgcl_copy_kernel_file(benchmarks kernel_optimizations.cl)
mgcl_copy_kernel_file(benchmarks blockstencil_kernels.cl)
mgcl_copy_kernel_file(benchmarks residual_kernels_inner_vs_boundary.cl)
mgcl_copy_kernel_file(benchmarks borderplanes_kernels.cl)