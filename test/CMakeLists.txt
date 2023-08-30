find_package(MPI REQUIRED)

# Don't use static analyzers for tests.
if(USE_STATIC_ANALYZER)
  include(../cmake/tools.cmake)
  reset_clang_tidy()
  reset_cppcheck()
  reset_include_what_you_use()
endif()

CPMAddPackage("gh:catchorg/Catch2#${MGCL_catch2_version}")

# Test binary for unit testing mgcl
# These tests can use the Catch2-provided main
set(SOURCES_TEST 
test_cuboid.cpp
test_problem_conf.cpp
test_opencl_helper.cpp
test_utility.hpp
test_utility.cpp
test_test_utility.cpp
test_ghosts.cpp
test_restriction.cpp
test_prolongation.cpp
test_residual.cpp
test_jacobi.cpp
test_level.cpp
test_stencil.cpp
test_stencil_gpu.cpp
test_hypercube.cpp
test_galerkin.cpp
matrix2d.hpp
matrix2d.cpp
test_matrix2d.cpp
test_results.hpp
test_results.cpp
test_util.cpp
ocl_wrapper.hpp
ocl_wrapper.cpp
mpi/catch2_main_mpi.cpp
)

add_executable(tests ${SOURCES_TEST})

target_link_libraries(tests PRIVATE Catch2::Catch2 MPI::MPI_CXX)
target_link_libraries(tests PRIVATE mgcl)

# ==========================

# Test binary for solving problems
# These tests must use custom Catch2-Main for MPI
set(SOURCES_TEST_SOLVING
test_problem_solve.cpp
test_utility.hpp
test_utility.cpp
matrix2d.hpp
matrix2d.cpp
test_results.hpp
test_results.cpp
mpi/catch2_main_mpi.cpp
)

add_executable(tests_solving ${SOURCES_TEST_SOLVING})

# find_package(MPI REQUIRED)
# target_include_directories(pmg SYSTEM PUBLIC ${MPI_INCLUDE_PATH})
# set(MPI_CXX_SKIP_MPICXX TRUE)
target_link_libraries(tests_solving PRIVATE MPI::MPI_CXX)
target_link_libraries(tests_solving PRIVATE Catch2::Catch2)
target_link_libraries(tests_solving PRIVATE mgcl)
target_link_libraries(tests_solving PRIVATE pmg)

# ==========================

# Test binary for mpi
# These tests must use custom Catch2-Main for MPI
set(SOURCES_TEST_MPI
test_utility.hpp
test_utility.cpp
mpi/test_mpi_level.cpp
mpi/test_mpi_problem.cpp
mpi/test_mpi_ghosts.cpp
mpi/test_mpi_jacobi.cpp
mpi/test_mpi_util.cpp
mpi/test_mpi_vcycle.cpp
mpi/catch2_main_mpi.cpp
)

add_executable(tests_mpi ${SOURCES_TEST_MPI})

# find_package(MPI REQUIRED)
# target_include_directories(pmg SYSTEM PUBLIC ${MPI_INCLUDE_PATH})
# set(MPI_CXX_SKIP_MPICXX TRUE)
target_link_libraries(tests_mpi PRIVATE MPI::MPI_CXX Catch2::Catch2 mgcl)

# copy run-all script file to build directory
add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/test/run-all.sh"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/mpi/run-all.sh" "${CMAKE_BINARY_DIR}/test"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mpi/run-all.sh"
  )

# Custom target for activate the custom command above
add_custom_target(copy_script DEPENDS "${CMAKE_BINARY_DIR}/test/run-all.sh")

# Whenever target 'tests_mpi' is requested, the custom target will be evaluated.
add_dependencies(tests_mpi copy_script)
