# ==========================
Include(FetchContent)

if(${CMAKE_VERSION} VERSION_LESS "3.24.0") 
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        ${MGCL_catch2_version} # or a later release
    GIT_SHALLOW TRUE
  )
else()
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        ${MGCL_catch2_version} # or a later release
    GIT_SHALLOW TRUE
    FIND_PACKAGE_ARGS # try to find using find_package first
  )
endif()

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
)

add_executable(tests ${SOURCES_TEST})
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(tests PRIVATE mgcl)

# don't use iwyu for tests
set_property(
  TARGET tests
  PROPERTY CXX_INCLUDE_WHAT_YOU_USE
)

# Test binary for solving problems
# These tests can use the Catch2-provided main
set(SOURCES_TEST_SOLVING
test_problem_solve.cpp
test_utility.hpp
test_utility.cpp
matrix2d.hpp
matrix2d.cpp
test_results.hpp
test_results.cpp
catch2_main_mpi.cpp
)

add_executable(tests_solving ${SOURCES_TEST_SOLVING})

find_package(MPI REQUIRED)
# target_include_directories(pmg SYSTEM PUBLIC ${MPI_INCLUDE_PATH})
# set(MPI_CXX_SKIP_MPICXX TRUE)
target_link_libraries(tests_solving PRIVATE MPI::MPI_CXX)
target_link_libraries(tests_solving PRIVATE Catch2::Catch2)
target_link_libraries(tests_solving PRIVATE mgcl)
target_link_libraries(tests_solving PRIVATE pmg)

# don't use iwyu for tests_solving
set_property(
  TARGET tests_solving
  PROPERTY CXX_INCLUDE_WHAT_YOU_USE
)
