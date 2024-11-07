find_package(MPI REQUIRED)

# ==========================
# example program that initializes a Problem, i.e. test the galerkin operator,
# that can be profiled using e.g. gprof
set(SOURCES_EXAMPLE_GALERKIN ../test/test_utility.hpp ../test/test_utility.cpp
                             example_galerkin.cpp)

add_executable(example_galerkin ${SOURCES_EXAMPLE_GALERKIN})
target_link_libraries(example_galerkin PRIVATE mgcl)
# target_compile_options(example_galerkin PRIVATE "-pg")
# target_link_options(example_galerkin PRIVATE "-pg")
target_link_libraries(example_galerkin PRIVATE MPI::MPI_CXX)

# ==========================
# example program that uses MPI
set(SOURCES_EXAMPLE_MPI example_mpi.cpp)

add_executable(example_mpi ${SOURCES_EXAMPLE_MPI})
target_link_libraries(example_mpi PRIVATE mgcl MPI::MPI_CXX)
# target_compile_options(example_galerkin PRIVATE "-pg")
# target_link_options(example_galerkin PRIVATE "-pg")

# ==========================
# example program that just prints all available devices
add_executable(example_print_devices example_print_devices.cpp)
target_link_libraries(example_print_devices PRIVATE mgcl MPI::MPI_CXX)

# ==========================
# mgclTune, tries to find the best work-group sizes for each kernel
set(SOURCES_MGCL_TUNE mgclTune.cpp)

add_executable(mgcl_tune ${SOURCES_MGCL_TUNE})
target_link_libraries(mgcl_tune PRIVATE mgcl)
# target_compile_options(mgcl_tune PRIVATE "-pg")
# target_link_options(mgcl_tune PRIVATE "-pg")
target_link_libraries(mgcl_tune PRIVATE MPI::MPI_CXX)

# ==========================
# example_profile, demonstrates mgcl's kernel profiling feature
set(SOURCES_EXAMPLE_PROFILE example_profile.cpp)

add_executable(example_profile ${SOURCES_EXAMPLE_PROFILE})
target_link_libraries(example_profile PRIVATE mgcl)
# target_compile_options(mgcl_tune PRIVATE "-pg")
# target_link_options(mgcl_tune PRIVATE "-pg")

# ==========================
# example_use_binary, demonstrates the usage of a prebuilt kernel binary
set(SOURCES_EXAMPLE_USE_BINARY example_use_binary.cpp)

add_executable(example_use_binary ${SOURCES_EXAMPLE_USE_BINARY})
target_link_libraries(example_use_binary PRIVATE mgcl)
# target_compile_options(mgcl_tune PRIVATE "-pg")
# target_link_options(mgcl_tune PRIVATE "-pg")


