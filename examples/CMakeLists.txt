if (MGCL_USE_MPI)
  find_package(MPI REQUIRED)
  add_compile_definitions("MGCL_USE_MPI")
endif()

# ==========================
# example main that can be profiled using e.g. gprof
set(SOURCES_EXAMPLE_MAIN
../test/test_utility.hpp
../test/test_utility.cpp
example_main.cpp
)

add_executable(example_main ${SOURCES_EXAMPLE_MAIN})
target_link_libraries(example_main PRIVATE mgcl_c)
target_link_libraries(example_main PRIVATE mgcl)
add_dependencies(example_main copy_kernels)
add_dependencies(example_main copy_kernels_mgcl_c)
#target_compile_options(example_main PRIVATE "-pg")
#target_link_options(example_main PRIVATE "-pg")

if (MGCL_USE_MPI)
  target_link_libraries(example_main PRIVATE MPI::MPI_CXX) 
endif()

# don't use iwyu for example_main
set_property(
  TARGET example_main
  PROPERTY CXX_INCLUDE_WHAT_YOU_USE
)

# ==========================
# example program that initializes a Problem, i.e. test the galerkin operator, that can be profiled using e.g. gprof
set(SOURCES_EXAMPLE_GALERKIN
../test/test_utility.hpp
../test/test_utility.cpp
example_galerkin.cpp
)

add_executable(example_galerkin ${SOURCES_EXAMPLE_GALERKIN})
target_link_libraries(example_galerkin PRIVATE mgcl)
add_dependencies(example_galerkin copy_kernels)
#target_compile_options(example_galerkin PRIVATE "-pg")
#target_link_options(example_galerkin PRIVATE "-pg")

if (MGCL_USE_MPI)
  target_link_libraries(example_galerkin PRIVATE MPI::MPI_CXX) 
endif()

# don't use iwyu for example_galerkin
set_property(
  TARGET example_galerkin
  PROPERTY CXX_INCLUDE_WHAT_YOU_USE
)

# ==========================
if (MGCL_USE_MPI)

  # example program that uses MPI
  set(SOURCES_EXAMPLE_MPI
    example_mpi.cpp
  )

  add_executable(example_mpi ${SOURCES_EXAMPLE_MPI})
  target_link_libraries(example_mpi PRIVATE mgcl MPI::MPI_CXX)
  add_dependencies(example_mpi copy_kernels)
  #target_compile_options(example_galerkin PRIVATE "-pg")
  #target_link_options(example_galerkin PRIVATE "-pg")

  # don't use iwyu for example_galerkin
  set_property(
    TARGET example_mpi
    PROPERTY CXX_INCLUDE_WHAT_YOU_USE
  )
endif()