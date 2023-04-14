# ==========================
# pmg configuration for comparison in benchmarks
set (SOURCES_PMG
pmg/cuboid.c
pmg/cuboid.h
pmg/galerkin.c
pmg/galerkin.h
pmg/ghosts.c
pmg/ghosts.h
pmg/interpolate.c
pmg/interpolate.h
pmg/jacobi.c
pmg/jacobi.h
pmg/lueqf.c
pmg/lueqf.h
pmg/mg.c
pmg/mg.h
pmg/rectangle.c
pmg/rectangle.h
pmg/restrict.c
pmg/restrict.h
pmg/stencil.c
pmg/stencil.h
)

add_library(pmg ${SOURCES_PMG})

# don't use iwyu for pmg
set_property(
  TARGET pmg
  PROPERTY CXX_INCLUDE_WHAT_YOU_USE
)

# maybe need to run cmake with
# cmake -DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/openmpi
# in order to find MPI libraries
find_package(MPI REQUIRED)
# target_include_directories(pmg SYSTEM PUBLIC ${MPI_INCLUDE_PATH})
set(MPI_CXX_SKIP_MPICXX TRUE)
target_link_libraries(pmg PRIVATE MPI::MPI_C)

# ==========================
find_package(OpenCL REQUIRED)

# old mgcl c configuration for comparison in benchmarks
set (SOURCES_MGCL_C
mgcl_c/clutil.cpp
mgcl_c/clutil.hpp
mgcl_c/cuboid.cpp
mgcl_c/cuboid.hpp
mgcl_c/ghostscl.cpp
mgcl_c/ghostscl.hpp
mgcl_c/jacobicl.cpp
mgcl_c/jacobicl.hpp
mgcl_c/mgcl.cpp
mgcl_c/mgcl.hpp
mgcl_c/stencilcl.cpp
mgcl_c/stencilcl.hpp
)

add_library(mgcl_c ${SOURCES_MGCL_C})

find_package(MPI REQUIRED)
# target_include_directories(mgcl_c SYSTEM PUBLIC ${MPI_INCLUDE_PATH})
add_definitions(-DOMPI_SKIP_MPICXX)
target_link_libraries(mgcl_c PRIVATE MPI::MPI_C)
target_link_libraries (mgcl_c PRIVATE OpenCL::OpenCL)

# don't use iwyu for mgcl_c
set_property(
  TARGET mgcl_c
  PROPERTY CXX_INCLUDE_WHAT_YOU_USE
)

# copy kernel file to build directory
add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/thirdparty/mgcl_old.cl"
  COMMAND ${CMAKE_COMMAND} -E copy
  "${CMAKE_CURRENT_SOURCE_DIR}/mgcl_c/mgcl_old.cl" "${CMAKE_BINARY_DIR}/thirdparty"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mgcl_c/mgcl_old.cl"
)

# Custom target for activate the custom command above
add_custom_target(copy_kernels_mgcl_c DEPENDS "${CMAKE_BINARY_DIR}/thirdparty/mgcl_old.cl")

# Whenever target 'mgcl' is requested, the custom target will be evaluated.
add_dependencies(mgcl_c copy_kernels_mgcl_c)

