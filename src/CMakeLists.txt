find_package(OpenCL REQUIRED)

if (MGCL_USE_MPI)
  find_package(MPI REQUIRED)
  add_compile_definitions("MGCL_USE_MPI")
endif()

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
util.hpp
util.cpp
mpi_data.hpp
)

if (MGCL_USE_MPI)
  list(APPEND SOURCES_LIB mpi_data.cpp)
endif()

add_library(mgcl ${SOURCES_LIB})
target_link_libraries(mgcl PRIVATE OpenCL::OpenCL)

if (MGCL_USE_MPI)
  target_link_libraries(mgcl PRIVATE MPI::MPI_CXX)
endif()

# Uncomment this to get Debug output in mgcl
# target_compile_definitions(mgcl PRIVATE MGCL_DEBUG)

# copy kernel file to build directory
add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/src/mgcl.cl"
  "${CMAKE_BINARY_DIR}/test/mgcl.cl"
  "${CMAKE_BINARY_DIR}/benchmarks/mgcl.cl"
  "${CMAKE_BINARY_DIR}/examples/mgcl.cl"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/mgcl.cl" "${CMAKE_BINARY_DIR}/src"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/mgcl.cl" "${CMAKE_BINARY_DIR}/test"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/mgcl.cl" "${CMAKE_BINARY_DIR}/benchmarks"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/mgcl.cl" "${CMAKE_BINARY_DIR}/examples"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mgcl.cl"
  )

# Custom target for activate the custom command above
add_custom_target(copy_kernels DEPENDS "${CMAKE_BINARY_DIR}/src/mgcl.cl")

# Whenever target 'mgcl' is requested, the custom target will be evaluated.
add_dependencies(mgcl copy_kernels)
