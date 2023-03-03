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
target_link_libraries (mgcl OpenCL::OpenCL)

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
