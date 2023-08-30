find_package(OpenCL REQUIRED)
find_package(MPI REQUIRED)

# Use static analyzers if variables are set.
if(USE_STATIC_ANALYZER)
  include(../cmake/tools.cmake)

  if (CLANG_TIDY)
      clang_tidy(${CLANG_TIDY_ARGS})
    endif()

    if (IWYU)
      include_what_you_use(${IWYU_ARGS})
    endif()

    if (CPPCHECK)
      cppcheck(${CPPCHECK_ARGS})
    endif()
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
mpi_data.cpp
mpi_util.hpp
mpi_util.cpp
)

add_library(mgcl ${SOURCES_LIB})
target_link_libraries(mgcl PRIVATE OpenCL::OpenCL)
target_link_libraries(mgcl PRIVATE MPI::MPI_CXX)

# Use additional flags for debugging purposes if build type is Debug.
if(CMAKE_BUILD_TYPE MATCHES Debug)
  if (MSVC)
    # warning level 4
    target_compile_options(mgcl PRIVATE /W4)
  else()
    # additional warnings
    target_compile_options(mgcl PRIVATE -Wall -Wextra -Wpedantic -Wno-reorder -Wno-vla)
  endif()
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
