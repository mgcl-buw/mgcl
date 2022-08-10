cmake_minimum_required(VERSION 3.17)
project(MGCL VERSION 0.1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# find_package(CUDAToolkit 8.0 REQUIRED)
find_package(OpenCL REQUIRED)

set(SOURCES_LIB 
clutil.hpp
clutil.cpp
cuboid.hpp
cuboid.cpp
ghostscl.hpp
ghostscl.cpp
jacobicl.hpp
jacobicl.cpp
mgcl.hpp
mgcl.cpp
stencilcl.hpp
stencilcl.cpp
)

add_library(mgcl ${SOURCES_LIB})
# target_link_libraries(main PRIVATE CUDA::OpenCL)
# target_link_libraries(main OpenCL::OpenCL)
# target_link_options(main PRIVATE -lOpenCL)
target_link_libraries(mgcl PRIVATE "-lOpenCL")

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

# installing catch2: https://github.com/catchorg/Catch2/blob/devel/docs/cmake-integration.md#installing-catch2-from-git-repository
find_package(Catch2 3 REQUIRED)

# These tests can use the Catch2-provided main
set(SOURCES_TEST 
test/test_cuboid.cpp
)

add_executable(tests ${SOURCES_TEST})
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)
target_link_libraries(tests PRIVATE mgcl)
