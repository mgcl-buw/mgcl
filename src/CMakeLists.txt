find_package(OpenCL REQUIRED)
find_package(MPI REQUIRED)

# ============================================

# Add cpptrace for debug builds for easier stacktrace printing.
if(CMAKE_BUILD_TYPE MATCHES Debug)
  cpmaddpackage("gh:jeremy-rifkin/cpptrace#${MGCL_cpptrace_version}")
endif()

# ============================================
set(MGCL_PUBLIC_HEADERS
  mgcl/cuboid.hpp
  mgcl/cuboid_gpu.hpp
  mgcl/mgcl.hpp
  mgcl/problem.hpp
  mgcl/opencl_helper.hpp
  mgcl/stencil.hpp
  mgcl/level.hpp
  mgcl/mpi_global_data.hpp
  mgcl/mpi_level_data.hpp
  mgcl/hypercube.hpp
  mgcl/kernel_config.hpp
  mgcl/profiling_data.hpp
  mgcl/buffer_gpu.hpp
  mgcl/cuboid_bs.hpp
  mgcl/cuboid_bs_gpu.hpp
  mgcl/blockstencil.hpp
  mgcl/blockstencil_gpu.hpp
)

set(SOURCES_LIB
  mgcl/cuboid.cpp
  mgcl/cuboid_gpu.cpp
  mgcl/ghostscl.cpp
  mgcl/jacobicl.cpp
  mgcl/problem.cpp
  mgcl/level.cpp
  mgcl/opencl_helper.cpp
  mgcl/multigrid_engine.hpp
  mgcl/multigrid_engine.cpp
  mgcl/prolongation.cpp
  mgcl/restriction.cpp
  mgcl/stencil.cpp
  mgcl/hypercube.cpp
  mgcl/util.hpp
  mgcl/util.cpp
  mgcl/mpi_level_data.cpp
  mgcl/mpi_global_data.cpp
  mgcl/mpi_util.hpp
  mgcl/mpi_util.cpp
  mgcl/mpi_stencil.hpp
  mgcl/mpi_stencil.cpp
  mgcl/mgcl_kernel.hpp
  mgcl/kernel_config.cpp
  mgcl/profiling_data.cpp
  mgcl/buffer_gpu.cpp
  mgcl/cuboid_bs.cpp
  mgcl/cuboid_bs_gpu.cpp
  mgcl/blockstencil.cpp
  mgcl/blockstencil_gpu.cpp
)

add_library(mgcl ${MGCL_PUBLIC_HEADERS} ${SOURCES_LIB})
add_library(mgcl::mgcl ALIAS mgcl)

target_link_libraries(mgcl PUBLIC OpenCL::OpenCL MPI::MPI_CXX)
if(CMAKE_BUILD_TYPE MATCHES Debug AND cpptrace_ADDED)
  # Add -DMGCL_PRINT_TRACE to compile options and link against cpptrace, if available.
  target_compile_definitions(mgcl PRIVATE MGCL_PRINT_TRACE)
  target_link_libraries(mgcl PRIVATE cpptrace::cpptrace)

  # Copy the cpptrace.dll on windows to the same directory as the executable for your_target.
  # Not required if static linking.
  if(WIN32)
    add_custom_command(
      TARGET mgcl POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      $<TARGET_FILE:cpptrace::cpptrace>
      $<TARGET_FILE_DIR:mgcl>
    )
  endif()
endif()

# Set PUBLIC_HEADER property so they get installed.
set_target_properties(mgcl PROPERTIES PUBLIC_HEADER "${MGCL_PUBLIC_HEADERS}")

# Set include directories depending on build or install type.
target_include_directories(mgcl PUBLIC
  $<BUILD_INTERFACE:${mgcl_BINARY_DIR}/src>
  $<BUILD_INTERFACE:${mgcl_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include/mgcl>
)

# Install targets and export interface
include(GNUInstallDirs)
install(TARGETS mgcl EXPORT mgclTargets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mgcl" # include/mgcl
)
install(EXPORT mgclTargets
  NAMESPACE mgcl::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mgcl
)
# Install kernel file into include directory
install(FILES mgcl/mgcl.cl DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/mgcl")

# Write config package files for making mgcl includable in other projects
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/mgclConfigVersion.cmake"
  COMPATIBILITY
    SameMajorVersion
)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/../cmake/mgclConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/mgclConfig.cmake
  INSTALL_DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/mgcl
)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/mgclConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/mgclConfigVersion.cmake"
  DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/cmake/mgcl
)

# Use additional flags for debugging purposes if build type is Debug.
if(CMAKE_BUILD_TYPE MATCHES Debug)
  if(MSVC)
    # warning level 4
    target_compile_options(mgcl PRIVATE /W4)
  else()
    # Use address sanitizer (ASAN) ASAN is not compatible with OpenCL, see
    # https://stackoverflow.com/a/68027496/4108363 A workaround is to set
    # ASAN_OPTIONS=protect_shadow_gap=0. 
    # target_compile_options(mgcl PUBLIC -fsanitize=address -fno-omit-frame-pointer)
    # target_link_options(mgcl PUBLIC -fsanitize=address)

    # additional warnings
    target_compile_options(mgcl PRIVATE -Wall -Wextra -Wno-reorder)

    # enable cppcheck and and iwyu
    include (../cmake/util.cmake)
    target_add_cppcheck(mgcl)
    target_add_iwyu(mgcl)
  endif()
endif()

# Uncomment this to get Debug output in mgcl target_compile_definitions(mgcl
# PRIVATE MGCL_DEBUG)

# copy kernel file to build directory
# TODO check MGCL_BUILD_TESTS etc.
add_custom_command(
  OUTPUT "${CMAKE_BINARY_DIR}/src/mgcl/mgcl.cl"
         "${CMAKE_BINARY_DIR}/test/mgcl.cl"
         "${CMAKE_BINARY_DIR}/benchmarks/mgcl.cl"
         "${CMAKE_BINARY_DIR}/examples/mgcl.cl"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/mgcl/mgcl.cl"
          "${CMAKE_BINARY_DIR}/src"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/mgcl/mgcl.cl"
          "${CMAKE_BINARY_DIR}/test"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/mgcl/mgcl.cl"
          "${CMAKE_BINARY_DIR}/benchmarks"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/mgcl/mgcl.cl"
          "${CMAKE_BINARY_DIR}/examples"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/mgcl/mgcl.cl")

# Custom target for activate the custom command above
add_custom_target(copy_kernels DEPENDS "${CMAKE_BINARY_DIR}/src/mgcl/mgcl.cl")

# Whenever target 'mgcl' is requested, the custom target will be evaluated.
add_dependencies(mgcl copy_kernels)

# Create a custom command to update kernel cpp src whenever the .cl-file changes
add_custom_command(
  OUTPUT 
    "${CMAKE_CURRENT_SOURCE_DIR}/mgcl/mgcl_kernel.hpp"
    
  COMMAND "${CMAKE_COMMAND}" 
    "-Dcl_file=${CMAKE_CURRENT_SOURCE_DIR}/mgcl/mgcl.cl"
    "-Dcpp_file_template=${CMAKE_CURRENT_SOURCE_DIR}/mgcl/mgcl_kernel.hpp.in"
    "-Dcpp_file=${CMAKE_CURRENT_SOURCE_DIR}/mgcl/mgcl_kernel.hpp"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/substitude_kernel_source.cmake"
  
  DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/mgcl/mgcl.cl"
    "${CMAKE_CURRENT_SOURCE_DIR}/mgcl/mgcl_kernel.hpp.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/substitude_kernel_source.cmake"
)
