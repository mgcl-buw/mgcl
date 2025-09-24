cmake_minimum_required(VERSION 3.17)
project(MGCL VERSION 0.1.0)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# enable link time optimization (LTO) if supported for Release build type TODO
# do this via tools.cmake?
if(CMAKE_BUILD_TYPE MATCHES Release OR CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT supported OUTPUT error)

  if(supported)
    message(STATUS "IPO / LTO enabled")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(STATUS "IPO / LTO not supported: <${error}>")
  endif()
endif()

# set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")

set(MGCL_catch2_version "v3.4.0")
set(MGCL_nanobench_version "v4.3.11")
set(MGCL_backwardcpp_version "v1.6")
set(MGCL_cpptrace_version "v0.6.3")

if(CMAKE_BUILD_TYPE MATCHES Debug)
  include(cmake/CPM.cmake)
endif()

# add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
# add_link_options(-fsanitize=address -fno-omit-frame-pointer)

add_subdirectory(src)

# Only add other subdirs if mgcl is not used in another project. See
# https://www.foonathan.net/2022/06/cmake-fetchcontent/
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  # We're in the root, define additional targets for developers.

  # Not requiring 3.24 for now (would enable FIND_PACKAGE_ARGS for FetchContent,
  # but Pleiades does not support 3.24 yet) cmake_minimum_required(VERSION 3.24)

  option(MGCL_BUILD_TESTS "whether or not tests should be built" ON)
  option(MGCL_BUILD_BENCHMARKS "whether or not benchmarks should be built" ON)
  option(MGCL_BUILD_EXAMPLES "whether or not examples should be built" ON)

  # ==============================
  # Add CPM as dependency manager for tests and benchmarks. Download it if not already present. see
  # https://github.com/cpm-cmake/CPM.cmake/wiki/Downloading-CPM.cmake-in-CMake
  if(MGCL_BUILD_TESTS OR MGCL_BUILD_BENCHMARKS)
    include(cmake/CPM.cmake)
  endif()
  # ==============================

  if(MGCL_BUILD_TESTS)
    add_subdirectory(test)

    add_dependencies(tests copy_kernels)
    add_dependencies(tests_solving copy_kernels)
    add_dependencies(tests_mpi copy_kernels)
  endif()

  if(MGCL_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)

    add_dependencies(benchmarks copy_kernels)
  endif()

  if(MGCL_BUILD_EXAMPLES)
    add_subdirectory(examples)
  endif()
endif()
