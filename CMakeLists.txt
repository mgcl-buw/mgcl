cmake_minimum_required(VERSION 3.17)
project(MGCL VERSION 0.1.0)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# enable link time optimization (LTO) if supported for Release build type
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

add_subdirectory(src)
add_subdirectory(thirdparty)
add_subdirectory(test)
add_subdirectory(benchmarks)
add_subdirectory(examples)
