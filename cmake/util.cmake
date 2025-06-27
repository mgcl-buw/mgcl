# This file contains utility functions.

# Add cppcheck to a target
function (target_add_cppcheck target)
    find_program(CPPCHECK_PATH NAMES cppcheck)
    if(NOT CPPCHECK_PATH)
        message(INFO "MGCL: cppcheck not found.")
    else()
    message(STATUS "MGCL: Using cppcheck for target ${target}.")
        set_target_properties(
            ${target}
            PROPERTIES 
              CMAKE_CXX_CPPCHECK ${CPPCHECK_PATH}
        )
    endif()
endfunction()

# Add include-what-you-use to a target
function (target_add_iwyu target)
    find_program(IWYU_PATH NAMES include-what-you-use iwyu)
    if(NOT IWYU_PATH)
        message(INFO "MGCL: include-what-you-use not found.")
    else()
    message(STATUS "MGCL: Using include-what-you-use for target ${target}.")
        set_target_properties(
            ${target}
            PROPERTIES 
              CMAKE_CXX_INCLUDE_WHAT_YOU_USE ${IWYU_PATH} "-Xiwyu;--cxx17ns"
              CMAKE_C_INCLUDE_WHAT_YOU_USE ${IWYU_PATH} "-Xiwyu;--cxx17ns"
        )
    endif()
endfunction()

# Copies a kernel file to the build directory.
# 
# This function creates a custom target that copies a kernel file from the source
# directory to the build directory. The custom target is then added as a dependency
# of the specified dependency target.
# 
# @param dependency_target The target that depends on the copied kernel file.
# @param filename The name of the kernel file to copy.
# 
function(mgcl_copy_kernel_file dependency_target filename)
  add_custom_command(
    OUTPUT "${CMAKE_BINARY_DIR}/benchmarks/${filename}"
    COMMAND
      ${CMAKE_COMMAND} -E copy
      "${CMAKE_CURRENT_SOURCE_DIR}/${filename}"
      "${CMAKE_BINARY_DIR}/benchmarks"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${filename}"
  )

  add_custom_target(
    copy_${filename}
    DEPENDS "${CMAKE_BINARY_DIR}/benchmarks/${filename}"
  )

  add_dependencies(${dependency_target} copy_${filename})
endfunction()