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
