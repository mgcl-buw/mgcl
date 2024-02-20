# This function places the OpenCL kernel source code from mgcl.cl into the mgcl_kernel.hpp.in file
function(substitude_kernel_source cl_file cpp_file_template cpp_file)
    # Define the paths to the two files
    set(output_file ${cl_file})
    
    message(STATUS "MGCL: substitude_kernel_source: ${output_file} -> ${cpp_file}")

    # Read the content of file_b.txt
    file(READ ${output_file} output_content)

    # Replace the specific pattern inside file_a.txt with the content of file_b.txt
    file(READ ${cpp_file_template} input_content)
    string(REPLACE "<replaceme>" "${output_content}" modified_content "${input_content}")

    # Write the modified content back to file_a.txt
    file(WRITE ${cpp_file} "${modified_content}")
endfunction()

# These parameters need to be set when executing as a CMake script, e.g.
# "-Dcl_file=${CMAKE_CURRENT_SOURCE_DIR}/mgcl/mgcl.cl" etc.
substitude_kernel_source(
    "${cl_file}"
    "${cpp_file_template}"
    "${cpp_file}"
)