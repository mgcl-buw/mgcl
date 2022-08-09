#include "clutil.hpp"
#include "cuboid.hpp"
#include "ghostscl.hpp"

#include <fstream>
#include <string.h>

/* Inits OpenCL stuff for mgcl */
int mgcl_init_opencl(mgcl_config *conf, mgcl_level_data *data)
{
    int err, i;
    cl_uint numPlatforms;
    cl_device_id device_id;

    // initialize opencl stuff if buffers should not be reused
    if (!conf->reuse_opencl_buffers && !conf->copy_buffer_data)
    {
        // Find number of platforms
        err = clGetPlatformIDs(0, NULL, &numPlatforms);
        mgclCheckError(err, "Finding platforms");
        if (numPlatforms == 0)
        {
            printf("Found 0 platforms!\n");
            return EXIT_FAILURE;
        }

        // Get all platforms
        cl_platform_id Platform[numPlatforms];
        err = clGetPlatformIDs(numPlatforms, Platform, NULL);
        mgclCheckError(err, "Getting platforms");

        cl_char device_name_available[1024] = {0}; // string to hold name of compute device

        // take first device that conforms given device_type and name
        for (i = 0; i < numPlatforms; i++)
        {
            err = clGetDeviceIDs(Platform[i], conf->device_type, 1, &device_id, NULL);
            if (err == CL_SUCCESS)
            {
                if (conf->device_name != "" && conf->device_name != "default")
                {
                    err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name_available),
                                          &device_name_available, NULL);
                    if (err != CL_SUCCESS)
                    {
                        printf("Error: Failed to access device name!\n");
                        return EXIT_FAILURE;
                    }

                    // continue to next device if name doesn't fit
                    if (strstr((char *)device_name_available, conf->device_name.c_str()) == NULL)
                        continue;
                }

                conf->device_id = device_id;
                break;
            }
        }

        if (device_id == NULL)
            mgclCheckError(err, "Finding a device");

        err = mgcl_output_device_info(device_id);
        mgclCheckError(err, "Printing device output");

        // Create a compute context
        conf->context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
        mgclCheckError(err, "Creating context");

        // Create a command queue
        conf->commands = clCreateCommandQueue(conf->context, device_id, 0, &err);
        mgclCheckError(err, "Creating command queue");
    }

    // read kernel source
    std::string filename = conf->kernel_dir + "mgcl.cl";
    const char *KernelSource = mgcl_load_kernel_source(filename.c_str());
    if (KernelSource == NULL)
        return EXIT_FAILURE;

    // Create the compute program from the source buffer
    conf->program = clCreateProgramWithSource(conf->context, 1, &KernelSource, NULL, &err);
    mgclCheckError(err, "Creating program");

    // Build the program
    err = clBuildProgram(conf->program, 0, NULL, "-cl-fast-relaxed-math", NULL, NULL);
    if (err != CL_SUCCESS)
    {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(conf->program, conf->device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *)malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(conf->program, conf->device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
        return EXIT_FAILURE;
    }

    // init stencil_values buffer to null for first level
    data[0].d_stencil_values = NULL;

    // create d_v_in and d_f buffers on level zero and copy data to it only if buffers should not be reused
    if (conf->reuse_opencl_buffers)
    {
        data[0].d_v_in = conf->d_v;
        data[0].d_f = conf->d_f;
        data[0].d_stencil_values = conf->d_stencil_values;
    }
    else if (conf->copy_buffer_data)
    {
        data[0].d_v_in = clCreateBuffer(conf->context, CL_MEM_READ_WRITE,
                                        sizeof(double) * data[0].m * data[0].n * data[0].o, NULL, &err);
        data[0].d_f = clCreateBuffer(conf->context, CL_MEM_READ_WRITE,
                                     sizeof(double) * data[0].m * data[0].n * data[0].o, NULL, &err);
        // TODO stencil_values
        mgcl_copy_input_buffers(conf, &data[0]);
    }
    else
    {
        int pointer_flag = conf->device_type == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
        data[0].d_v_in = clCreateBuffer(conf->context, CL_MEM_READ_WRITE | pointer_flag,
                                        sizeof(double) * data[0].m * data[0].n * data[0].o, data[0].v[0][0], &err);
        data[0].d_f = clCreateBuffer(conf->context, CL_MEM_READ_WRITE | pointer_flag,
                                     sizeof(double) * data[0].m * data[0].n * data[0].o, data[0].f[0][0], &err);

        // create buffers for stencil values if no fixed stencil shall be used
        if (conf->stencil_values)
            data[0].d_stencil_values =
                clCreateBuffer(conf->context, CL_MEM_READ_WRITE | pointer_flag,
                               sizeof(double) * data[0].m * data[0].n * data[0].o * conf->stencil_size_multiplier,
                               data[0].stencil_values[0][0], &err);
    }
    data[0].d_v_out = clCreateBuffer(conf->context, CL_MEM_READ_WRITE,
                                     sizeof(double) * data[0].m * data[0].n * data[0].o, NULL, &err);
    data[0].d_r = clCreateBuffer(conf->context, CL_MEM_READ_WRITE, sizeof(double) * data[0].m * data[0].n * data[0].o,
                                 NULL, &err);

    // create buffers for higher levels. Data never gets copied back to host so no host pointers are specified
    for (int level = 1; level < conf->maxlevel; level++)
    {
        data[level].d_v_in = clCreateBuffer(conf->context, CL_MEM_READ_WRITE,
                                            sizeof(double) * data[level].m * data[level].n * data[level].o, NULL, &err);
        data[level].d_f = clCreateBuffer(conf->context, CL_MEM_READ_WRITE,
                                         sizeof(double) * data[level].m * data[level].n * data[level].o, NULL, &err);
        data[level].d_v_out =
            clCreateBuffer(conf->context, CL_MEM_READ_WRITE,
                           sizeof(double) * data[level].m * data[level].n * data[level].o, NULL, &err);
        data[level].d_r = clCreateBuffer(conf->context, CL_MEM_READ_WRITE,
                                         sizeof(double) * data[level].m * data[level].n * data[level].o, NULL, &err);

        if (conf->stencil_values != NULL)
        {
            if (conf->restrict_prolongate_stencil)
                data[level].d_stencil_values = clCreateBuffer(conf->context, CL_MEM_READ_WRITE,
                                                              sizeof(double) * data[level].m * data[level].n *
                                                                  data[level].o * conf->stencil_size_multiplier,
                                                              NULL, &err);
            else
                data[level].d_stencil_values = data[0].d_stencil_values;
        }
    }
    mgclCheckError(err, "Creating device buffers");

    err = mgcl_update_ghosts(conf, data[0].d_f, data[0].m, data[0].n, data[0].o, conf->ghosts, conf->ghosts,
                             conf->ghosts);
    mgclCheckError(err, "Updating ghosts of d_f");

    if (data[0].d_stencil_values)
    {
        err = mgcl_update_ghosts(conf, data[0].d_stencil_values, data[0].m, data[0].n,
                                 data[0].o * conf->stencil_size_multiplier, conf->ghosts, conf->ghosts,
                                 conf->ghosts * conf->stencil_size_multiplier);
        mgclCheckError(err, "Updating ghosts of d_stencil_values");
    }

    return CL_SUCCESS;
}

/* Releases OpenCL stuff */
void mgcl_release_opencl(mgcl_config *conf, mgcl_level_data *data)
{
    // release buffer of v_in and f only if it was not reused
    if (!conf->reuse_opencl_buffers)
    {
        clReleaseMemObject(data[0].d_v_in);
        clReleaseMemObject(data[0].d_f);

        if (conf->stencil_values || conf->d_stencil_values)
            clReleaseMemObject(data[0].d_stencil_values);
    }
    clReleaseMemObject(data[0].d_v_out);
    clReleaseMemObject(data[0].d_r);

    for (int level = 1; level < conf->maxlevel; level++)
    {
        clReleaseMemObject(data[level].d_v_in);
        clReleaseMemObject(data[level].d_v_out);
        clReleaseMemObject(data[level].d_r);
        clReleaseMemObject(data[level].d_f);

        if ((conf->stencil_values || conf->d_stencil_values) && conf->restrict_prolongate_stencil)
            clReleaseMemObject(data[level].d_stencil_values);
    }

    clReleaseProgram(conf->program);
    if (!conf->reuse_opencl_buffers && !conf->copy_buffer_data)
    {
        clReleaseCommandQueue(conf->commands);
        clReleaseContext(conf->context);
    }
}

/* Copies input data from conf->d_v and d_f into level 0 data d_v_in and d_f on device, respecting nearfield ghost cell
 * count */
static int mgcl_copy_input_buffers(mgcl_config *conf, mgcl_level_data *data)
{
    int err;
    int m = data->m;
    int n = data->n;
    int o = data->o;
    int ghosts_in = conf->ghosts_in;

    // Create the compute kernel from the program
    cl_kernel kernel = clCreateKernel(conf->program, "copy_input_data", &err);
    mgclCheckError(err, "Creating copy input data kernel");

    // assign kernel arguments
    int pos = 0;
    err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &conf->d_v);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &data->d_v_in);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &conf->d_f);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &data->d_f);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_in);
    mgclCheckError(err, "Setting copy input data kernel arguments");

    // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
    size_t global[3] = {static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(o)};
    const size_t local[3] = {static_cast<size_t>(m > 4 ? 4 : m), static_cast<size_t>(n > 4 ? 4 : n),
                             static_cast<size_t>(o > 4 ? 4 : o)};

    for (int i = 0; i < 3; i++)
        if (global[i] % local[i] != 0)
        {
            // printf("padding global size %d from %ld to ", i, global[i]);
            global[i] += local[i] - (global[i] % local[i]);
            // printf("%ld (multiple of %ld)\n", global[i], local[i]);
        }

    err = clEnqueueNDRangeKernel(conf->commands, kernel, 3, NULL, global, local, 0, NULL, NULL);
    mgclCheckError(err, "Enqueueing copy input data kernel");

    clReleaseKernel(kernel);
    return err;
}

/* Copies output data from d_v_in into conf->d_v on device, respecting nearfield ghost cell count */
int mgcl_copy_output_buffers(mgcl_config *conf, mgcl_level_data *data)
{
    int err;
    int m = data->m;
    int n = data->n;
    int o = data->o;
    int ghosts_in = conf->ghosts_in;

    // Create the compute kernel from the program
    cl_kernel kernel = clCreateKernel(conf->program, "copy_output_data", &err);
    mgclCheckError(err, "Creating copy output data kernel");

    // assign kernel arguments
    int pos = 0;
    err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &conf->d_v);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &data->d_v_in);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_in);
    mgclCheckError(err, "Setting copy output data kernel arguments");

    // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
    size_t global[3] = {static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(o)};
    const size_t local[3] = {static_cast<size_t>(m > 4 ? 4 : m), static_cast<size_t>(n > 4 ? 4 : n),
                             static_cast<size_t>(o > 4 ? 4 : o)};

    for (int i = 0; i < 3; i++)
        if (global[i] % local[i] != 0)
        {
            // printf("padding global size %d from %ld to ", i, global[i]);
            global[i] += local[i] - (global[i] % local[i]);
            // printf("%ld (multiple of %ld)\n", global[i], local[i]);
        }

    err = clEnqueueNDRangeKernel(conf->commands, kernel, 3, NULL, global, local, 0, NULL, NULL);
    mgclCheckError(err, "Enqueueing copy output data kernel");

    clReleaseKernel(kernel);
    return err;
}

/* loads kernel source from file into src */
static char *mgcl_load_kernel_source(const char *file)
{
    FILE *fp;
    char *src;
    size_t source_size, program_size;

    fp = fopen(file, "rb");
    if (!fp)
    {
        printf("Failed to load kernel file: %s\n", file);
        return NULL;
    }

    fseek(fp, 0, SEEK_END);
    program_size = ftell(fp);
    rewind(fp);
    src = (char *)malloc(program_size + 1);
    src[program_size] = '\0';
    size_t ret = fread(src, sizeof(char), program_size, fp);
    fclose(fp);
    return src;
}

static int mgcl_output_device_info(cl_device_id device_id)
{
    int err;                         // error code returned from OpenCL calls
    cl_device_type device_type;      // Parameter defining the type of the compute device
    cl_uint comp_units;              // the max number of compute units on a device
    cl_char vendor_name[1024] = {0}; // string to hold vendor name for compute device
    cl_char device_name[1024] = {0}; // string to hold name of compute device

    err = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(device_name), &device_name, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to access device name!\n");
        return EXIT_FAILURE;
    }
    printf("Using OpenCL device %s ", device_name);

    err = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to access device type information!\n");
        return EXIT_FAILURE;
    }
    if (device_type == CL_DEVICE_TYPE_GPU)
        printf("GPU from ");

    else if (device_type == CL_DEVICE_TYPE_CPU)
        printf("\n CPU from ");

    else
        printf("\n non CPU or GPU processor from ");

    err = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR, sizeof(vendor_name), &vendor_name, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to access device vendor name!\n");
        return EXIT_FAILURE;
    }
    printf("%s", vendor_name);

    err = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &comp_units, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to access device number of compute units !\n");
        return EXIT_FAILURE;
    }
    printf(" with a max of %d compute units \n", comp_units);

    return CL_SUCCESS;
}

/* Reads a buffer from device and writes into out. If out is NULL, it is allocated.
 * Calls clFinish before reading buffer.
 * Returns error of opencl calls. */
int mgcl_read_buffer(mgcl_config *conf, double ****out, cl_mem d_buf, int m, int n, int o)
{
    int err;
    err = clFinish(conf->commands);
    mgclCheckError(err, "Waiting for kernel to finish");

    if (!(*out))
        *out = cuboid_alloc(m, n, o);

    err =
        clEnqueueReadBuffer(conf->commands, d_buf, CL_TRUE, 0, sizeof(double) * m * n * o, out[0][0][0], 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output arrays from device!\n%s\n", mgcl_err_code(err));
    }

    return err;
}

/* reads a buffer from device and prints it */
void mgcl_print_buffer(mgcl_config *conf, cl_mem d_buf, int m, int n, int o)
{
    printf("printing buffer with size %d,%d,%d\n", m, n, o);
    double ***out = NULL;
    mgcl_read_buffer(conf, &out, d_buf, m, n, o);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < o; k++)
            {
                printf("i,j,k, %d,%d,%d val = %f\n", i, j, k, out[i][j][k]);
            }
    cuboid_free(out, m, n, o);
}
