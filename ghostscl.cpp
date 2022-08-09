#include "ghostscl.hpp"
#include "clutil.hpp"
#include <CL/cl.h>

// TODO ghosts parameter
/* updates ghost cells for periodic boundary condition
 * m,n,o are dimensions of real grid without ghost cells */
void update_ghosts_seq(double ***v, int m, int n, int o, int ghosts_m, int ghosts_n, int ghosts_o)
{
    // sending data in x-direction
    for (int i = 0; i < n + 2 * ghosts_n; i++)
        for (int j = 0; j < o + 2 * ghosts_o; j++)
            for (int k = 0; k < ghosts_m; k++)
            {
                v[k][i][j] = v[m + k][i][j];                       // left ghost cell = right real cell
                v[m + ghosts_m + k][i][j] = v[ghosts_m + k][i][j]; // right ghost cell = left real cell
            }

    // sending data in y-direction
    for (int i = 0; i < m + 2 * ghosts_m; i++)
        for (int j = 0; j < o + 2 * ghosts_o; j++)
            for (int k = 0; k < ghosts_n; k++)
            {
                v[i][k][j] = v[i][n + k][j];                       // top ghost cell = bottom real cell
                v[i][n + ghosts_n + k][j] = v[i][ghosts_n + k][j]; // bottom ghost cell = top real cell
            }

    // sending data in z-direction
    for (int i = 0; i < m + 2 * ghosts_m; i++)
        for (int j = 0; j < n + 2 * ghosts_n; j++)
            for (int k = 0; k < ghosts_o; k++)
            {
                v[i][j][k] = v[i][j][o + k];                       // front ghost cell = back real cell
                v[i][j][o + ghosts_o + k] = v[i][j][ghosts_o + k]; // back ghost cell = front real cell
            }

    // now send diagonal edges in each direction
    // for (int i = 1; i < m + 1; i++)
    // {
    //     v[i][0][0] = v[i][n][o]; // top front
    //     v[i][0][o+1] = v[i][n][1]; // top back
    //     v[i][n+1][0] = v[i][1][o]; // bottom front
    //     v[i][n+1][o+1] = v[i][1][1]; // bottom back

    //     v[0][i][0] = v[m][i][o]; // front left
    //     v[0][i][o+1] = v[m][i][1]; // back left
    //     v[m+1][i][0] = v[1][i][o]; // front left
    //     v[m+1][i][o+1] = v[1][i][1]; // back right

    //     v[0][0][i] = v[m][n][i]; // top left
    //     v[0][n+1][i] = v[m][1][i]; // bottom left
    //     v[m+1][0][i] = v[1][n][i]; // top right
    //     v[m+1][n+1][i] = v[1][1][i]; // bottom right
    // }

    // // corners
    // v[0][0][0] = v[m][n][o]; // top left front
    // v[0][0][o+1] = v[m][n][1]; // top left back
    // v[0][n+1][0] = v[m][1][o]; // bottom left front
    // v[0][n+1][o+1] = v[m][1][1]; // bottom left back
    // v[m+1][0][0] = v[1][n][o]; // top right front
    // v[m+1][0][o+1] = v[1][n][1]; // top right back
    // v[m+1][n+1][0] = v[1][1][o]; // bottom right front
    // v[m+1][n+1][o+1] = v[1][1][1]; // bottom right back

    // printf("v[1,0,0] = %f\n", v[1][0][0]);
}

/* updates ghost cells on opencl device.
 * m,n,o must be size of ghosted grid.
 * Only enqueues the kernel. Neither waits for kernel to finish nor reads back results */
int mgcl_update_ghosts(mgcl_config *conf, cl_mem d_v, int m, int n, int o, int ghosts_m, int ghosts_n, int ghosts_o)
{
    int err;

    // Create the compute kernel from the program
    cl_kernel kernel = clCreateKernel(conf->program, "update_ghosts_2d", &err);
    mgclCheckError(err, "Creating kernel");

    // assign kernel arguments
    int pos = 0;
    err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &d_v);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_m);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_n);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_o);
    mgclCheckError(err, "Setting kernel arguments");

    // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
    size_t global[2] = {static_cast<size_t>(n), static_cast<size_t>(o)};
    const size_t local[2] = {static_cast<size_t>(n > 4 ? 4 : n), static_cast<size_t>(o > 4 ? 4 : o)};

    for (int i = 0; i < 2; i++)
        if (global[i] % local[i] != 0)
        {
            // printf("padding global size %d from %ld to ", i, global[i]);
            global[i] += local[i] - (global[i] % local[i]);
            // printf("%ld (multiple of %ld)\n", global[i], local[i]);
        }

    err = clEnqueueNDRangeKernel(conf->commands, kernel, 2, NULL, global, local, 0, NULL, NULL);
    mgclCheckError(err, "Enqueueing kernel");

    clReleaseKernel(kernel);
    return err;
}

/* Tests the update of ghost cells on opencl device.
 * m,n,o must be size of ghosted grid.
 * Creates device buffers, enqueues kernel, waits for completion and reads back results */
int mgcl_update_ghosts_test(mgcl_config *conf, double ***v, int m, int n, int o, int ghosts_m, int ghosts_n,
                            int ghosts_o)
{
    int err;

    // create device buffers
    int pointer_flag = conf->device_type == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
    cl_mem d_v =
        clCreateBuffer(conf->context, CL_MEM_READ_WRITE | pointer_flag, sizeof(double) * m * n * o, v[0][0], &err);

    // call update ghosts
    auto t_start_iter = std::chrono::steady_clock::now();
    mgcl_update_ghosts(conf, d_v, m, n, o, ghosts_m, ghosts_n, ghosts_o);

    // Wait for the commands to complete before stopping the timer
    err = clFinish(conf->commands);
    mgclCheckError(err, "Waiting for kernel to finish");
    auto t_end_iter = mgcl_since(t_start_iter).count() * 1000.0;
    printf("update ghosts on opencl took %2.5lf s\n", t_end_iter);

    // read back results
    err = clEnqueueReadBuffer(conf->commands, d_v, CL_TRUE, 0, sizeof(double) * m * n * o, v[0][0], 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output arrays from device!\n%s\n", mgcl_err_code(err));
        exit(1);
    }

    clReleaseMemObject(d_v);
    return err;
}
