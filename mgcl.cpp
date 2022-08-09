#include "mgcl.hpp"
#include "clutil.hpp"
#include "cuboid.hpp"
#include "ghostscl.hpp"
#include "jacobicl.hpp"
#include "stencilcl.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <tgmath.h>

/* Generates and returns a default config object for mgcl. Needs to be freed manually by the user when mgcl has
 * finished. */
void mgcl_generate_config(mgcl_config **outconf)
{
    mgcl_config *conf = new mgcl_config;

    conf->v = NULL;
    conf->f = NULL;
    conf->m = -1;
    conf->n = -1;
    conf->o = -1;
    conf->ghosts = 1;
    conf->ghosts_in = 0;
    conf->nu1 = 2;
    conf->nu2 = 2;
    conf->omega = 0.8;
    conf->maxiter_vcycles = 5;
    conf->tol = 1e-7;
    conf->maxlevel = -1;
    conf->residual_norm = MGCL_L2;
    conf->stencil = MGCL_7POINT;

    conf->device_type = CL_DEVICE_TYPE_DEFAULT;
    conf->kernel_dir = "./";
    conf->device_name = "";
    conf->device_id = NULL;
    conf->commands = NULL;
    conf->context = NULL;
    conf->d_v = NULL;
    conf->d_f = NULL;
    conf->use_opencl = 0;
    conf->reuse_opencl_buffers = 0;
    conf->copy_buffer_data = 0;
    conf->read_results = 0;
    conf->use_local_memory = 0;

    conf->jacobi_wg_size_x = 16;
    conf->jacobi_wg_size_y = 16;
    conf->jacobi_iterations_per_kernel = 3;

    conf->stencil_size_multiplier = 1;
    conf->stencil_values = NULL;
    conf->d_stencil_values = NULL;
    conf->restrict_prolongate_stencil = 1;

    *outconf = conf;
}

// TODO init with zeros
/* Checks conf object and initializes data for each level */
int mgcl_init(mgcl_config *conf, mgcl_level_data **outdata)
{
    const int m = conf->m, n = conf->n, o = conf->o;

    // check mandatory config fields
    if ((conf->v == NULL || conf->f == NULL) && (conf->d_v == NULL || conf->d_f == NULL))
    {
        printf("mgcl: supplied v or f and d_v or d_f is NULL. Aborting.\n");
        return EXIT_FAILURE;
    }

    if (conf->m < 1 || conf->n < 1 || conf->o < 1)
    {
        printf("mgcl: m, n or o not supplied, zero or negative. Aborting.\n");
        return EXIT_FAILURE;
    }

    if (conf->ghosts < 1)
    {
        printf("mgcl: ghosts must be >= 1. Aborting.\n");
        return EXIT_FAILURE;
    }

    if (conf->ghosts_in < 0)
    {
        printf("mgcl: ghosts_in must be >= 0. Aborting.\n");
        return EXIT_FAILURE;
    }

    if (((conf->stencil_values == NULL && !conf->use_opencl) ||
         (conf->d_stencil_values == NULL && conf->reuse_opencl_buffers)) &&
        (conf->stencil == MGCL_7POINT_VARSYM || conf->stencil == MGCL_19POINT_VARSYM ||
         conf->stencil == MGCL_27POINT_VARSYM))
    {
        printf("stencil is set to be varying symmetric but stencil_values is NULL! Aborting.\n");
        return EXIT_FAILURE;
    }

    // set stencil size if stencil is set to a varying symmetric one
    if (conf->stencil == MGCL_7POINT_VARSYM)
        conf->stencil_size_multiplier = 4;
    else if (conf->stencil == MGCL_19POINT_VARSYM)
        conf->stencil_size_multiplier = 7;
    else if (conf->stencil == MGCL_27POINT_VARSYM)
        conf->stencil_size_multiplier = 8;

    // check opencl components if device buffers should be reused
    if (conf->reuse_opencl_buffers || conf->copy_buffer_data)
    {
        if (conf->d_v == NULL || conf->d_f == NULL)
        {
            printf("OpenCL buffers d_v and d_f not set but reuse_opencl_buffers or copy_buffer_data specified. "
                   "Aborting.\n");
            return EXIT_FAILURE;
        }

        if (conf->device_id == NULL)
        {
            printf("reuse_opencl_buffers or copy_buffer_data specified but device ID (mgcl_config.device_id) not set. "
                   "Aborting.\n");
            return EXIT_FAILURE;
        }

        if (conf->commands == NULL)
        {
            printf("reuse_opencl_buffers or copy_buffer_data specified but command queue (mgcl_config.commands) not "
                   "set. Aborting.\n");
            return EXIT_FAILURE;
        }

        if (conf->context == NULL)
        {
            printf("reuse_opencl_buffers or copy_buffer_data specified but context (mgcl_config.context) not set. "
                   "Aborting.\n");
            return EXIT_FAILURE;
        }

        // check size of buffers
        if (conf->reuse_opencl_buffers)
        {
            size_t bufsize;
            int sizeNeeded = sizeof(double) * (m + 2 * conf->ghosts) * (n + 2 * conf->ghosts) * (o + 2 * conf->ghosts);
            int err = clGetMemObjectInfo(conf->d_v, CL_MEM_SIZE, sizeof(size_t), &bufsize, NULL);
            mgclCheckError(err, "Querying buffer size of d_v\n");
            if (bufsize != sizeNeeded)
            {
                printf("OpenCL buffer d_v has wrong size (%ld but need %d)\n", bufsize, sizeNeeded);
                return EXIT_FAILURE;
            }

            err = clGetMemObjectInfo(conf->d_f, CL_MEM_SIZE, sizeof(size_t), &bufsize, NULL);
            mgclCheckError(err, "Querying buffer size of d_f\n");
            if (bufsize != sizeNeeded)
            {
                printf("OpenCL buffer d_f has wrong size (%ld but need %d)\n", bufsize, sizeNeeded);
                return EXIT_FAILURE;
            }
        }
    }

    // find max level or use user specified one
    int minsize = m < n ? m : n;
    minsize = minsize < o ? minsize : o;
    int maxlevel = log2(minsize) + 1;

    if (conf->maxlevel >= 0) // user has specified a maxlevel
    {
        if (maxlevel < conf->maxlevel) // user specified maxlevel is too high
        {
            printf("user specified maxlevel of %d is too high! Using %d instead.\n", conf->maxlevel, maxlevel);
            conf->maxlevel = maxlevel;
        }
    }
    else
        conf->maxlevel = maxlevel; // use calculated maxlevel
    printf("maxlevel = %d\n", conf->maxlevel);

    mgcl_level_data *data = new mgcl_level_data[maxlevel];

    for (int level = 0; level < maxlevel; level++)
    {
        if (level == 0)
        {
            int mg = data[level].m = m + 2 * conf->ghosts;
            int ng = data[level].n = n + 2 * conf->ghosts;
            int og = data[level].o = o + 2 * conf->ghosts;

            data[level].v = NULL; // initialize to null for check later in finish

            // create ghosted arrays for v and f on host if device buffer should not be reused
            if (!conf->reuse_opencl_buffers && !conf->copy_buffer_data)
            {
                data[level].v = cuboid_alloc(mg, ng, og);
                data[level].f = cuboid_alloc(mg, ng, og);
                // copy initial input data from conf into mgcl data struct
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                        for (int k = 0; k < o; k++)
                        {
                            data[level].v[i + conf->ghosts][j + conf->ghosts][k + conf->ghosts] =
                                conf->v[i + conf->ghosts_in][j + conf->ghosts_in][k + conf->ghosts_in];
                            data[level].f[i + conf->ghosts][j + conf->ghosts][k + conf->ghosts] =
                                conf->f[i + conf->ghosts_in][j + conf->ghosts_in][k + conf->ghosts_in];
                        }

                update_ghosts_seq(data[level].f, m, n, o, conf->ghosts, conf->ghosts, conf->ghosts);

                // allocate initial stencil_values, including ghost cells, if varying symmetric stencil shall be used
                if (conf->stencil_values || conf->d_stencil_values)
                {
                    data[level].stencil_values = cuboid_alloc(mg, ng, og * conf->stencil_size_multiplier);

                    // copy initial input stencil data from conf into mgcl data struct
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < n; j++)
                            for (int k = 0; k < o * conf->stencil_size_multiplier; k++)
                            {
                                data[level].stencil_values[i + conf->ghosts][j + conf->ghosts]
                                                          [k + conf->ghosts * conf->stencil_size_multiplier] =
                                    conf->stencil_values[i + conf->ghosts_in][j + conf->ghosts_in]
                                                        [k + conf->ghosts_in * conf->stencil_size_multiplier];
                            }

                    update_ghosts_seq(data[level].stencil_values, m, n, o * conf->stencil_size_multiplier, conf->ghosts,
                                      conf->ghosts, conf->ghosts * conf->stencil_size_multiplier);
                }
            }

            // r on host is only needed if opencl should not be used
            if (!conf->use_opencl)
            {
                data[level].r = cuboid_alloc(mg, ng, og);
            }
        }
        else
        {
            // ghosted sizes of current level's grid
            int mg = (data[level - 1].m - 2 * conf->ghosts) / 2 + 2 * conf->ghosts;
            int ng = (data[level - 1].n - 2 * conf->ghosts) / 2 + 2 * conf->ghosts;
            int og = (data[level - 1].o - 2 * conf->ghosts) / 2 + 2 * conf->ghosts;
            data[level].m = mg;
            data[level].n = ng;
            data[level].o = og;

            if (!conf->use_opencl)
            {
                data[level].v = cuboid_alloc(mg, ng, og);
                data[level].f = cuboid_alloc(mg, ng, og);
                data[level].r = cuboid_alloc(mg, ng, og);

                if (conf->stencil_values != NULL)
                {
                    if (conf->restrict_prolongate_stencil)
                        data[level].stencil_values = cuboid_alloc(mg, ng, og * conf->stencil_size_multiplier);
                    else
                        data[level].stencil_values = data[0].stencil_values;
                }
            }
        }

        data[level].h = 1.0 / (m * m);
    }
    *outdata = data;
    return CL_SUCCESS;
}

/* Restricts residual to coarser grid using full-weighted restriction operator.
 * m, n and o must be the dimensions of the coarser grid without ghost cells. */
void mgcl_restrict_seq(double ***r_fine, double ***f_coarse, int m, int n, int o, int ghosts)
{
    update_ghosts_seq(r_fine, m * 2, n * 2, o * 2, ghosts, ghosts, ghosts);
    int ioff = 1, joff = 1, koff = 1; // offset grows by 1 for each step
    int i2, j2, k2;
    for (int i = ghosts; i < m + ghosts; i++, ioff++)
    {
        i2 = i + ioff; // == i*2+ghosts+1
        joff = 1;
        for (int j = ghosts; j < n + ghosts; j++, joff++)
        {
            j2 = j + joff;
            koff = 1;
            for (int k = ghosts; k < o + ghosts; k++, koff++)
            {
                k2 = k + koff;
                f_coarse[i][j][k] =
                    0.125 * r_fine[i2][j2][k2] // self
                                               // direct neighbours
                    + 0.0625 * r_fine[i2 - 1][j2][k2] + 0.0625 * r_fine[i2 + 1][j2][k2] +
                    0.0625 * r_fine[i2][j2 - 1][k2] + 0.0625 * r_fine[i2][j2 + 1][k2] +
                    0.0625 * r_fine[i2][j2][k2 - 1] +
                    0.0625 * r_fine[i2][j2][k2 + 1]
                    // edge midpoints xy-plane
                    + 0.03125 * r_fine[i2 - 1][j2 - 1][k2] + 0.03125 * r_fine[i2 - 1][j2 + 1][k2] +
                    0.03125 * r_fine[i2 + 1][j2 - 1][k2] +
                    0.03125 * r_fine[i2 + 1][j2 + 1][k2]
                    // edge midpoints xz-plane
                    + 0.03125 * r_fine[i2 - 1][j2][k2 - 1] + 0.03125 * r_fine[i2 - 1][j2][k2 + 1] +
                    0.03125 * r_fine[i2 + 1][j2][k2 - 1] +
                    0.03125 * r_fine[i2 + 1][j2][k2 + 1]
                    // edge midpoints yz-plane
                    + 0.03125 * r_fine[i2][j2 - 1][k2 - 1] + 0.03125 * r_fine[i2][j2 - 1][k2 + 1] +
                    0.03125 * r_fine[i2][j2 + 1][k2 - 1] +
                    0.03125 * r_fine[i2][j2 + 1][k2 + 1]
                    // corners
                    + 0.015625 * r_fine[i2 - 1][j2 - 1][k2 - 1] + 0.015625 * r_fine[i2 - 1][j2 - 1][k2 + 1] +
                    0.015625 * r_fine[i2 - 1][j2 + 1][k2 - 1] + 0.015625 * r_fine[i2 - 1][j2 + 1][k2 + 1] +
                    0.015625 * r_fine[i2 + 1][j2 - 1][k2 - 1] + 0.015625 * r_fine[i2 + 1][j2 - 1][k2 + 1] +
                    0.015625 * r_fine[i2 + 1][j2 + 1][k2 - 1] + 0.015625 * r_fine[i2 + 1][j2 + 1][k2 + 1];
            }
        }
    }
    update_ghosts_seq(f_coarse, m, n, o, ghosts, ghosts, ghosts);
}

/* Restricts from fine to coarse grid. m,n,o is size of coarse ghosted grid.
 * Creates buffers, starts kernel and reads back results */
void mgcl_restrict_test(mgcl_config *conf, double ***fine, double ***coarse, int m, int n, int o, int ghosts)
{
    int err;
    int mf = (m - 2 * ghosts) * 2 + 2 * ghosts, nf = (n - 2 * ghosts) * 2 + 2 * ghosts,
        of = (o - 2 * ghosts) * 2 + 2 * ghosts; // fine grid sizes

    // create device buffers
    int pointer_flag = conf->device_type == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
    cl_mem d_fine =
        clCreateBuffer(conf->context, CL_MEM_READ_ONLY | pointer_flag, sizeof(double) * mf * nf * of, fine[0][0], &err);
    cl_mem d_coarse =
        clCreateBuffer(conf->context, CL_MEM_WRITE_ONLY | pointer_flag, sizeof(double) * m * n * o, coarse[0][0], &err);

    // create level data for both grids
    mgcl_level_data data_fine;
    data_fine.d_r = d_fine;
    data_fine.m = mf;
    data_fine.n = nf;
    data_fine.o = of;

    mgcl_level_data data_coarse;
    data_coarse.d_f = d_coarse;
    data_coarse.m = m;
    data_coarse.n = n;
    data_coarse.o = o;

    auto t_start_iter = std::chrono::steady_clock::now();

    mgcl_restrict(conf, &data_fine, &data_coarse);

    // Wait for the commands to complete before stopping the timer
    err = clFinish(conf->commands);
    mgclCheckError(err, "Waiting for kernel to finish");
    double t_end_iter = mgcl_since(t_start_iter).count() * 1000.0;
    printf("restrict on opencl took %2.5lf s\n", t_end_iter);

    // read back results
    err = clEnqueueReadBuffer(conf->commands, d_coarse, CL_TRUE, 0, sizeof(double) * m * n * o, coarse[0][0], 0, NULL,
                              NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output arrays from device!\n%s\n", mgcl_err_code(err));
        exit(1);
    }

    clReleaseMemObject(d_fine);
    clReleaseMemObject(d_coarse);
}

/* Restricts from fine to coarse grid.
 * Doesn't create buffers or copy memory from or to device.
 * Writes results into d_v_in. */
void mgcl_restrict(mgcl_config *conf, mgcl_level_data *fine, mgcl_level_data *coarse)
{
    int err;
    int mreal = coarse->m - 2 * conf->ghosts;
    int nreal = coarse->n - 2 * conf->ghosts;
    int oreal = coarse->o - 2 * conf->ghosts;

    // Create the compute kernel from the program
    cl_kernel kernel = clCreateKernel(conf->program, "restrict_to_coarse", &err);
    mgclCheckError(err, "Creating kernel");

    // assign kernel arguments
    int pos = 0;
    err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &fine->d_r);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &coarse->d_f);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &coarse->m);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &coarse->n);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &coarse->o);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &conf->ghosts);
    mgclCheckError(err, "Setting kernel arguments");

    // one work-item per cell (excluding ghost cells). Pad global sizes to fit to local sizes
    size_t global[3] = {static_cast<size_t>(mreal), static_cast<size_t>(nreal), static_cast<size_t>(oreal)};
    const size_t local[3] = {static_cast<size_t>(mreal > 4 ? 4 : mreal), static_cast<size_t>(nreal > 4 ? 4 : nreal),
                             static_cast<size_t>(oreal > 4 ? 4 : oreal)};

    for (int i = 0; i < 3; i++)
        if (global[i] % local[i] != 0)
        {
            // printf("padding global size %d from %ld to ", i, global[i]);
            global[i] += local[i] - (global[i] % local[i]);
            // printf("%ld (multiple of %ld)\n", global[i], local[i]);
        }

    err = mgcl_update_ghosts(conf, fine->d_r, fine->m, fine->n, fine->o, conf->ghosts, conf->ghosts, conf->ghosts);
    mgclCheckError(err, "Updating fine ghosts");
    err = clEnqueueNDRangeKernel(conf->commands, kernel, 3, NULL, global, local, 0, NULL, NULL);
    mgclCheckError(err, "Enqueueing restriction kernel");
    err = mgcl_update_ghosts(conf, coarse->d_f, coarse->m, coarse->n, coarse->o, conf->ghosts, conf->ghosts,
                             conf->ghosts);
    mgclCheckError(err, "Updating coarse ghosts");

    clReleaseKernel(kernel);
}

/* Prolongates from coarse to fine grid.
 * m, n and o must be dimensions of the fine grid without ghost cells. */
void mgcl_prolongate_seq(double ***fine, double ***coarse, int m, int n, int o, int ghosts)
{
    update_ghosts_seq(coarse, m / 2, n / 2, o / 2, ghosts, ghosts, ghosts);
    int ioff = 1, joff = 1, koff = 1; // offset grows by 1 for each step
    int i2, j2, k2;
    for (int i = ghosts; i < m / 2 + ghosts; i++, ioff++)
    {
        i2 = i + ioff; // == i*2+ghosts+1
        joff = 1;
        for (int j = ghosts; j < n / 2 + ghosts; j++, joff++)
        {
            j2 = j + joff;
            koff = 1;
            for (int k = ghosts; k < o / 2 + ghosts; k++, koff++)
            {
                k2 = k + koff;
                fine[i2][j2][k2] = coarse[i][j][k];

                fine[i2][j2][k2 - 1] = 0.5 * (coarse[i][j][k] + coarse[i][j][k - 1]);
                fine[i2][j2 - 1][k2] = 0.5 * (coarse[i][j][k] + coarse[i][j - 1][k]);
                fine[i2 - 1][j2][k2] = 0.5 * (coarse[i][j][k] + coarse[i - 1][j][k]);

                fine[i2][j2 - 1][k2 - 1] =
                    0.25 * (coarse[i][j][k] + coarse[i][j][k - 1] + coarse[i][j - 1][k] + coarse[i][j - 1][k - 1]);
                fine[i2 - 1][j2][k2 - 1] =
                    0.25 * (coarse[i][j][k] + coarse[i][j][k - 1] + coarse[i - 1][j][k] + coarse[i - 1][j][k - 1]);
                fine[i2 - 1][j2 - 1][k2] =
                    0.25 * (coarse[i][j][k] + coarse[i][j - 1][k] + coarse[i - 1][j][k] + coarse[i - 1][j - 1][k]);

                fine[i2 - 1][j2 - 1][k2 - 1] =
                    0.125 * (coarse[i][j][k] + coarse[i][j][k - 1] + coarse[i][j - 1][k] + coarse[i][j - 1][k - 1] +
                             coarse[i - 1][j][k] + coarse[i - 1][j][k - 1] + coarse[i - 1][j - 1][k] +
                             coarse[i - 1][j - 1][k - 1]);
            }
        }
    }
    // update_ghosts_seq(fine, m, n, o);
}

/* Prolongates from coarse to fine grid. m,n,o is size of fine ghosted grid.
 * Creates buffers, starts kernel and reads back results */
void mgcl_prolongate_test(mgcl_config *conf, double ***fine, double ***coarse, int m, int n, int o, int ghosts)
{
    int err;
    int mc = (m - 2 * ghosts) / 2 + 2 * ghosts, nc = (n - 2 * ghosts) / 2 + 2 * ghosts,
        oc = (o - 2 * ghosts) / 2 + 2 * ghosts; // coarse grid sizes

    // create device buffers
    int pointer_flag = conf->device_type == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
    cl_mem d_fine_r =
        clCreateBuffer(conf->context, CL_MEM_READ_WRITE | pointer_flag, sizeof(double) * m * n * o, fine[0][0], &err);
    cl_mem d_fine_v =
        clCreateBuffer(conf->context, CL_MEM_READ_WRITE | pointer_flag, sizeof(double) * m * n * o, fine[0][0], &err);
    cl_mem d_coarse = clCreateBuffer(conf->context, CL_MEM_READ_WRITE | pointer_flag, sizeof(double) * mc * nc * oc,
                                     coarse[0][0], &err);

    // create level data for both grids
    mgcl_level_data data_fine;
    data_fine.d_r = d_fine_r;
    data_fine.d_v_in = d_fine_v;
    data_fine.m = m;
    data_fine.n = n;
    data_fine.o = o;

    mgcl_level_data data_coarse;
    data_coarse.d_v_in = d_coarse;
    data_coarse.m = mc;
    data_coarse.n = nc;
    data_coarse.o = oc;

    auto t_start_iter = std::chrono::steady_clock::now();

    mgcl_prolongate(conf, &data_fine, &data_coarse);

    // Wait for the commands to complete before stopping the timer
    err = clFinish(conf->commands);
    mgclCheckError(err, "Waiting for kernel to finish");
    auto t_end_iter = mgcl_since(t_start_iter).count() * 1000.0;
    printf("prolongate on opencl took %2.5lf s\n", t_end_iter);

    // read back results
    err = clEnqueueReadBuffer(conf->commands, d_fine_r, CL_TRUE, 0, sizeof(double) * m * n * o, fine[0][0], 0, NULL,
                              NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output arrays from device!\n%s\n", mgcl_err_code(err));
        exit(1);
    }

    clReleaseMemObject(d_fine_r);
    clReleaseMemObject(d_fine_v);
    clReleaseMemObject(d_coarse);
}

/* Prolongates from coarse to fine grid.
 * Doesn't create buffers or copy memory from or to device. */
void mgcl_prolongate(mgcl_config *conf, mgcl_level_data *fine, mgcl_level_data *coarse)
{
    int err;

    // Create the compute kernel from the program
    cl_kernel kernel = clCreateKernel(conf->program, "prolongate_to_fine", &err);
    mgclCheckError(err, "Creating kernel");

    // assign kernel arguments
    int pos = 0;
    err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &fine->d_r);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &coarse->d_v_in);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &fine->m);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &fine->n);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &fine->o);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &conf->ghosts);
    mgclCheckError(err, "Setting kernel arguments");

    // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
    size_t global[3] = {static_cast<size_t>(coarse->m), static_cast<size_t>(coarse->n), static_cast<size_t>(coarse->o)};
    const size_t local[3] = {static_cast<size_t>(coarse->m > 4 ? 4 : coarse->m),
                             static_cast<size_t>(coarse->n > 4 ? 4 : coarse->n),
                             static_cast<size_t>(coarse->o > 4 ? 4 : coarse->o)};

    for (int i = 0; i < 3; i++)
        if (global[i] % local[i] != 0)
        {
            // printf("padding global size %d from %ld to ", i, global[i]);
            global[i] += local[i] - (global[i] % local[i]);
            // printf("%ld (multiple of %ld)\n", global[i], local[i]);
        }

    err = mgcl_update_ghosts(conf, coarse->d_v_in, coarse->m, coarse->n, coarse->o, conf->ghosts, conf->ghosts,
                             conf->ghosts);
    mgclCheckError(err, "Updating ghosts coarse");
    err = clEnqueueNDRangeKernel(conf->commands, kernel, 3, NULL, global, local, 0, NULL, NULL);
    mgclCheckError(err, "Enqueueing kernel");
    // err = mgcl_update_ghosts(conf, fine->d_r, fine->m, fine->n, fine->o, conf->ghosts, conf->ghosts, conf->ghosts);
    // mgclCheckError(err, "Updating ghosts fine");

    clReleaseKernel(kernel);
}

/* Runs V-cycle recursively and sequentially without using ocl */
double mgcl_vcycle_seq(mgcl_config *conf, mgcl_level_data *data, int level)
{
    // printf("level = %d, m = %3.d\n", level, data[level].m-2);
    // conf->maxlevel = 7;
    double res;

    // reset initial guess of coarser grid
    if (level < conf->maxlevel - 1)
    {
        for (int i = conf->ghosts; i < data[level + 1].m - conf->ghosts; i++)
            for (int j = conf->ghosts; j < data[level + 1].n - conf->ghosts; j++)
                for (int k = conf->ghosts; k < data[level + 1].o - conf->ghosts; k++)
                    data[level + 1].v[i][j][k] = 0;
    }

    // relax nu1 times
    if (conf->stencil_values)
        res = mgcl_stencil_jacobi_seq(data[level].v, data[level].f, data[level].r, data[level].m - 2 * conf->ghosts,
                                      data[level].n - 2 * conf->ghosts, data[level].o - 2 * conf->ghosts, conf->ghosts,
                                      conf->omega, conf->nu1, conf->residual_norm, conf->stencil, conf->stencil_values,
                                      conf->stencil_size_multiplier);
    else
        res = mgcl_jacobi_seq(data[level].v, data[level].f, data[level].r, data[level].m - 2 * conf->ghosts,
                              data[level].n - 2 * conf->ghosts, data[level].o - 2 * conf->ghosts, conf->ghosts,
                              conf->omega, conf->nu1, conf->residual_norm, conf->stencil);

    // update residual without D^-1
    // res = residual(data[level].f, data[level].v, data[level].r, data[level].m-2, data[level].n-2, data[level].o-2,
    // conf->residual_norm, conf->stencil); printf("res on level %d, upwards: %e\n", level, sqrt(res));

    // restrict residual as right hand side on coarser grid
    mgcl_restrict_seq(data[level].r, data[level + 1].f, data[level + 1].m - 2 * conf->ghosts,
                      data[level + 1].n - 2 * conf->ghosts, data[level + 1].o - 2 * conf->ghosts, conf->ghosts);

    // restrict stencil values if stencil is not fixed
    if (conf->stencil_values && conf->restrict_prolongate_stencil)
        mgcl_stencil_restrict_seq(data[level].stencil_values, data[level + 1].stencil_values,
                                  data[level + 1].m - 2 * conf->ghosts, data[level + 1].n - 2 * conf->ghosts,
                                  data[level + 1].o - 2 * conf->ghosts, conf->ghosts, conf->stencil_size_multiplier);

    // if not at highest level...
    if (level < conf->maxlevel - 1)
    {
        // start next v-cycle iteration
        if (level < conf->maxlevel - 2)
            mgcl_vcycle_seq(conf, data, level + 1);
        else
        {
            // printf(" pre v[0] = %e, f[0] = %e\n", data[level+1].v[1][1][1], data[level+1].f[1][1][1]);
            if (conf->stencil_values)
                mgcl_stencil_jacobi_seq(data[level + 1].v, data[level + 1].f, data[level + 1].r,
                                        data[level + 1].m - 2 * conf->ghosts, data[level + 1].n - 2 * conf->ghosts,
                                        data[level + 1].o - 2 * conf->ghosts, conf->ghosts, conf->omega,
                                        conf->nu1 + conf->nu2, conf->residual_norm, conf->stencil, conf->stencil_values,
                                        conf->stencil_size_multiplier);
            else
                mgcl_jacobi_seq(data[level + 1].v, data[level + 1].f, data[level + 1].r,
                                data[level + 1].m - 2 * conf->ghosts, data[level + 1].n - 2 * conf->ghosts,
                                data[level + 1].o - 2 * conf->ghosts, conf->ghosts, conf->omega, conf->nu1 + conf->nu2,
                                conf->residual_norm, conf->stencil);

            // printf("post v[0] = %e, f[0] = %e\n", data[level+1].v[1][1][1], data[level+1].f[1][1][1]);
        }
    }

    // prolongate from coarser to finer grid
    // r of this level is reused here and should actually be called e
    mgcl_prolongate_seq(data[level].r, data[level + 1].v, data[level].m - 2 * conf->ghosts,
                        data[level].n - 2 * conf->ghosts, data[level].o - 2 * conf->ghosts, conf->ghosts);

    // prolongate stencil values if stencil is not fixed
    if (conf->stencil_values && conf->restrict_prolongate_stencil)
        mgcl_stencil_prolongate_seq(data[level].stencil_values, data[level + 1].stencil_values,
                                    data[level].m - 2 * conf->ghosts, data[level].n - 2 * conf->ghosts,
                                    data[level].o - 2 * conf->ghosts, conf->ghosts, conf->stencil_size_multiplier);

    // correct error
    for (int i = conf->ghosts; i < data[level].m - conf->ghosts; i++)
        for (int j = conf->ghosts; j < data[level].n - conf->ghosts; j++)
            for (int k = conf->ghosts; k < data[level].o - conf->ghosts; k++)
                data[level].v[i][j][k] += data[level].r[i][j][k];

    // relax nu2 times
    if (conf->stencil_values)
        res = mgcl_stencil_jacobi_seq(data[level].v, data[level].f, data[level].r, data[level].m - 2 * conf->ghosts,
                                      data[level].n - 2 * conf->ghosts, data[level].o - 2 * conf->ghosts, conf->ghosts,
                                      conf->omega, conf->nu2, conf->residual_norm, conf->stencil, conf->stencil_values,
                                      conf->stencil_size_multiplier);
    else
        res = mgcl_jacobi_seq(data[level].v, data[level].f, data[level].r, data[level].m - 2 * conf->ghosts,
                              data[level].n - 2 * conf->ghosts, data[level].o - 2 * conf->ghosts, conf->ghosts,
                              conf->omega, conf->nu2, conf->residual_norm, conf->stencil);
    // printf("res on level %d, downwards: %e\n", level, sqrt(res));
    return res;
}

/* Runs V-cycle recursively using ocl */
double mgcl_vcycle(mgcl_config *conf, mgcl_level_data *data, int level)
{
    // printf("level = %d, m = %3.d\n", level, data[level].m-2);
    // conf->maxlevel = 3;
    double res;
    int err;
    cl_uint zero = 0;

    if (level < conf->maxlevel - 1) // if not at highest level
    {
        // reset v to zero for coarser grids (for another possible v-cycle)
        err = clEnqueueFillBuffer(conf->commands, data[level + 1].d_v_in, &zero, sizeof(cl_uint), 0,
                                  sizeof(double) * data[level + 1].m * data[level + 1].n * data[level + 1].o, 0, NULL,
                                  NULL);
        mgclCheckError(err, "resetting d_v_in to 0");
    }

    // relax nu1 times
    if (conf->stencil_values)
        res = mgcl_stencil_jacobi(conf, &data[level], conf->nu1, 1);
    else
        res = mgcl_jacobi(conf, &data[level], conf->nu1, 1);

    // update residual without D^-1
    // res = mgcl_residual(conf, &data[level], 1);
    // printf("res on level %d, upwards: %e\n", level, res);

    // restrict to coarser grid
    mgcl_restrict(conf, &data[level], &data[level + 1]);

    // restrict stencil values if stencil is not fixed
    if (conf->stencil_values && conf->restrict_prolongate_stencil)
        mgcl_stencil_restrict(conf, &data[level], &data[level + 1]);

    if (level < conf->maxlevel - 1)
    {
        // start next v-cycle iteration
        if (level < conf->maxlevel - 2)
            mgcl_vcycle(conf, data, level + 1);
        else
        {
            if (conf->stencil_values)
                res = mgcl_stencil_jacobi(conf, &data[level + 1], conf->nu1 + conf->nu2, 0);
            else
                res = mgcl_jacobi(conf, &data[level + 1], conf->nu1 + conf->nu2, 0);
        }
    }

    // prolongate from coarser to finer grid
    // r of this level is reused here and should actually be called e
    mgcl_prolongate(conf, &data[level], &data[level + 1]);

    // prolongate stencil values if stencil is not fixed
    if (conf->stencil_values && conf->restrict_prolongate_stencil)
        mgcl_stencil_prolongate(conf, &data[level], &data[level + 1]);

    // correct error
    mgcl_correct_error(conf, data[level].d_v_in, data[level].d_r, data[level].m, data[level].n, data[level].o);

    // relax nu2 times
    if (conf->stencil_values)
        res = mgcl_stencil_jacobi(conf, &data[level], conf->nu2, 1);
    else
        res = mgcl_jacobi(conf, &data[level], conf->nu2, 1);

    // calculate residual again for the norm TODO in jacobi
    // res = mgcl_residual(conf, &data[level], 1);
    // printf("res on level %d, downwards: %e\n", level, res);
    return res;
}

/* Cleans everything up */
void mgcl_finish(mgcl_config *conf, mgcl_level_data *data)
{
    if (data[0].v != NULL)
    {
        cuboid_free(data[0].v, data[0].m, data[0].n, data[0].o);
        data[0].v = NULL;
    }

    if (!conf->reuse_opencl_buffers && !conf->copy_buffer_data)
    {
        cuboid_free(data[0].f, data[0].m, data[0].n, data[0].o);

        if (conf->stencil_values != NULL)
            cuboid_free(data[0].stencil_values, data[0].m, data[0].n, data[0].o * conf->stencil_size_multiplier);
    }
    else if (!conf->use_opencl)
    {
        for (int level = 0; level < conf->maxlevel; level++)
        {
            cuboid_free(data[level].v, data[level].m, data[level].n, data[level].o);
            cuboid_free(data[level].f, data[level].m, data[level].n, data[level].o);
            cuboid_free(data[level].r, data[level].m, data[level].n, data[level].o);

            if (conf->stencil_values != NULL && conf->restrict_prolongate_stencil)
                cuboid_free(data[level].stencil_values, data[level].m, data[level].n,
                            data[level].o * conf->stencil_size_multiplier);
        }
    }

    delete[] data;
    // free(conf);
}

/* Starts kernel to correct error, e.g. v = v + e
 * m,n,o is size of ghosted grid */
int mgcl_correct_error(mgcl_config *conf, cl_mem d_v, cl_mem d_r, int m, int n, int o)
{
    int err;

    // Create the compute kernel from the program
    cl_kernel kernel = clCreateKernel(conf->program, "correct_error", &err);
    mgclCheckError(err, "Creating kernel");

    // assign kernel arguments
    int pos = 0;
    err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &d_v);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &d_r);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &conf->ghosts);
    mgclCheckError(err, "Setting kernel arguments");

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
    mgclCheckError(err, "Enqueueing kernel");

    clReleaseKernel(kernel);
    return err;
}

/* Waits for all running OpenCL kernels to finish and reads back results from device. Creates arrays on host if none
 * were specified */
int mgcl_read_results(mgcl_config *conf, mgcl_level_data *data)
{
    int err = clFinish(conf->commands);
    mgclCheckError(err, "Waiting for kernels to finish");

    if (conf->reuse_opencl_buffers || conf->copy_buffer_data)
    {
        data[0].v = cuboid_alloc(data[0].m, data[0].n, data[0].o); // gets freed automatically in finish
        if (conf->v == NULL)
            conf->v = cuboid_alloc(conf->m + 2 * conf->ghosts_in, conf->n + 2 * conf->ghosts_in,
                                   conf->o + 2 * conf->ghosts_in);
    }

    // read back results TODO: only for testing purposes, maybe define TESTING?
    err = clEnqueueReadBuffer(conf->commands, data[0].d_v_in, CL_TRUE, 0,
                              sizeof(double) * data[0].m * data[0].n * data[0].o, data[0].v[0][0], 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output arrays from device!\n%s\n", mgcl_err_code(err));
        exit(1);
    }

    // copy result to initial v vector
    for (int i = 0; i < conf->m; i++)
        for (int j = 0; j < conf->n; j++)
            for (int k = 0; k < conf->o; k++)
            {
                conf->v[i + conf->ghosts_in][j + conf->ghosts_in][k + conf->ghosts_in] =
                    data[0].v[i + conf->ghosts][j + conf->ghosts][k + conf->ghosts];
            }

    return err;
}

/* Reads values for one level from device for testing purposes */
void mgcl_test_read(mgcl_config *conf, mgcl_level_data *data, int level)
{
    int err = clFinish(conf->commands);
    mgclCheckError(err, "Waiting for kernels to finish");

    // read back results TODO: only for testing purposes, maybe define TESTING?
    err = clEnqueueReadBuffer(conf->commands, data[level].d_v_in, CL_TRUE, 0,
                              sizeof(double) * data[level].m * data[level].n * data[level].o, data[level].v[0][0], 0,
                              NULL, NULL);
    err = clEnqueueReadBuffer(conf->commands, data[level].d_f, CL_TRUE, 0,
                              sizeof(double) * data[level].m * data[level].n * data[level].o, data[level].f[0][0], 0,
                              NULL, NULL);
    err = clEnqueueReadBuffer(conf->commands, data[level].d_r, CL_TRUE, 0,
                              sizeof(double) * data[level].m * data[level].n * data[level].o, data[level].r[0][0], 0,
                              NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output arrays from device!\n%s\n", mgcl_err_code(err));
        exit(1);
    }

    printf("0 level = %d, v[1,1,1] = %e\n", level, data[level].v[1][1][1]);
    printf("0 level = %d, f[1,1,1] = %e\n", level, data[level].f[1][1][1]);
    printf("0 level = %d, r[1,1,1] = %e\n", level, data[level].r[1][1][1]);
}

/* Runs multigrid solver for u and f using OpenCL.
 * u and f must be of size m x n x o.
 * nu1 and nu2 are pre and post smoothing steps.
 * m,n,o is size of real grid (without ghost cells) */
void mgcl(mgcl_config *conf)
{
    // run mgcl_seq if use_opencl is not set
    if (!conf->use_opencl)
    {
        printf("Not using OpenCL (not specified in conf). Running mgcl sequentially.\n");
        mgcl_seq(conf);
        return;
    }
    printf("Starting mgcl using OpenCL\n");

    // set up data for each level TODO reuse device buffers in final code
    mgcl_level_data *data;
    if (mgcl_init(conf, &data) == EXIT_FAILURE)
        return;
    mgcl_init_opencl(conf, data);

    // calculate initial residual
    double initres;
    if (conf->stencil_values)
        initres = mgcl_stencil_residual(conf, &data[0], 1);
    else
        initres = mgcl_residual(conf, &data[0], 1);
    printf("Starting mgcl with initres = %e\n", initres);

    // run vcycle maxiter_vcycles times
    double res, relres;
    for (int i = 0; i < conf->maxiter_vcycles; i++)
    {
        auto tstart = std::chrono::steady_clock::now();
        res = mgcl_vcycle(conf, data, 0);
        auto tend = mgcl_since(tstart).count() * 1000.0;
        relres = initres == 0 ? 0 : res / initres;
        printf("iter = %d, elapsed time = %2.5lf s, rel. res = %e\n", i, tend, relres);

        if (relres < conf->tol)
            break;
    }

    // copy resulting v to conf->d_v on device
    if (conf->copy_buffer_data)
        mgcl_copy_output_buffers(conf, &data[0]);

    // write result into conf->v on host
    if (conf->read_results)
        mgcl_read_results(conf, data);

    // clean up memory
    mgcl_release_opencl(conf, data);
    mgcl_finish(conf, data);
}

/* Runs multigrid solver without using OpenCL */
void mgcl_seq(mgcl_config *conf)
{
    // set up data for each level
    mgcl_level_data *data;
    if (mgcl_init(conf, &data) == EXIT_FAILURE)
        return;

    // calculate initial residual (different from pmg's initres bc ghosts are not updated in pmg first)
    update_ghosts_seq(data[0].v, conf->m, conf->n, conf->o, conf->ghosts, conf->ghosts, conf->ghosts);
    // update_ghosts_seq(data[0].f, conf->m, conf->n, conf->o);
    double initres;
    if (conf->stencil_values)
        initres =
            stencil_residual(data[0].f, data[0].v, data[0].r, conf->m, conf->n, conf->o, conf->ghosts,
                             conf->residual_norm, conf->stencil, conf->stencil_values, conf->stencil_size_multiplier);
    else
        initres = residual(data[0].f, data[0].v, data[0].r, conf->m, conf->n, conf->o, conf->ghosts,
                           conf->residual_norm, conf->stencil);
    printf("Starting mgcl with initres = %e\n", initres);

    // run vcycle maxiter_vcycles times
    double res, relres;
    for (int i = 0; i < conf->maxiter_vcycles; i++)
    {
        auto tstart = std::chrono::steady_clock::now();
        res = mgcl_vcycle_seq(conf, data, 0);
        auto tend = mgcl_since(tstart).count() * 1000.0;
        relres = initres == 0 ? 0 : res / initres;
        printf("iter = %d, elapsed time = %2.5lf s, rel. res = %e\n", i, tend, relres);

        if (relres < conf->tol)
            break;
    }

    // write data to output
    for (int i = 0; i < conf->m; i++)
        for (int j = 0; j < conf->n; j++)
            for (int k = 0; k < conf->o; k++)
            {
                conf->v[i + conf->ghosts_in][j + conf->ghosts_in][k + conf->ghosts_in] =
                    data[0].v[i + conf->ghosts][j + conf->ghosts][k + conf->ghosts];
            }

    // clean up memory
    mgcl_finish(conf, data);
}
