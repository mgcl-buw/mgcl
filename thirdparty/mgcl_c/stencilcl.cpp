#include "stencilcl.hpp"
#include "clutil.hpp"
#include "cuboid.hpp"
#include "ghostscl.hpp"
#include "jacobicl.hpp"

#include <mpi.h>
#include <stdio.h>
#include <tgmath.h>

// TODO proper implementations, just copying values right now

/* Restricts stencil to coarser grid using full-weighted restriction operator.
 * m, n and o must be the dimensions of the coarser grid without ghost cells. */
void mgcl_stencil_restrict_seq(double ***fine, double ***coarse, int m, int n, int o, int ghosts,
                               int stencil_size_multiplier)
{
    // mgcl_debug("untested yet\n");

    update_ghosts_seq(fine, m * 2, n * 2, o * 2 * stencil_size_multiplier, ghosts, ghosts,
                      ghosts * stencil_size_multiplier);
    int ioff = 1, joff = 1,
        koff = stencil_size_multiplier; // offset grows by 1 for each step, or by stencil_size for innermost loop
    int i2, j2, k2;
    for (int i = ghosts; i < m + ghosts; i++, ioff++)
    {
        i2 = i + ioff; // == i*2+ghosts+1
        joff = 1;
        for (int j = ghosts; j < n + ghosts; j++, joff++)
        {
            j2 = j + joff; // TODO adjust indices
            koff = stencil_size_multiplier;
            for (int k = ghosts * stencil_size_multiplier; k < (o + ghosts) * stencil_size_multiplier;
                 k += stencil_size_multiplier, koff += stencil_size_multiplier)
            {
                // restrict each stencil value for current grid point
                for (int st = 0; st < stencil_size_multiplier; st++)
                {
                    k2 = k + koff + st;
                    coarse[i][j][k + st] = 0.125 * fine[i2][j2][k2] // self
                                                                    // direct neighbours
                                           + 0.0625 * fine[i2 - 1][j2][k2] + 0.0625 * fine[i2 + 1][j2][k2] +
                                           0.0625 * fine[i2][j2 - 1][k2] + 0.0625 * fine[i2][j2 + 1][k2] +
                                           0.0625 * fine[i2][j2][k2 - stencil_size_multiplier] +
                                           0.0625 * fine[i2][j2][k2 + stencil_size_multiplier]
                                           // edge midpoints xy-plane
                                           + 0.03125 * fine[i2 - 1][j2 - 1][k2] + 0.03125 * fine[i2 - 1][j2 + 1][k2] +
                                           0.03125 * fine[i2 + 1][j2 - 1][k2] +
                                           0.03125 * fine[i2 + 1][j2 + 1][k2]
                                           // edge midpoints xz-plane
                                           + 0.03125 * fine[i2 - 1][j2][k2 - stencil_size_multiplier] +
                                           0.03125 * fine[i2 - 1][j2][k2 + stencil_size_multiplier] +
                                           0.03125 * fine[i2 + 1][j2][k2 - stencil_size_multiplier] +
                                           0.03125 * fine[i2 + 1][j2][k2 + stencil_size_multiplier]
                                           // edge midpoints yz-plane
                                           + 0.03125 * fine[i2][j2 - 1][k2 - stencil_size_multiplier] +
                                           0.03125 * fine[i2][j2 - 1][k2 + stencil_size_multiplier] +
                                           0.03125 * fine[i2][j2 + 1][k2 - stencil_size_multiplier] +
                                           0.03125 * fine[i2][j2 + 1][k2 + stencil_size_multiplier]
                                           // corners
                                           + 0.015625 * fine[i2 - 1][j2 - 1][k2 - stencil_size_multiplier] +
                                           0.015625 * fine[i2 - 1][j2 - 1][k2 + stencil_size_multiplier] +
                                           0.015625 * fine[i2 - 1][j2 + 1][k2 - stencil_size_multiplier] +
                                           0.015625 * fine[i2 - 1][j2 + 1][k2 + stencil_size_multiplier] +
                                           0.015625 * fine[i2 + 1][j2 - 1][k2 - stencil_size_multiplier] +
                                           0.015625 * fine[i2 + 1][j2 - 1][k2 + stencil_size_multiplier] +
                                           0.015625 * fine[i2 + 1][j2 + 1][k2 - stencil_size_multiplier] +
                                           0.015625 * fine[i2 + 1][j2 + 1][k2 + stencil_size_multiplier];
                }
            }
        }
    }
    update_ghosts_seq(coarse, m, n, o * stencil_size_multiplier, ghosts, ghosts, ghosts * stencil_size_multiplier);
}

void mgcl_stencil_restrict_test(mgcl_config *conf, double ***fine, double ***coarse, int m, int n, int o, int ghosts,
                                int stencil_size_multiplier)
{
    mgcl_debug("NOT YET INPLEMENTED\n");
}

void mgcl_stencil_restrict(mgcl_config *conf, mgcl_level_data *fine, mgcl_level_data *coarse)
{
    // mgcl_debug("untested yet\n");
    // TODO correct global and arguments for stencil

    int err;
    int mreal = coarse->m - 2 * conf->ghosts;
    int nreal = coarse->n - 2 * conf->ghosts;
    int oreal = coarse->o - 2 * conf->ghosts;
    int ost = coarse->o * conf->stencil_size_multiplier;

    // Create the compute kernel from the program
    cl_kernel kernel = clCreateKernel(conf->program, "stencil_restrict_to_coarse", &err);
    mgclCheckError(err, "Creating kernel");

    // assign kernel arguments
    int pos = 0;
    err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &fine->d_stencil_values);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &coarse->d_stencil_values);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &coarse->m);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &coarse->n);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ost);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &conf->ghosts);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &conf->stencil_size_multiplier);
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

    err = mgcl_update_ghosts(conf, fine->d_stencil_values, fine->m, fine->n, fine->o * conf->stencil_size_multiplier,
                             conf->ghosts, conf->ghosts, conf->ghosts * conf->stencil_size_multiplier);
    mgclCheckError(err, "Updating fine ghosts");
    err = clEnqueueNDRangeKernel(conf->commands, kernel, 3, NULL, global, local, 0, NULL, NULL);
    mgclCheckError(err, "Enqueueing restriction kernel");
    err = mgcl_update_ghosts(conf, coarse->d_stencil_values, coarse->m, coarse->n,
                             coarse->o * conf->stencil_size_multiplier, conf->ghosts, conf->ghosts,
                             conf->ghosts * conf->stencil_size_multiplier);
    mgclCheckError(err, "Updating coarse ghosts");

    clReleaseKernel(kernel);
}

/* Prolongates stencil from coarse to fine grid.
 * m, n and o must be dimensions of the fine grid without ghost cells. */
void mgcl_stencil_prolongate_seq(double ***fine, double ***coarse, int m, int n, int o, int ghosts,
                                 int stencil_size_multiplier)
{
    // mgcl_debug("untested yet\n");

    update_ghosts_seq(coarse, m / 2, n / 2, (o / 2) * stencil_size_multiplier, ghosts, ghosts,
                      ghosts * stencil_size_multiplier);
    int ioff = 1, joff = 1, koff = 1; // offset grows by 1 for each step
    int i2, j2, k2;
    for (int i = ghosts; i < m / 2 + ghosts; i++, ioff++)
    {
        i2 = i + ioff; // == i*2+ghosts+1
        joff = 1;
        for (int j = ghosts; j < n / 2 + ghosts; j++, joff++)
        {
            j2 = j + joff;
            koff = stencil_size_multiplier; // TODO check indices and boundaries
            for (int k = ghosts * stencil_size_multiplier; k < (o / 2 + ghosts) * stencil_size_multiplier;
                 k += stencil_size_multiplier, koff += stencil_size_multiplier)
            {
                // prolongate each stencil value for current grid point
                for (int st = 0; st < stencil_size_multiplier; st++)
                {
                    k2 = k + koff + st;
                    fine[i2][j2][k2] = coarse[i][j][k + st];

                    fine[i2][j2][k2 - stencil_size_multiplier] =
                        0.5 * (coarse[i][j][k + st] + coarse[i][j][k + st - stencil_size_multiplier]);
                    fine[i2][j2 - 1][k2] = 0.5 * (coarse[i][j][k + st] + coarse[i][j - 1][k + st]);
                    fine[i2 - 1][j2][k2] = 0.5 * (coarse[i][j][k + st] + coarse[i - 1][j][k + st]);

                    fine[i2][j2 - 1][k2 - stencil_size_multiplier] =
                        0.25 * (coarse[i][j][k + st] + coarse[i][j][k + st - stencil_size_multiplier] +
                                coarse[i][j - 1][k + st] + coarse[i][j - 1][k + st - stencil_size_multiplier]);
                    fine[i2 - 1][j2][k2 - stencil_size_multiplier] =
                        0.25 * (coarse[i][j][k + st] + coarse[i][j][k + st - stencil_size_multiplier] +
                                coarse[i - 1][j][k + st] + coarse[i - 1][j][k + st - stencil_size_multiplier]);
                    fine[i2 - 1][j2 - 1][k2] = 0.25 * (coarse[i][j][k + st] + coarse[i][j - 1][k + st] +
                                                       coarse[i - 1][j][k + st] + coarse[i - 1][j - 1][k + st]);

                    fine[i2 - 1][j2 - 1][k2 - stencil_size_multiplier] =
                        0.125 * (coarse[i][j][k + st] + coarse[i][j][k + st - stencil_size_multiplier] +
                                 coarse[i][j - 1][k + st] + coarse[i][j - 1][k + st - stencil_size_multiplier] +
                                 coarse[i - 1][j][k + st] + coarse[i - 1][j][k + st - stencil_size_multiplier] +
                                 coarse[i - 1][j - 1][k + st] + coarse[i - 1][j - 1][k + st - stencil_size_multiplier]);
                }
            }
        }
    }
    update_ghosts_seq(fine, m, n, o * stencil_size_multiplier, ghosts, ghosts, ghosts * stencil_size_multiplier);
}

void mgcl_stencil_prolongate_test(mgcl_config *conf, double ***fine, double ***coarse, int m, int n, int o, int ghosts,
                                  int stencil_size_multiplier)
{
    mgcl_debug("NOT YET INPLEMENTED\n");
}

void mgcl_stencil_prolongate(mgcl_config *conf, mgcl_level_data *fine, mgcl_level_data *coarse)
{
    // mgcl_debug("untested yet\n");

    int err;

    int ost = fine->o * conf->stencil_size_multiplier;

    // Create the compute kernel from the program
    cl_kernel kernel = clCreateKernel(conf->program, "stencil_prolongate_to_fine", &err);
    mgclCheckError(err, "Creating kernel");

    // assign kernel arguments
    int pos = 0;
    err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &fine->d_stencil_values);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &coarse->d_stencil_values);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &fine->m);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &fine->n);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ost);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &conf->ghosts);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &conf->stencil_size_multiplier);
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

    err = mgcl_update_ghosts(conf, coarse->d_stencil_values, coarse->m, coarse->n,
                             coarse->o * conf->stencil_size_multiplier, conf->ghosts, conf->ghosts, conf->ghosts);
    mgclCheckError(err, "Updating ghosts coarse");
    err = clEnqueueNDRangeKernel(conf->commands, kernel, 3, NULL, global, local, 0, NULL, NULL);
    mgclCheckError(err, "Enqueueing kernel");
    err = mgcl_update_ghosts(conf, fine->d_stencil_values, fine->m, fine->n, fine->o * conf->stencil_size_multiplier,
                             conf->ghosts, conf->ghosts, conf->ghosts * conf->stencil_size_multiplier);
    mgclCheckError(err, "Updating ghosts fine");

    clReleaseKernel(kernel);
}

/* Runs jacobi method sequentially.
 * v, f and r must be of size [m+2][n+2][o+2] for periodic boundary condition.
 * m,n,o is size of real grid */
double mgcl_stencil_jacobi_seq(double ***v, double ***f, double ***r, int m, int n, int o, int ghosts, double omega,
                               int maxiter, MGCL_RESIDUAL_NORM resnorm, MGCL_STENCIL stencil, double ***stencil_values,
                               int stencil_size_multiplier)
{
    double res = 0.0;

    if (!stencil_values)
    {
        printf("mgcl_stencil_jacobi_seq: stencil_values is NULL!\n");
        return -1;
    }

    for (int iter = 0; iter < maxiter; iter++)
    {
        // update ghost cells for periodic boundary condition
        update_ghosts_seq(v, m, n, o, ghosts, ghosts, ghosts);

        // damped/weighted iteration formula: u_(m+1) = u_(m) + omega * D^-1 * r_(m)

        // r = f - A*v
        res = stencil_residual(f, v, r, m, n, o, ghosts, resnorm, stencil, stencil_values, stencil_size_multiplier);
        for (int i = ghosts; i < m + ghosts; i++)
            for (int j = ghosts; j < n + ghosts; j++)
                for (int k = ghosts, kst = ghosts * stencil_size_multiplier; k < o + ghosts;
                     k++, kst += stencil_size_multiplier)
                {
                    // if (i == ghosts && j == ghosts && k == ghosts)
                    //     printf("seq dinv = %f\n", (1.0 / stencil_values[i][j][kst]));
                    // if (i == 1 && j == 1 && k == 1)
                    //     printf("v[%d][%d][%d] = %f, r[%d][%d][%d] = %f, omega = %f\n", i,j,k, v[i][j][k],
                    //     i,j,k,r[i][j][k], omega);
                    v[i][j][k] = v[i][j][k] + omega * (1.0 / stencil_values[i][j][kst]) * r[i][j][k];
                }
    }
    update_ghosts_seq(v, m, n, o, ghosts, ghosts, ghosts);
    return res;
}

/* Calculates r = f - A*v using 7-point stencil of 3D laplacian.
 * m,n,o is size of real grid */
double stencil_residual(double ***f, double ***v, double ***r, int m, int n, int o, int ghosts,
                        MGCL_RESIDUAL_NORM resnorm, MGCL_STENCIL stencil, double ***stencil_values,
                        int stencil_size_multiplier)
{
    double res = 0.0;
    double stencilsum = 0;
    int kst;

    for (int i = ghosts; i < m + ghosts; i++)
        for (int j = ghosts; j < n + ghosts; j++)
            for (int k = ghosts, kst = ghosts * stencil_size_multiplier; k < o + ghosts;
                 k++, kst += stencil_size_multiplier)
            {
                // A*v
                if (stencil == MGCL_7POINT_VARSYM)
                    stencilsum =
                        stencil_values[i][j][kst] * v[i][j][k] + stencil_values[i][j][kst + 1] * v[i][j][k - 1] +
                        stencil_values[i][j][kst + 1] * v[i][j][k + 1] +
                        stencil_values[i][j][kst + 2] * v[i][j - 1][k] +
                        stencil_values[i][j][kst + 2] * v[i][j + 1][k] +
                        stencil_values[i][j][kst + 3] * v[i - 1][j][k] + stencil_values[i][j][kst + 3] * v[i + 1][j][k];
                else if (stencil == MGCL_19POINT_VARSYM)
                    stencilsum =
                        stencil_values[i][j][kst] * v[i][j][k] + stencil_values[i][j][kst + 1] * v[i][j][k - 1] +
                        stencil_values[i][j][kst + 1] * v[i][j][k + 1] +
                        stencil_values[i][j][kst + 2] * v[i][j - 1][k] +
                        stencil_values[i][j][kst + 2] * v[i][j + 1][k] +
                        stencil_values[i][j][kst + 3] * v[i - 1][j][k] + stencil_values[i][j][kst + 3] * v[i + 1][j][k]

                        + stencil_values[i][j][kst + 4] * v[i][j - 1][k - 1] +
                        stencil_values[i][j][kst + 4] * v[i][j - 1][k + 1] +
                        stencil_values[i][j][kst + 4] * v[i][j + 1][k - 1] +
                        stencil_values[i][j][kst + 4] * v[i][j + 1][k + 1] +
                        stencil_values[i][j][kst + 6] * v[i - 1][j][k - 1] +
                        stencil_values[i][j][kst + 6] * v[i - 1][j][k + 1] +
                        stencil_values[i][j][kst + 6] * v[i + 1][j][k - 1] +
                        stencil_values[i][j][kst + 6] * v[i + 1][j][k + 1] +
                        stencil_values[i][j][kst + 5] * v[i - 1][j - 1][k] +
                        stencil_values[i][j][kst + 5] * v[i - 1][j + 1][k] +
                        stencil_values[i][j][kst + 5] * v[i + 1][j - 1][k] +
                        stencil_values[i][j][kst + 5] * v[i + 1][j + 1][k];
                else if (stencil == MGCL_27POINT_VARSYM)
                    stencilsum =
                        stencil_values[i][j][kst] * v[i][j][k] + stencil_values[i][j][kst + 1] * v[i][j][k - 1] +
                        stencil_values[i][j][kst + 1] * v[i][j][k + 1] +
                        stencil_values[i][j][kst + 2] * v[i][j - 1][k] +
                        stencil_values[i][j][kst + 2] * v[i][j + 1][k] +
                        stencil_values[i][j][kst + 3] * v[i - 1][j][k] + stencil_values[i][j][kst + 3] * v[i + 1][j][k]

                        + stencil_values[i][j][kst + 4] * v[i][j - 1][k - 1] +
                        stencil_values[i][j][kst + 4] * v[i][j - 1][k + 1] +
                        stencil_values[i][j][kst + 4] * v[i][j + 1][k - 1] +
                        stencil_values[i][j][kst + 4] * v[i][j + 1][k + 1] +
                        stencil_values[i][j][kst + 6] * v[i - 1][j][k - 1] +
                        stencil_values[i][j][kst + 6] * v[i - 1][j][k + 1] +
                        stencil_values[i][j][kst + 6] * v[i + 1][j][k - 1] +
                        stencil_values[i][j][kst + 6] * v[i + 1][j][k + 1] +
                        stencil_values[i][j][kst + 5] * v[i - 1][j - 1][k] +
                        stencil_values[i][j][kst + 5] * v[i - 1][j + 1][k] +
                        stencil_values[i][j][kst + 5] * v[i + 1][j - 1][k] +
                        stencil_values[i][j][kst + 5] * v[i + 1][j + 1][k]

                        + stencil_values[i][j][kst + 7] * v[i - 1][j - 1][k - 1] +
                        stencil_values[i][j][kst + 7] * v[i - 1][j - 1][k + 1] +
                        stencil_values[i][j][kst + 7] * v[i - 1][j + 1][k - 1] +
                        stencil_values[i][j][kst + 7] * v[i - 1][j + 1][k + 1] +
                        stencil_values[i][j][kst + 7] * v[i + 1][j - 1][k - 1] +
                        stencil_values[i][j][kst + 7] * v[i + 1][j - 1][k + 1] +
                        stencil_values[i][j][kst + 7] * v[i + 1][j + 1][k - 1] +
                        stencil_values[i][j][kst + 7] * v[i + 1][j + 1][k + 1];

                // if (i == 1 && j == 1 && k == 2)
                //     printf("stencilsum = %e\n", stencilsum);
                // if (i >= 0 && i <= 6 && j >= 0 && j <= 6 && k >= 0 && k <= 6 && m > 4 && stencil ==
                // MGCL_7POINT_VARSYM)
                //     print_7point(v, i, j, k);
                // if (i >= 4 && i <= 4 && j >= 4 && j <= 4 && k >= 4 && k <= 4 && m > 4 && stencil ==
                // MGCL_7POINT_VARSYM)
                //     printf("stencil_values: %f, %f, %f, %f\n", stencil_values[i][j][kst],
                //     stencil_values[i][j][kst+3], stencil_values[i][j][kst+3], stencil_values[i][j][kst+3]);

                // if (i == ghosts && j == ghosts && k == ghosts)
                // {
                //     print_7point(v, i, j, k);
                //     printf("stencil = %d\n", stencil);
                //     printf("stencil_values[i][j][kst] = %e, stencilsum = %e\n", stencil_values[i][j][kst],
                //     stencilsum); printf("stencil_values: %f, %f, %f, %f\n", stencil_values[i][j][kst],
                //     stencil_values[i][j][kst+1], stencil_values[i][j][kst+2], stencil_values[i][j][kst+3]);
                // }

                // r = f - A*v
                r[i][j][k] = f[i][j][k] - stencilsum;
                if (resnorm == MGCL_L2)
                    res += r[i][j][k] * r[i][j][k];
                else if (fabs(r[i][j][k]) > res)
                    res = r[i][j][k];
            }
    update_ghosts_seq(r, m, n, o, ghosts, ghosts, ghosts);
    return resnorm == MGCL_L2 ? sqrt(res) : res;
}

/* Runs jacobi method using OpenCL.
 * Doesn't creates ocl buffers and doesn't copy data from host to device and vice versa
 * v, f and r must be of size [m][n][o] for periodic boundary condition.
 * m, n and o must be the dimensions of grid + 2*ghosts
 * ghosts of stencil_values must be updated manually beforehand.
 * If return_residual is true, the residual's 2-norm or inf-norm will be read back from device and returned, else -1.
 * It's not
 * really performant to do so because we have to wait for all kernels to complete and reading a buffer to host is slow.
 */
double mgcl_stencil_jacobi(mgcl_config *conf, mgcl_level_data *data, int maxiter, int return_residual)
{
    int err;
    int m = data->m;
    int n = data->n;
    int o = data->o;
    int store_res = 0;
    double res = -1;

    if (conf->use_local_memory)
    {
        res = mgcl_stencil_jacobi_local_mem(conf, data, maxiter, return_residual);
        if (res != -2)
            return res;

        mgcl_debug("mgcl_jacobi_local_mem apparently failed. Running global mem version instead.\n");
    }

    double h2 = (1.0 / (double)(m - 2 * conf->ghosts)) *
                (1.0 / (double)(m - 2 * conf->ghosts)); // TODO minimum of m,n,o when not cube?
    double h2inv = 1.0 / h2;                            // divisor of the stencil, inverted to use * instead of / in kernel

    // Create the compute kernel from the program
    const char *kernel_name;
    if (conf->stencil == MGCL_7POINT_VARSYM)
        kernel_name = "jacobi_stencil_iter_7point";
    else if (conf->stencil == MGCL_19POINT_VARSYM)
    {
        kernel_name = "jacobi_stencil_iter_19point";
        h2inv = 1.0 / (6.0 * h2);
    }
    else if (conf->stencil == MGCL_27POINT_VARSYM)
    {
        kernel_name = "jacobi_stencil_iter_27point";
        h2inv = 1.0 / (30.0 * h2);
    }
    cl_kernel kernel = clCreateKernel(conf->program, kernel_name, &err);
    mgclCheckError(err, "Creating kernel");

    // assign kernel arguments
    int pos = 0;
    err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &data->d_v_in);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &data->d_v_out);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &data->d_f);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &data->d_r);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &data->d_stencil_values);
    err |= clSetKernelArg(kernel, ++pos, sizeof(double), &h2inv);
    err |= clSetKernelArg(kernel, ++pos, sizeof(double), &conf->omega);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &conf->ghosts);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &store_res);
    mgclCheckError(err, "Setting kernel arguments");

    // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
    size_t global[2] = {static_cast<size_t>(n), static_cast<size_t>(o)};
    const size_t local[2] = {static_cast<size_t>(n > 4 ? 4 : n),
                             static_cast<size_t>(o > 8 ? 8 : o)}; // TODO conf->jacobi_wg_size_x

    for (int i = 0; i < 2; i++)
        if (global[i] % local[i] != 0)
        {
            // printf("padding global size %d from %ld to ", i, global[i]);
            global[i] += local[i] - (global[i] % local[i]);
            // printf("%ld (multiple of %ld)\n", global[i], local[i]);
        }

    for (int iter = 0; iter < maxiter; iter++)
    {
        // switch arguments d_v_in -> d_v_out to use latest values in next iteration
        if (iter % 2 == 1)
        {
            err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &data->d_v_in);
            err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &data->d_v_out);
            mgclCheckError(err, "Setting kernel arguments");

            err = mgcl_update_ghosts(conf, data->d_v_out, m, n, o, conf->ghosts, conf->ghosts, conf->ghosts);
            mgclCheckError(err, "Updating ghosts");
        }
        else
        {
            err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &data->d_v_in);
            err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &data->d_v_out);
            mgclCheckError(err, "Setting kernel arguments");

            err = mgcl_update_ghosts(conf, data->d_v_in, m, n, o, conf->ghosts, conf->ghosts, conf->ghosts);
            mgclCheckError(err, "Updating ghosts");
        }

        // set flag to store residual in last iteration
        if (iter == maxiter - 1)
        {
            store_res = 1;
            err = clSetKernelArg(kernel, pos, sizeof(int), &store_res);
            mgclCheckError(err, "Setting kernel arguments");
        }

        err = clEnqueueNDRangeKernel(conf->commands, kernel, 2, NULL, global, local, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing kernel");
    }

    // copy result into d_v_in if needed
    if (maxiter % 2 == 1)
    {
        err = clEnqueueCopyBuffer(conf->commands, data->d_v_out, data->d_v_in, 0, 0, sizeof(double) * m * n * o, 0,
                                  NULL, NULL);
        mgclCheckError(err, "Update v");
    }

    err = mgcl_update_ghosts(conf, data->d_v_in, m, n, o, conf->ghosts, conf->ghosts, conf->ghosts);
    mgclCheckError(err, "Updating ghosts of v_in");

    // calculate residual's 2-norm. Square elements on device and sum up on host
    if (return_residual)
    {
        if (conf->residual_norm == MGCL_L2)
        {
            // calculate 2-Norm
            double ***rsquares = cuboid_alloc(m, n, o);
            int pointer_flag = conf->device_type == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
            cl_mem d_rsquares = clCreateBuffer(conf->context, CL_MEM_WRITE_ONLY | pointer_flag,
                                               sizeof(double) * m * n * o, rsquares[0][0], &err);
            mgclCheckError(err, "Creating rsquares buffer");

            // Create the compute kernel from the program
            cl_kernel kernel_square = clCreateKernel(conf->program, "residual_squared", &err);
            mgclCheckError(err, "Creating residual squared kernel");

            pos = 0;
            err = clSetKernelArg(kernel_square, pos, sizeof(cl_mem), &data->d_r);
            err |= clSetKernelArg(kernel_square, ++pos, sizeof(cl_mem), &d_rsquares);
            err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &m);
            err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &n);
            err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &o);
            err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &conf->ghosts);
            mgclCheckError(err, "Setting residual squared kernel arguments");

            // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
            size_t global3d[3] = {static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(o)};
            const size_t local3d[3] = {static_cast<size_t>(m > 4 ? 4 : m), static_cast<size_t>(n > 4 ? 4 : n),
                                       static_cast<size_t>(o > 4 ? 4 : o)};

            for (int i = 0; i < 3; i++)
                if (global3d[i] % local3d[i] != 0)
                {
                    // printf("padding global size %d from %ld to ", i, global[i]);
                    global3d[i] += local3d[i] - (global3d[i] % local3d[i]);
                    // printf("%ld (multiple of %ld)\n", global[i], local[i]);
                }

            err = clEnqueueNDRangeKernel(conf->commands, kernel_square, 3, NULL, global3d, local3d, 0, NULL, NULL);
            mgclCheckError(err, "Enqueueing residual squared kernel");

            err = clFinish(conf->commands);
            mgclCheckError(err, "Waiting for kernels to finish");

            err = clEnqueueReadBuffer(conf->commands, d_rsquares, CL_TRUE, 0, sizeof(double) * m * n * o,
                                      rsquares[0][0], 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: Failed to read rsquares array from device!\n%s\n", mgcl_err_code(err));
                exit(1);
            }

            // sum up residual squares
            res = 0;
            for (int i = conf->ghosts; i < m - conf->ghosts; i++)
                for (int j = conf->ghosts; j < n - conf->ghosts; j++)
                    for (int k = conf->ghosts; k < o - conf->ghosts; k++)
                        res += rsquares[i][j][k];
            res = sqrt(res);

            clReleaseMemObject(d_rsquares);
            cuboid_free(rsquares, m, n, o);
            clReleaseKernel(kernel_square);
        }
        else
        {
            // calculate Infinity-Norm (on host, TODO do on opencl)
            err = clFinish(conf->commands);
            mgclCheckError(err, "Waiting for kernels to finish");

            double ***rtmp = cuboid_alloc(m, n, o);

            err = clEnqueueReadBuffer(conf->commands, data->d_r, CL_TRUE, 0, sizeof(double) * m * n * o, rtmp[0][0], 0,
                                      NULL, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: Failed to read rsquares array from device!\n%s\n", mgcl_err_code(err));
                exit(1);
            }

            // find maximum residual
            res = 0;
            for (int i = conf->ghosts; i < m - conf->ghosts; i++)
                for (int j = conf->ghosts; j < n - conf->ghosts; j++)
                    for (int k = conf->ghosts; k < o - conf->ghosts; k++)
                        if (fabs(rtmp[i][j][k]) > res)
                            res = rtmp[i][j][k];

            cuboid_free(rtmp, m, n, o);
        }
    }

    clReleaseKernel(kernel); // TODO maybe clFinish before release?

    return res;
}

/* Runs jacobi method using OpenCL.
 * Doesn't creates ocl buffers and doesn't copy data from host to device and vice versa
 * v, f and r must be of size [m][n][o] for periodic boundary condition.
 * m, n and o must be the dimensions of grid + 2*ghosts
 * If return_residual is true, the residual's 2-norm or inf-norm will be read back from device and returned, else -1.
 * It's not really performant to do so because we have to wait for all kernels to complete and reading a buffer to host
 * is slow. Runs multiple iterations per kernel call using local memory if enough is available Only jacobi_wg_size_x is
 * used for now. If there is not enough local memory available the kernel will not be called and -2 is returned. */
double mgcl_stencil_jacobi_local_mem(mgcl_config *conf, mgcl_level_data *data, int maxiter, int return_residual)
{
    int err;
    int m = data->m;
    int n = data->n;
    int o = data->o;
    int store_res = 0;
    double res = -1;
    int ipk = conf->jacobi_iterations_per_kernel;
    int wg_size = conf->jacobi_wg_size_x;

    if (n - 2 * conf->ghosts < wg_size)
        wg_size = n - 2 * conf->ghosts;
    if (o < n)
        wg_size = o - 2 * conf->ghosts;
    mgcl_debug("Using wg_size = %d (conf->jacobi_wg_size_x = %d)\n", wg_size, conf->jacobi_wg_size_x);

    if (conf->ghosts < conf->jacobi_iterations_per_kernel)
    {
        ipk = conf->ghosts;
        mgcl_debug("Reducing iterations_per_kernel, ghosts = %d < %d = ipk\n", conf->ghosts,
                   conf->jacobi_iterations_per_kernel);
    }

    if (maxiter < conf->jacobi_iterations_per_kernel)
    {
        ipk = maxiter;
        mgcl_debug("Reducing iterations_per_kernel, maxiter = %d < %d = ipk\n", maxiter,
                   conf->jacobi_iterations_per_kernel);
    }

    // check if there is enough local memory available on device for given conf->ghosts = iterations per kernel call
    // TODO do in mgcl_init?
    cl_ulong available_local_mem;
    err = clGetDeviceInfo(conf->device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &available_local_mem, 0);
    mgclCheckError(err, "Querying local memory size info");
    mgcl_debug("Available local memory on device: %ld Bytes\n", available_local_mem);

    // size of shared memory size for one work-group
    int locmem_size_wg = 3 * ipk * (wg_size + 2 * ipk) * (wg_size + 2 * ipk) * sizeof(double);

    // halve wg size until local memory fits
    while (available_local_mem < locmem_size_wg)
    {
        if (wg_size == 1)
        {
            printf("Not enough local memory available to start Jacobi kernel using local memory. Please set "
                   "conf->use_local_memory to false. Aborting.\n");
            return -2;
        }

        wg_size >>= 1;
        locmem_size_wg = 3 * ipk * (wg_size + 2 * ipk) * (wg_size + 2 * ipk) * sizeof(double);
        mgcl_debug("reducing wg_size from %d to %d (now %d Bytes of local memory needed)\n", wg_size << 1, wg_size,
                   locmem_size_wg);
    }
    // TODO pad wg size?
    mgcl_debug("using %d Bytes of local memory\n", locmem_size_wg);

    // ghosted wg size (ghosted grid excluding outmost ghosted border)
    size_t wg_size_ghosted = wg_size + 2 * (ipk - 1);

    double h2 = (1.0 / (double)(m - 2 * conf->ghosts)) *
                (1.0 / (double)(m - 2 * conf->ghosts)); // TODO minimum of m,n,o when not cube?
    double h2inv = 1 / h2;                              // divisor of the stencil, inverted to use * instead of / in kernel

    // Create the compute kernel from the program
    const char *kernel_name;
    if (conf->stencil == MGCL_7POINT_VARSYM)
        kernel_name = "jacobi_stencil_stream_shmem_7point";
    else if (conf->stencil == MGCL_19POINT_VARSYM)
    {
        kernel_name = "jacobi_stencil_stream_shmem_19point";
        h2inv = 1.0 / (6.0 * h2);
    }
    else if (conf->stencil == MGCL_27POINT_VARSYM)
    {
        kernel_name = "jacobi_stencil_stream_shmem_27point";
        h2inv = 1.0 / (30.0 * h2);
    }
    cl_kernel kernel = clCreateKernel(conf->program, kernel_name, &err);
    mgclCheckError(err, "Creating kernel");

    // assign kernel arguments
    int pos = 0;
    err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &data->d_v_in);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &data->d_v_out);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &data->d_f);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &data->d_r);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &data->d_stencil_values);
    err |= clSetKernelArg(kernel, ++pos, locmem_size_wg, NULL);
    err |= clSetKernelArg(kernel, ++pos, sizeof(double), &h2inv);
    err |= clSetKernelArg(kernel, ++pos, sizeof(double), &conf->omega);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &conf->ghosts);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ipk);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &store_res);
    mgclCheckError(err, "Setting kernel arguments");

    // initial kernel dimensions
    size_t global_n = ceil((double)n / (double)wg_size) * wg_size_ghosted;
    size_t global_o = ceil((double)o / (double)wg_size) * wg_size_ghosted;
    size_t global[2] = {global_n, global_o};
    size_t local[2] = {wg_size_ghosted, wg_size_ghosted};
    mgcl_debug("Running Jacobi kernel with %ld,%ld work-items and %ld,%ld work group size\n", global[0], global[1],
               local[0], local[1]);

    cl_mem tmp;
    if (ipk == maxiter)
    {
        // set flag to store residual of last iteration
        store_res = 1;
        err = clSetKernelArg(kernel, pos, sizeof(int), &store_res);
        mgclCheckError(err, "Setting kernel arguments");

        err = mgcl_update_ghosts(conf, data->d_v_in, m, n, o, conf->ghosts, conf->ghosts, conf->ghosts);
        mgclCheckError(err, "Updating ghosts");

        err = clEnqueueNDRangeKernel(conf->commands, kernel, 2, NULL, global, local, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing kernel");

        // swap pointers so result is in d_v_in
        tmp = data->d_v_in;
        data->d_v_in = data->d_v_out;
        data->d_v_out = tmp;
    }
    else
    {
        // start multiple kernels with lesser iterations
        int call_count = ((double)maxiter) / ((double)ipk);
        int iter_rest = maxiter % ipk;
        mgcl_debug("starting kernels %d time(s) for %d iterations and once for %d iteration(s)\n", call_count, ipk,
                   iter_rest);

        // call kernel multiple times with given ipk
        for (int k = 0; k < call_count; k++)
        {
            if (k == call_count - 1 && !iter_rest)
            {
                // set flag to store residual of last iteration
                store_res = 1;
                err = clSetKernelArg(kernel, pos, sizeof(int), &store_res);
                mgclCheckError(err, "Setting kernel argument store_residual");
            }

            err = clEnqueueNDRangeKernel(conf->commands, kernel, 2, NULL, global, local, 0, NULL, NULL);
            mgclCheckError(err, "Enqueueing kernel");

            err = mgcl_update_ghosts(conf, data->d_v_out, m, n, o, conf->ghosts, conf->ghosts, conf->ghosts);
            mgclCheckError(err, "Updating ghosts");

            // swap pointers so result is in d_v_in
            tmp = data->d_v_in;
            data->d_v_in = data->d_v_out;
            data->d_v_out = tmp;

            err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &data->d_v_in);
            err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &data->d_v_out);
        }

        // call once for remaining iterations
        if (iter_rest)
        {
            // set flag to store residual of last iteration
            store_res = 1;
            err = clSetKernelArg(kernel, pos, sizeof(int), &store_res);
            mgclCheckError(err, "Setting kernel argument store_residual");

            // set ipk argument for this kernel call
            err = clSetKernelArg(kernel, pos - 1, sizeof(int), &iter_rest);
            mgclCheckError(err, "Setting kernel arguments iter_rest");

            err = clEnqueueNDRangeKernel(conf->commands, kernel, 2, NULL, global, local, 0, NULL, NULL);
            mgclCheckError(err, "Enqueueing kernel");

            err = mgcl_update_ghosts(conf, data->d_v_out, m, n, o, conf->ghosts, conf->ghosts, conf->ghosts);
            mgclCheckError(err, "Updating ghosts");

            // swap pointers so result is in d_v_in
            tmp = data->d_v_in;
            data->d_v_in = data->d_v_out;
            data->d_v_out = tmp;
        }
    }
    // result is in d_v_in now since pointers were swapped at the end of the loops above

    err = mgcl_update_ghosts(conf, data->d_v_in, m, n, o, conf->ghosts, conf->ghosts, conf->ghosts);
    mgclCheckError(err, "Updating ghosts of v_in");

    // calculate residual's 2-norm. Square elements on device and sum up on host
    if (return_residual)
    {
        if (conf->residual_norm == MGCL_L2)
        {
            // calculate 2-Norm
            double ***rsquares = cuboid_alloc(m, n, o);
            int pointer_flag = conf->device_type == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
            cl_mem d_rsquares = clCreateBuffer(conf->context, CL_MEM_WRITE_ONLY | pointer_flag,
                                               sizeof(double) * m * n * o, rsquares[0][0], &err);
            mgclCheckError(err, "Creating rsquares buffer");

            // Create the compute kernel from the program
            cl_kernel kernel_square = clCreateKernel(conf->program, "residual_squared", &err);
            mgclCheckError(err, "Creating residual squared kernel");

            pos = 0;
            err = clSetKernelArg(kernel_square, pos, sizeof(cl_mem), &data->d_r);
            err |= clSetKernelArg(kernel_square, ++pos, sizeof(cl_mem), &d_rsquares);
            err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &m);
            err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &n);
            err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &o);
            err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &conf->ghosts);
            mgclCheckError(err, "Setting residual squared kernel arguments");

            // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
            size_t global3d[3] = {static_cast<size_t>(m), static_cast<size_t>(n), static_cast<size_t>(o)};
            const size_t local3d[3] = {static_cast<size_t>(m > 4 ? 4 : m), static_cast<size_t>(n > 4 ? 4 : n),
                                       static_cast<size_t>(o > 4 ? 4 : o)};

            for (int i = 0; i < 3; i++)
                if (global3d[i] % local3d[i] != 0)
                {
                    // printf("padding global size %d from %ld to ", i, global[i]);
                    global3d[i] += local3d[i] - (global3d[i] % local3d[i]);
                    // printf("%ld (multiple of %ld)\n", global[i], local[i]);
                }

            err = clEnqueueNDRangeKernel(conf->commands, kernel_square, 3, NULL, global3d, local3d, 0, NULL, NULL);
            mgclCheckError(err, "Enqueueing residual squared kernel");

            err = clFinish(conf->commands);
            mgclCheckError(err, "Waiting for kernels to finish");

            err = clEnqueueReadBuffer(conf->commands, d_rsquares, CL_TRUE, 0, sizeof(double) * m * n * o,
                                      rsquares[0][0], 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: Failed to read rsquares array from device!\n%s\n", mgcl_err_code(err));
                exit(1);
            }

            // sum up residual squares
            res = 0;
            for (int i = conf->ghosts; i < m - conf->ghosts; i++)
                for (int j = conf->ghosts; j < n - conf->ghosts; j++)
                    for (int k = conf->ghosts; k < o - conf->ghosts; k++)
                        res += rsquares[i][j][k];
            res = sqrt(res);

            clReleaseMemObject(d_rsquares);
            cuboid_free(rsquares, m, n, o);
            clReleaseKernel(kernel_square);
        }
        else
        {
            // calculate Infinity-Norm (on host, TODO do on opencl)
            err = clFinish(conf->commands);
            mgclCheckError(err, "Waiting for kernels to finish");

            double ***rtmp = cuboid_alloc(m, n, o);

            err = clEnqueueReadBuffer(conf->commands, data->d_r, CL_TRUE, 0, sizeof(double) * m * n * o, rtmp[0][0], 0,
                                      NULL, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: Failed to read rsquares array from device!\n%s\n", mgcl_err_code(err));
                exit(1);
            }

            // find maximum residual
            res = 0;
            for (int i = conf->ghosts; i < m - conf->ghosts; i++)
                for (int j = conf->ghosts; j < n - conf->ghosts; j++)
                    for (int k = conf->ghosts; k < o - conf->ghosts; k++)
                        if (fabs(rtmp[i][j][k]) > res)
                            res = rtmp[i][j][k];

            cuboid_free(rtmp, m, n, o);
        }
    }

    clReleaseKernel(kernel); // TODO maybe clFinish before release?

    return res;
}

/* Tests jacobi method using OpenCL. Creates buffers and copies memory from host to device and back.
 * v, f and r must be of size [m][n][o] for periodic boundary condition.
 * m, n and o must be the dimensions of grid + 2*ghosts */
void mgcl_stencil_jacobi_test(mgcl_config *conf, mgcl_level_data *data, double ***v, double ***r, int m, int n, int o,
                              int maxiter)
{
    int err;

    if (!conf->stencil_values)
    {
        printf("conf::stencil_values null! Aborting.\n");
        return;
    }

    // create device buffers
    // int pointer_flag = conf->device_type == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
    // cl_mem d_v_in = clCreateBuffer(conf->context, CL_MEM_READ_WRITE | pointer_flag, sizeof(double) * m*n*o, v[0][0],
    // &err); cl_mem d_v_out = clCreateBuffer(conf->context, CL_MEM_READ_WRITE | pointer_flag, sizeof(double) * m*n*o,
    // v[0][0], &err); cl_mem d_f = clCreateBuffer(conf->context, CL_MEM_READ_ONLY | pointer_flag, sizeof(double) *
    // m*n*o, f[0][0], &err); cl_mem d_r = clCreateBuffer(conf->context, CL_MEM_READ_WRITE | pointer_flag,
    // sizeof(double) * m*n*o, r[0][0], &err);

    // create level data
    // mgcl_level_data data = {
    //     .d_v_in = d_v_in,
    //     .d_v_out = d_v_out,
    //     .d_f = d_f,
    //     .d_r = d_r,
    //     .m = m, .n = n, .o = o
    // };

    mgcl_update_ghosts(conf, data->d_v_in, m, n, o, conf->ghosts, conf->ghosts, conf->ghosts);
    mgcl_update_ghosts(conf, data->d_f, m, n, o, conf->ghosts, conf->ghosts, conf->ghosts);
    mgcl_update_ghosts(conf, data->d_stencil_values, m, n, o * conf->stencil_size_multiplier, conf->ghosts,
                       conf->ghosts, conf->ghosts * conf->stencil_size_multiplier);

    // mgcl_print_buffer(conf, data->d_stencil_values, m, n, o * conf->stencil_size_multiplier);

    double t_start_iter = MPI_Wtime();
    mgcl_stencil_jacobi(conf, data, maxiter, 0);

    // Wait for the commands to complete before stopping the timer
    err = clFinish(conf->commands);
    mgclCheckError(err, "Waiting for kernel to finish");
    double t_end_iter = MPI_Wtime();
    printf("jacobi on opencl took %.3e s\n", t_end_iter - t_start_iter);

    // read back results TODO: only for testing purposes, maybe define TESTING?
    err = clEnqueueReadBuffer(conf->commands, data->d_v_in, CL_FALSE, 0, sizeof(double) * m * n * o, v[0][0], 0, NULL,
                              NULL);
    err =
        clEnqueueReadBuffer(conf->commands, data->d_r, CL_TRUE, 0, sizeof(double) * m * n * o, r[0][0], 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error: Failed to read output arrays from device!\n%s\n", mgcl_err_code(err));
        exit(1);
    }

    // clReleaseMemObject(d_v_in);
    // clReleaseMemObject(d_v_out);
    // clReleaseMemObject(d_f);
    // clReleaseMemObject(d_r);
}

/* Calculates the residual using OpenCL.
 * Doesn't creates ocl buffers and doesn't copy data from host to device and vice versa
 * v, f and r must be of size [m][n][o] for periodic boundary condition.
 * m, n and o must be the dimensions of grid + 2.
 * If return_residual is true, the residual's 2-norm or inf-norm will be read back from device and returned, else -1.
 * It's not
 * really performant to do so because we have to wait for all kernels to complete and reading a buffer to host is slow.
 */
double mgcl_stencil_residual(mgcl_config *conf, mgcl_level_data *data, int return_residual)
{
    int err;
    int m = data->m;
    int n = data->n;
    int o = data->o;
    double res = -1;

    double h2 = (1.0 / (double)(m - 2 * conf->ghosts)) *
                (1.0 / (double)(m - 2 * conf->ghosts)); // TODO minimum of m,n,o when not cube?
    double h2inv = 1 / h2;                              // divisor of the stencil, inverted to use * instead of / in kernel

    // Create the compute kernel from the program
    const char *kernel_name;
    if (conf->stencil == MGCL_7POINT_VARSYM)
        kernel_name = "residual_stencil_7point";
    else if (conf->stencil == MGCL_19POINT_VARSYM)
    {
        kernel_name = "residual_stencil_19point";
        h2inv = 1.0 / (6.0 * h2);
    }
    else if (conf->stencil == MGCL_27POINT_VARSYM)
    {
        kernel_name = "residual_stencil_27point";
        h2inv = 1.0 / (30.0 * h2);
    }

    // Create the compute kernel from the program
    cl_kernel kernel = clCreateKernel(conf->program, kernel_name, &err);
    mgclCheckError(err, "Creating kernel");

    // assign kernel arguments
    int pos = 0;
    err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &data->d_v_in);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &data->d_f);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &data->d_r);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &data->d_stencil_values);
    err |= clSetKernelArg(kernel, ++pos, sizeof(double), &h2inv);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &conf->ghosts);
    mgclCheckError(err, "Setting residual kernel arguments");

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

    err = mgcl_update_ghosts(conf, data->d_v_in, m, n, o, conf->ghosts, conf->ghosts, conf->ghosts);
    mgclCheckError(err, "Updating ghosts");
    err = clEnqueueNDRangeKernel(conf->commands, kernel, 3, NULL, global, local, 0, NULL, NULL);
    mgclCheckError(err, "Enqueueing residual kernel");
    err = mgcl_update_ghosts(conf, data->d_r, m, n, o, conf->ghosts, conf->ghosts, conf->ghosts);
    mgclCheckError(err, "Updating ghosts of r");

    // calculate residual's 2-norm. Square elements on device and sum up on host
    if (return_residual)
    {
        if (conf->residual_norm == MGCL_L2)
        {
            // calculate 2-Norm
            double ***rsquares = cuboid_alloc(m, n, o);
            int pointer_flag = conf->device_type == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
            cl_mem d_rsquares = clCreateBuffer(conf->context, CL_MEM_WRITE_ONLY | pointer_flag,
                                               sizeof(double) * m * n * o, rsquares[0][0], &err);
            mgclCheckError(err, "Creating rsquares buffer");

            // Create the compute kernel from the program
            cl_kernel kernel_square = clCreateKernel(conf->program, "residual_squared", &err);
            mgclCheckError(err, "Creating residual squared kernel");

            pos = 0;
            err = clSetKernelArg(kernel_square, pos, sizeof(cl_mem), &data->d_r);
            err |= clSetKernelArg(kernel_square, ++pos, sizeof(cl_mem), &d_rsquares);
            err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &m);
            err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &n);
            err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &o);
            err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &conf->ghosts);
            mgclCheckError(err, "Setting residual squared kernel arguments");

            err = clEnqueueNDRangeKernel(conf->commands, kernel_square, 3, NULL, global, local, 0, NULL, NULL);
            mgclCheckError(err, "Enqueueing residual squared kernel");

            err = clFinish(conf->commands);
            mgclCheckError(err, "Waiting for kernels to finish");

            err = clEnqueueReadBuffer(conf->commands, d_rsquares, CL_TRUE, 0, sizeof(double) * m * n * o,
                                      rsquares[0][0], 0, NULL, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: Failed to read rsquares array from device!\n%s\n", mgcl_err_code(err));
                exit(1);
            }

            // sum up residual squares
            res = 0;
            for (int i = conf->ghosts; i < m - conf->ghosts; i++)
                for (int j = conf->ghosts; j < n - conf->ghosts; j++)
                    for (int k = conf->ghosts; k < o - conf->ghosts; k++)
                        res += rsquares[i][j][k];
            res = sqrt(res);

            clReleaseMemObject(d_rsquares);
            cuboid_free(rsquares, m, n, o);
            clReleaseKernel(kernel_square);
        }
        else
        {
            // calculate Infinity-Norm (on host, TODO do on opencl)
            err = clFinish(conf->commands);
            mgclCheckError(err, "Waiting for kernels to finish");

            double ***rtmp = cuboid_alloc(m, n, o);

            err = clEnqueueReadBuffer(conf->commands, data->d_r, CL_TRUE, 0, sizeof(double) * m * n * o, rtmp[0][0], 0,
                                      NULL, NULL);
            if (err != CL_SUCCESS)
            {
                printf("Error: Failed to read rsquares array from device!\n%s\n", mgcl_err_code(err));
                exit(1);
            }

            // find maximum residual
            res = 0;
            for (int i = conf->ghosts; i < m - conf->ghosts; i++)
                for (int j = conf->ghosts; j < n - conf->ghosts; j++)
                    for (int k = conf->ghosts; k < o - conf->ghosts; k++)
                        if (fabs(rtmp[i][j][k]) > res)
                            res = rtmp[i][j][k];

            cuboid_free(rtmp, m, n, o);
        }
    }

    clReleaseKernel(kernel); // TODO maybe clFinish before release?
    return res;
}
