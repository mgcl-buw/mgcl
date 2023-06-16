#include "cuboid.hpp"           // for Cuboid
#include "mgcl.hpp"             // for BC
#include "multigrid_engine.hpp" // for Problem, MultigridEngine
#include "opencl_helper.hpp"    // for mgclCheckError, OpenCLHelper

#include <cassert>
#include <cstddef> // for size_t, NULL

#ifdef MGCL_USE_MPI
#include "mpi_data.hpp"
#endif // MGCL_USE_MPI

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace mgcl
{
    using std::size_t;

    /* Updates ghost cells for periodic boundary condition.
     * mpiData parameter is optional (i.e. nullable) and is only used when MPI is used. */
    void MultigridEngine::updateGhostsSeq(Cuboid &c, MPIData *mpiData, bool periodic)
    {
        int m = c.getM();
        int n = c.getN();
        int o = c.getO();
        int ghosts_m = c.getGhostsM();
        int ghosts_n = c.getGhostsN();
        int ghosts_o = c.getGhostsO();

#ifdef MGCL_USE_MPI
        // TODO adjust for ghosts > 1
        // TODO test
        assert(mpiData != nullptr && "mpiData must not be null if MGCL_USE_MPI is true!");

        /* MPI variables */
        int myid;
        MPI_Status stats[2];
        MPI_Request reqs[2];

        /* Loop variables */
        int i, j, k;
        // int m = data[level].m_l, n = data[level].n_l, o = data[level].o_l;

        // Rectangle buffers
        double **sbufxy = mpiData->sbufxy();
        double **sbufxz = mpiData->sbufxz();
        double **sbufyz = mpiData->sbufyz();
        double **rbufxy = mpiData->rbufxy();
        double **rbufxz = mpiData->rbufxz();
        double **rbufyz = mpiData->rbufyz();

        /* Getting local rank */
        MPI_Comm_rank(mpiData->comm, &myid);

        /* Sending data to left */
        for (j = 0; j < n; j++)
        {
            for (k = 0; k < o; k++)
            {
                sbufyz[j][k] = c[1][j][k];
            }
        }
        reqs[0] = MPI_REQUEST_NULL;
        reqs[1] = MPI_REQUEST_NULL;
        if (myid != mpiData->left)
            MPI_Isend((void *)sbufyz[0], n * o, MPI_DOUBLE, mpiData->left, 0, mpiData->comm, &reqs[0]);
        if (myid != mpiData->right)
            MPI_Irecv((void *)rbufyz[0], n * o, MPI_DOUBLE, mpiData->right, 0, mpiData->comm, &reqs[1]);
        MPI_Waitall(2, reqs, stats);
        if (MPI_PROC_NULL != mpiData->right)
            for (j = 0; j < n; j++)
            {
                for (k = 0; k < o; k++)
                {
                    c[m - 1][j][k] = rbufyz[j][k];
                }
            }

        /* Sending data to right */
        for (j = 0; j < n; j++)
        {
            for (k = 0; k < o; k++)
            {
                sbufyz[j][k] = c[m - 2][j][k];
            }
        }
        reqs[0] = MPI_REQUEST_NULL;
        reqs[1] = MPI_REQUEST_NULL;
        if (myid != mpiData->right)
            MPI_Isend((void *)sbufyz[0], n * o, MPI_DOUBLE, mpiData->right, 0, mpiData->comm, &reqs[0]);
        if (myid != mpiData->left)
            MPI_Irecv((void *)rbufyz[0], n * o, MPI_DOUBLE, mpiData->left, 0, mpiData->comm, &reqs[1]);
        MPI_Waitall(2, reqs, stats);
        if (MPI_PROC_NULL != mpiData->left)
            for (j = 0; j < n; j++)
            {
                for (k = 0; k < o; k++)
                {
                    c[0][j][k] = rbufyz[j][k];
                }
            }

        /* Sending data downwards */
        for (i = 0; i < m; i++)
        {
            for (k = 0; k < o; k++)
            {
                sbufxz[i][k] = c[i][1][k];
            }
        }
        reqs[0] = MPI_REQUEST_NULL;
        reqs[1] = MPI_REQUEST_NULL;
        if (myid != mpiData->down)
            MPI_Isend((void *)sbufxz[0], m * o, MPI_DOUBLE, mpiData->down, 0, mpiData->comm, &reqs[0]);
        if (myid != mpiData->up)
            MPI_Irecv((void *)rbufxz[0], m * o, MPI_DOUBLE, mpiData->up, 0, mpiData->comm, &reqs[1]);
        MPI_Waitall(2, reqs, stats);
        if (MPI_PROC_NULL != mpiData->up)
            for (i = 0; i < m; i++)
            {
                for (k = 0; k < o; k++)
                {
                    c[i][n - 1][k] = rbufxz[i][k];
                }
            }

        /* Sending data upwards */
        for (i = 0; i < m; i++)
        {
            for (k = 0; k < o; k++)
            {
                sbufxz[i][k] = c[i][n - 2][k];
            }
        }
        reqs[0] = MPI_REQUEST_NULL;
        reqs[1] = MPI_REQUEST_NULL;
        if (myid != mpiData->up)
            MPI_Isend((void *)sbufxz[0], m * o, MPI_DOUBLE, mpiData->up, 0, mpiData->comm, &reqs[0]);
        if (myid != mpiData->down)
            MPI_Irecv((void *)rbufxz[0], m * o, MPI_DOUBLE, mpiData->down, 0, mpiData->comm, &reqs[1]);
        MPI_Waitall(2, reqs, stats);
        if (MPI_PROC_NULL != mpiData->down)
            for (i = 0; i < m; i++)
            {
                for (k = 0; k < o; k++)
                {
                    c[i][0][k] = rbufxz[i][k];
                }
            }

        /* Sending data backwards */
        for (i = 0; i < m; i++)
        {
            for (j = 0; j < n; j++)
            {
                sbufxy[i][j] = c[i][j][1];
            }
        }
        reqs[0] = MPI_REQUEST_NULL;
        reqs[1] = MPI_REQUEST_NULL;
        if (myid != mpiData->back)
            MPI_Isend((void *)sbufxy[0], m * n, MPI_DOUBLE, mpiData->back, 0, mpiData->comm, &reqs[0]);
        if (myid != mpiData->front)
            MPI_Irecv((void *)rbufxy[0], m * n, MPI_DOUBLE, mpiData->front, 0, mpiData->comm, &reqs[1]);
        MPI_Waitall(2, reqs, stats);
        if (MPI_PROC_NULL != mpiData->front)
            for (i = 0; i < m; i++)
            {
                for (j = 0; j < n; j++)
                {
                    c[i][j][o - 1] = rbufxy[i][j];
                }
            }

        /* Sending data to front */
        for (i = 0; i < m; i++)
        {
            for (j = 0; j < n; j++)
            {
                sbufxy[i][j] = c[i][j][o - 2];
            }
        }
        reqs[0] = MPI_REQUEST_NULL;
        reqs[1] = MPI_REQUEST_NULL;
        if (myid != mpiData->front)
            MPI_Isend((void *)sbufxy[0], m * n, MPI_DOUBLE, mpiData->front, 0, mpiData->comm, &reqs[0]);
        if (myid != mpiData->back)
            MPI_Irecv((void *)rbufxy[0], m * n, MPI_DOUBLE, mpiData->back, 0, mpiData->comm, &reqs[1]);
        MPI_Waitall(2, reqs, stats);
        if (MPI_PROC_NULL != mpiData->back)
            for (i = 0; i < m; i++)
            {
                for (j = 0; j < n; j++)
                {
                    c[i][j][0] = rbufxy[i][j];
                }
            }

        if (periodic)
        {
            if (myid == mpiData->back && myid == mpiData->front)
            {
                for (i = 0; i < m; i++)
                {
                    for (j = 0; j < n; j++)
                    {
                        c[i][j][0] = c[i][j][o - 2];
                        c[i][j][o - 1] = c[i][j][1];
                    }
                }
            }

            if (myid == mpiData->down && myid == mpiData->up)
            {
                for (i = 0; i < m; i++)
                {
                    for (k = 0; k < o; k++)
                    {
                        c[i][0][k] = c[i][n - 2][k];
                        c[i][n - 1][k] = c[i][1][k];
                    }
                }
            }

            if (myid == mpiData->left && myid == mpiData->right)
            {
                for (j = 0; j < n; j++)
                {
                    for (k = 0; k < o; k++)
                    {
                        c[0][j][k] = c[m - 2][j][k];
                        c[m - 1][j][k] = c[1][j][k];
                    }
                }
            }
        }
#else
        // sending data in x-direction
        for (int i = 0; i < n + 2 * ghosts_n; i++)
            for (int j = 0; j < o + 2 * ghosts_o; j++)
                for (int k = 0; k < ghosts_m; k++)
                {
                    c[k][i][j] = c[m + k][i][j];                       // left ghost cell = right real cell
                    c[m + ghosts_m + k][i][j] = c[ghosts_m + k][i][j]; // right ghost cell = left real cell
                }

        // sending data in y-direction
        for (int i = 0; i < m + 2 * ghosts_m; i++)
            for (int j = 0; j < o + 2 * ghosts_o; j++)
                for (int k = 0; k < ghosts_n; k++)
                {
                    c[i][k][j] = c[i][n + k][j];                       // top ghost cell = bottom real cell
                    c[i][n + ghosts_n + k][j] = c[i][ghosts_n + k][j]; // bottom ghost cell = top real cell
                }

        // sending data in z-direction
        for (int i = 0; i < m + 2 * ghosts_m; i++)
            for (int j = 0; j < n + 2 * ghosts_n; j++)
                for (int k = 0; k < ghosts_o; k++)
                {
                    c[i][j][k] = c[i][j][o + k];                       // front ghost cell = back real cell
                    c[i][j][o + ghosts_o + k] = c[i][j][ghosts_o + k]; // back ghost cell = front real cell
                }

#endif // MGCL_USE_MPI
    }

    /* updates ghost cells on opencl device.
     * m,n,o must be size of ghosted grid.
     * Only enqueues the kernel. Neither waits for kernel to finish nor reads back results */
    int MultigridEngine::updateGhosts(Problem &problem, cl_mem dBuffer, int mgh, int ngh, int ogh, int ghosts_m, int ghosts_n, int ghosts_o)
    {
        if (problem.bc != BC::PERIODIC)
            return CL_SUCCESS;

        int err;

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(problem.getOpenCLHelper().getProgram(), "update_ghosts_2d", &err);
        mgclCheckError(err, "Creating kernel");

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dBuffer);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_o);
        mgclCheckError(err, "Setting kernel arguments");

        // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
        size_t global[2] = {static_cast<size_t>(ngh), static_cast<size_t>(ogh)};
        const size_t local[2] = {static_cast<size_t>(ngh > 4 ? 4 : ngh), static_cast<size_t>(ogh > 4 ? 4 : ogh)};

        for (int i = 0; i < 2; i++)
            if (global[i] % local[i] != 0)
            {
                // printf("padding global size %d from %ld to ", i, global[i]);
                global[i] += local[i] - (global[i] % local[i]);
                // printf("%ld (multiple of %ld)\n", global[i], local[i]);
            }

        err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel, 2, NULL, global, local, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing kernel");

        clReleaseKernel(kernel);
        return err;
    }
}
