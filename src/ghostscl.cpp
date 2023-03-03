#include "cuboid.hpp"
#include "multigrid_engine.hpp"
#include "opencl_helper.hpp"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <cstddef>

namespace mgcl
{
    using std::size_t;

    /* updates ghost cells for periodic boundary condition
     * m,n,o are dimensions of real grid without ghost cells */
    void MultigridEngine::updateGhostsSeq(Cuboid &c)
    {
        int m = c.getM();
        int n = c.getN();
        int o = c.getO();
        int ghosts_m = c.getGhostsM();
        int ghosts_n = c.getGhostsN();
        int ghosts_o = c.getGhostsO();

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

        // now send diagonal edges in each direction
        // for (int i = 1; i < m + 1; i++)
        // {
        //     c[i][0][0] = c[i][n][o]; // top front
        //     c[i][0][o+1] = c[i][n][1]; // top back
        //     c[i][n+1][0] = c[i][1][o]; // bottom front
        //     c[i][n+1][o+1] = c[i][1][1]; // bottom back

        //     c[0][i][0] = c[m][i][o]; // front left
        //     c[0][i][o+1] = c[m][i][1]; // back left
        //     c[m+1][i][0] = c[1][i][o]; // front left
        //     c[m+1][i][o+1] = c[1][i][1]; // back right

        //     c[0][0][i] = c[m][n][i]; // top left
        //     c[0][n+1][i] = c[m][1][i]; // bottom left
        //     c[m+1][0][i] = c[1][n][i]; // top right
        //     c[m+1][n+1][i] = c[1][1][i]; // bottom right
        // }

        // // corners
        // c[0][0][0] = c[m][n][o]; // top left front
        // c[0][0][o+1] = c[m][n][1]; // top left back
        // c[0][n+1][0] = c[m][1][o]; // bottom left front
        // c[0][n+1][o+1] = c[m][1][1]; // bottom left back
        // c[m+1][0][0] = c[1][n][o]; // top right front
        // c[m+1][0][o+1] = c[1][n][1]; // top right back
        // c[m+1][n+1][0] = c[1][1][o]; // bottom right front
        // c[m+1][n+1][o+1] = c[1][1][1]; // bottom right back

        // printf("c[1,0,0] = %f\n", c[1][0][0]);
    }

    /* updates ghost cells on opencl device.
     * m,n,o must be size of ghosted grid.
     * Only enqueues the kernel. Neither waits for kernel to finish nor reads back results */
    int MultigridEngine::updateGhosts(Problem &problem, cl_mem dBuffer, int mgh, int ngh, int ogh, int ghosts_m, int ghosts_n, int ghosts_o)
    {
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
