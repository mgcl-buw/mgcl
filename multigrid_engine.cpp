#include "multigrid_engine.hpp"

namespace mgcl
{

    /* Runs V-cycle recursively and sequentially */
    double MultigridEngine::vcycleSeq(Problem &problem, Level &level)
    {
        auto levelAbove = *problem.levels[level.num + 1];
        // printf("level.getNum() = %d, m = %3.d\n", level.getNum(), level.m-2);
        // problem.maxlevel = 7;
        double res;

        // reset initial guess of coarser grid
        if (level.getNum() < problem.maxlevel - 1)
        {
            for (int i = problem.ghosts; i < levelAbove.m - problem.ghosts; i++)
                for (int j = problem.ghosts; j < levelAbove.n - problem.ghosts; j++)
                    for (int k = problem.ghosts; k < levelAbove.o - problem.ghosts; k++)
                        levelAbove.v[i][j][k] = 0;
        }

        // relax nu1 times
        if (problem.stencil_values)
            res = MultigridEngine::stencilJacobiSeq(level.v, level.f, level.r, level.m - 2 * problem.ghosts,
                                                    level.n - 2 * problem.ghosts, level.o - 2 * problem.ghosts, problem.ghosts,
                                                    problem.omega, problem.nu1, problem.residual_norm, problem.stencil, problem.stencil_values,
                                                    problem.stencil_size_multiplier);
        else
            res = MultigridEngine::jacobiSeq(level.v, level.f, level.r, level.m - 2 * problem.ghosts,
                                             level.n - 2 * problem.ghosts, level.o - 2 * problem.ghosts, problem.ghosts,
                                             problem.omega, problem.nu1, problem.residual_norm, problem.stencil);

        // update residual without D^-1
        // res = residual(level.f, level.v, level.r, level.m-2, level.n-2, level.o-2,
        // problem.residual_norm, problem.stencil); printf("res on level.getNum() %d, upwards: %e\n", level.getNum(), sqrt(res));

        // restrict residual as right hand side on coarser grid
        MultigridEngine::restrictSeq(level, levelAbove, level.r, levelAbove.f);

        // restrict stencil values if stencil is not fixed
        if (problem.stencil_values && problem.restrict_prolongate_stencil)
            MultigridEngine::stencilRestrictSeq(level.stencil_values, levelAbove.stencil_values,
                                                levelAbove.m - 2 * problem.ghosts, levelAbove.n - 2 * problem.ghosts,
                                                levelAbove.o - 2 * problem.ghosts, problem.ghosts, problem.stencil_size_multiplier);

        // if not at highest level.getNum()...
        if (level.getNum() < problem.maxlevel - 1)
        {
            // start next v-cycle iteration
            if (level.getNum() < problem.maxlevel - 2)
                MultigridEngine::vcycleSeq(problem, levelAbove);
            else
            {
                // printf(" pre v[0] = %e, f[0] = %e\n", data[level.getNum()+1].v[1][1][1], data[level.getNum()+1].f[1][1][1]);
                if (problem.stencil_values)
                    MultigridEngine::stencilJacobiSeq(levelAbove.v, levelAbove.f, levelAbove.r,
                                                      levelAbove.m - 2 * problem.ghosts, levelAbove.n - 2 * problem.ghosts,
                                                      levelAbove.o - 2 * problem.ghosts, problem.ghosts, problem.omega,
                                                      problem.nu1 + problem.nu2, problem.residual_norm, problem.stencil, problem.stencil_values,
                                                      problem.stencil_size_multiplier);
                else
                    MultigridEngine::jacobiSeq(levelAbove.v, levelAbove.f, levelAbove.r,
                                               levelAbove.m - 2 * problem.ghosts, levelAbove.n - 2 * problem.ghosts,
                                               levelAbove.o - 2 * problem.ghosts, problem.ghosts, problem.omega, problem.nu1 + problem.nu2,
                                               problem.residual_norm, problem.stencil);

                // printf("post v[0] = %e, f[0] = %e\n", data[level.getNum()+1].v[1][1][1], data[level.getNum()+1].f[1][1][1]);
            }
        }

        // prolongate from coarser to finer grid
        // r of this level.getNum() is reused here and should actually be called e
        MultigridEngine::prolongateSeq(level, levelAbove, level.r, levelAbove.v);

        // prolongate stencil values if stencil is not fixed
        if (problem.stencil_values && problem.restrict_prolongate_stencil)
            MultigridEngine::stencilProlongateSeq(level.stencil_values, levelAbove.stencil_values,
                                                  level.m - 2 * problem.ghosts, level.n - 2 * problem.ghosts,
                                                  level.o - 2 * problem.ghosts, problem.ghosts, problem.stencil_size_multiplier);

        // correct error
        for (int i = problem.ghosts; i < level.m - problem.ghosts; i++)
            for (int j = problem.ghosts; j < level.n - problem.ghosts; j++)
                for (int k = problem.ghosts; k < level.o - problem.ghosts; k++)
                    level.v[i][j][k] += level.r[i][j][k];

        // relax nu2 times
        if (problem.stencil_values)
            res = MultigridEngine::stencilJacobiSeq(level.v, level.f, level.r, level.m - 2 * problem.ghosts,
                                                    level.n - 2 * problem.ghosts, level.o - 2 * problem.ghosts, problem.ghosts,
                                                    problem.omega, problem.nu2, problem.residual_norm, problem.stencil, problem.stencil_values,
                                                    problem.stencil_size_multiplier);
        else
            res = MultigridEngine::jacobiSeq(level.v, level.f, level.r, level.m - 2 * problem.ghosts,
                                             level.n - 2 * problem.ghosts, level.o - 2 * problem.ghosts, problem.ghosts,
                                             problem.omega, problem.nu2, problem.residual_norm, problem.stencil);
        // printf("res on level.getNum() %d, downwards: %e\n", level.getNum(), sqrt(res));
        return res;
    }

    /* Runs V-cycle recursively using ocl */
    double MultigridEngine::vcycle(Problem &problem, Level &level)
    {
        auto levelAbove = *problem.levels[level.num + 1];
        // printf("level.getNum() = %d, m = %3.d\n", level.getNum(), level.m-2);
        // problem.maxlevel = 3;
        double res;
        int err;
        cl_uint zero = 0;

        if (level.getNum() < problem.maxlevel - 1) // if not at highest level
        {
            // reset v to zero for coarser grids (for another possible v-cycle)
            err = clEnqueueFillBuffer(problem.openCLHelper.getCommands(), levelAbove.dVIn, &zero, sizeof(cl_uint), 0,
                                      sizeof(double) * levelAbove.m * levelAbove.n * levelAbove.o, 0, NULL,
                                      NULL);
            mgclCheckError(err, "resetting dVIn to 0");
        }

        // relax nu1 times
        if (problem.stencil_values)
            res = stencilJacobi(problem, level, problem.nu1, 1);
        else
            res = jacobi(problem, level, problem.nu1, 1);

        // update residual without D^-1
        // res = residual(problem, level, 1);
        // printf("res on level.getNum() %d, upwards: %e\n", level.getNum(), res);

        // restrict to coarser grid
        restrict(level, levelAbove);

        // restrict stencil values if stencil is not fixed
        if (problem.stencil_values && problem.restrict_prolongate_stencil)
            stencilRestrict(problem, level, levelAbove);

        if (level.getNum() < problem.maxlevel - 1)
        {
            // start next v-cycle iteration
            if (level.getNum() < problem.maxlevel - 2)
                vcycle(problem, levelAbove);
            else
            {
                if (problem.stencil_values)
                    res = stencilJacobi(problem, levelAbove, problem.nu1 + problem.nu2, 0);
                else
                    res = jacobi(problem, levelAbove, problem.nu1 + problem.nu2, 0);
            }
        }

        // prolongate from coarser to finer grid
        // r of this level.getNum() is reused here and should actually be called e
        prolongate(level, levelAbove);

        // prolongate stencil values if stencil is not fixed
        if (problem.stencil_values && problem.restrict_prolongate_stencil)
            stencilProlongate(problem, level, levelAbove);

        // correct error
        correctError(problem, level.dVIn, level.dR, level.m, level.n, level.o);

        // relax nu2 times
        if (problem.stencil_values)
            res = stencilJacobi(problem, level, problem.nu2, 1);
        else
            res = jacobi(problem, level, problem.nu2, 1);

        // calculate residual again for the norm TODO in jacobi
        // res = residual(problem, level, 1);
        // printf("res on level.getNum() %d, downwards: %e\n", level.getNum(), res);
        return res;
    }

    /* Starts kernel to correct error, e.g. v = v + e
     * m,n,o is size of ghosted grid */
    int MultigridEngine::correctError(Problem &problem, cl_mem d_v, cl_mem d_r, int m, int n, int o)
    {
        int err;

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(problem.openCLHelper.getProgram(), "correct_error", &err);
        mgclCheckError(err, "Creating kernel");

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &d_v);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &d_r);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &problem.ghosts);
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

        err = clEnqueueNDRangeKernel(problem.openCLHelper.getCommands(), kernel, 3, NULL, global, local, 0, NULL, NULL);
        mgclCheckError(err, "Enqueueing kernel");

        clReleaseKernel(kernel);
        return err;
    }

    /* Reads values for one level from device for testing purposes */
    void MultigridEngine::testRead(Problem &problem, Level &level)
    {
        int err = clFinish(problem.openCLHelper.getCommands());
        mgclCheckError(err, "Waiting for kernels to finish");

        // read back results TODO: only for testing purposes, maybe define TESTING?
        err = clEnqueueReadBuffer(problem.openCLHelper.getCommands(), level.dVIn, CL_TRUE, 0,
                                  sizeof(double) * level.m * level.n * level.o, level.v[0][0], 0,
                                  NULL, NULL);
        err = clEnqueueReadBuffer(problem.openCLHelper.getCommands(), level.dF, CL_TRUE, 0,
                                  sizeof(double) * level.m * level.n * level.o, level.f[0][0], 0,
                                  NULL, NULL);
        err = clEnqueueReadBuffer(problem.openCLHelper.getCommands(), level.dR, CL_TRUE, 0,
                                  sizeof(double) * level.m * level.n * level.o, level.r[0][0], 0,
                                  NULL, NULL);
        mgclCheckError(err, "Error: Failed to read output arrays from device!");

        printf("0 level = %d, v[1,1,1] = %e\n", level.getNum(), level.v[1][1][1]);
        printf("0 level = %d, f[1,1,1] = %e\n", level.getNum(), level.f[1][1][1]);
        printf("0 level = %d, r[1,1,1] = %e\n", level.getNum(), level.r[1][1][1]);
    }
}