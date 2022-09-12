#include "multigrid_engine.hpp"

#include <iomanip>
#include <iostream>

namespace mgcl
{

    /* Runs V-cycle recursively and sequentially */
    double MultigridEngine::vcycleSeq(Problem &problem, Level &level)
    {
        auto levelAbove = problem.levels[level.num + 1];
        // printf("level.getNum() = %d, m = %3.d\n", level.getNum(), level.m-2);
        // problem.maxlevel = 7;
        double res;

        // reset initial guess of coarser grid
        if (level.getNum() < problem.maxlevel)
        {
            levelAbove->getV().fill(0);
        }

        // if (level.num == 1)
        // {
        //     level.getV().dumpToFile("out_v.txt");
        //     level.getF().dumpToFile("out_f.txt");
        //     level.getR().dumpToFile("out_r.txt");
        // }

        // relax nu1 times
        res = MultigridEngine::jacobiSeq(level.getV(), level.getF(), level.getR(),
                                         problem.omega, problem.nu1, problem.residual_norm, *level.stencil, false);

        // update residual without D^-1
        // res = residual(level.f, level.v, level.r, level.m-2, level.n-2, level.o-2,
        // problem.residual_norm, problem.stencil);
        // printf("res on level %d, upwards: %.17e\n", level.getNum(), res);
        // std::cout << "v[3][3][3] = " << std::scientific << std::setprecision(17) << level.getV()[3][3][3] << std::endl
        //           << "f[3][3][3] = " << std::scientific << std::setprecision(17) << level.getF()[3][3][3] << std::endl
        //           << "r[3][3][3] = " << std::scientific << std::setprecision(17) << level.getR()[3][3][3] << std::endl;

        // if (level.num == 0)
        // {
        //     level.getV().dumpToFile("out_v.txt");
        //     level.getF().dumpToFile("out_f.txt");
        //     level.getR().dumpToFile("out_r.txt");
        // }

        // restrict residual as right hand side on coarser grid
        MultigridEngine::restrictSeq(level, *levelAbove, level.getR(), levelAbove->getF());

        // start next v-cycle iteration if not at highest level
        if (level.getNum() < problem.maxlevel - 1)
            MultigridEngine::vcycleSeq(problem, *levelAbove);
        else
        {
            MultigridEngine::jacobiSeq(levelAbove->getV(), levelAbove->getF(), levelAbove->getR(), problem.omega, problem.nu1 + problem.nu2,
                                       problem.residual_norm, *level.stencil, false);

            // printf("post v[0] = %e, f[0] = %e\n", data[level.getNum()+1].getV()[1][1][1], data[level.getNum()+1].getF()[1][1][1]);
        }

        // prolongate from coarser to finer grid
        // r of this level is reused here and should actually be called e
        MultigridEngine::prolongateSeq(level, *levelAbove, level.getR(), levelAbove->getV());

        // correct error
        for (int i = problem.ghosts; i < level.m + problem.ghosts; i++)
            for (int j = problem.ghosts; j < level.n + problem.ghosts; j++)
                for (int k = problem.ghosts; k < level.o + problem.ghosts; k++)
                    level.getV()[i][j][k] += level.getR()[i][j][k];

        // relax nu2 times
        res = MultigridEngine::jacobiSeq(level.getV(), level.getF(), level.getR(),
                                         problem.omega, problem.nu2, problem.residual_norm, *level.stencil, !problem.ignoreTol);
        // printf("res on level %d, downwards: %.17e\n", level.getNum(), res);
        return res;
    }

    /* Runs V-cycle recursively using ocl */
    double MultigridEngine::vcycle(Problem &problem, Level &level)
    {
        auto levelAbove = problem.levels[level.num + 1];
        // printf("level.getNum() = %d, m = %3.d\n", level.getNum(), level.m-2);
        // problem.maxlevel = 3;
        double res;
        int err;
        cl_double zero = 0;

        if (level.getNum() < problem.maxlevel) // if not at highest level
        {
            // reset v to zero for coarser grids (for another possible v-cycle)
            err = clEnqueueFillBuffer(problem.openCLHelper.getCommands(), levelAbove->dVIn, &zero, sizeof(cl_double), 0,
                                      sizeof(double) * levelAbove->mgh * levelAbove->ngh * levelAbove->ogh, 0, NULL,
                                      NULL);
            mgclCheckError(err, "resetting dVIn to 0");
        }

        // relax nu1 times
        jacobi(problem, level, problem.nu1, 0);

        // update residual without D^-1
        // res = residual(problem, level, 1);
        // printf("res on level.getNum() %d, upwards: %e\n", level.getNum(), res);

        // restrict to coarser grid
        restrict(level, *levelAbove, level.getDR(), levelAbove->getDF());

        // start next v-cycle iteration if not at highest level
        if (level.getNum() < problem.maxlevel - 1)
            vcycle(problem, *levelAbove);
        else
        {
            jacobi(problem, *levelAbove, problem.nu1 + problem.nu2, 0);
        }

        // prolongate from coarser to finer grid
        // r of this level.getNum() is reused here and should actually be called e
        prolongate(level, *levelAbove, level.dR, levelAbove->dVIn);

        // correct error
        correctError(problem, level.dVIn, level.dR, level.mgh, level.ngh, level.ogh);

        // relax nu2 times
        res = jacobi(problem, level, problem.nu2, !problem.ignoreTol);

        // calculate residual again for the norm TODO in jacobi
        // res = residual(problem, level, 1);
        // printf("res on level.getNum() %d, downwards: %e\n", level.getNum(), res);
        return res;
    }

    /* Starts kernel to correct error, e.g. v = v + e
     * m,n,o is size of ghosted grid */
    int MultigridEngine::correctError(Problem &problem, cl_mem d_v, cl_mem d_r, int mgh, int ngh, int ogh)
    {
        int err;

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(problem.openCLHelper.getProgram(), "correct_error", &err);
        mgclCheckError(err, "Creating kernel");

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &d_v);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &d_r);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &problem.ghosts);
        mgclCheckError(err, "Setting kernel arguments");

        // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
        size_t global[3] = {static_cast<size_t>(mgh), static_cast<size_t>(ngh), static_cast<size_t>(ogh)};
        const size_t local[3] = {static_cast<size_t>(mgh > 4 ? 4 : mgh), static_cast<size_t>(ngh > 4 ? 4 : ngh),
                                 static_cast<size_t>(ogh > 4 ? 4 : ogh)};

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
                                  sizeof(double) * level.m * level.n * level.o, level.getV()[0][0], 0,
                                  NULL, NULL);
        err = clEnqueueReadBuffer(problem.openCLHelper.getCommands(), level.dF, CL_TRUE, 0,
                                  sizeof(double) * level.m * level.n * level.o, level.getF()[0][0], 0,
                                  NULL, NULL);
        err = clEnqueueReadBuffer(problem.openCLHelper.getCommands(), level.dR, CL_TRUE, 0,
                                  sizeof(double) * level.m * level.n * level.o, level.getR()[0][0], 0,
                                  NULL, NULL);
        mgclCheckError(err, "Error: Failed to read output arrays from device!");

        printf("0 level = %d, v[1,1,1] = %e\n", level.getNum(), level.getV()[1][1][1]);
        printf("0 level = %d, f[1,1,1] = %e\n", level.getNum(), level.getF()[1][1][1]);
        printf("0 level = %d, r[1,1,1] = %e\n", level.getNum(), level.getR()[1][1][1]);
    }
}