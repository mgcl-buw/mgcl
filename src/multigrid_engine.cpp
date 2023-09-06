#include "multigrid_engine.hpp"
#include "cuboid.hpp"    // for Cuboid
#include "hypercube.hpp" // for Hypercube6d
#include "level.hpp"     // for Level
#include "mpi_util.hpp"
#include "opencl_helper.hpp" // for mgclCheckError, OpenCLHelper
#include "problem.hpp"       // for Problem

#include <cstddef> // for size_t, NULL
// #include <iostream>
#include <memory> // for __shared_ptr_access, shared_ptr

#ifdef __APPLE__
#include <OpenCL/cl_platform.h>
#else
#include <CL/cl_platform.h> // for cl_double
#endif

namespace mgcl
{

    /* Runs V-cycle recursively and sequentially */
    double MultigridEngine::vcycleSeq(Problem& problem, Level& level)
    {
        auto& levelAbove = problem.getLevelAt(level.num + 1);
        // printf("level.getNum() = %d, m = %3.d\n", level.getNum(), level.m-2);
        // problem.maxlevel = 7;
        double res;

        // reset initial guess of coarser grid
        if (level.getNum() < problem.maxlevel && levelAbove.getVPtr() != nullptr)
        {
            levelAbove.getV().fill(0);
        }

        // if (level.num == 1)
        // {
        //     level.getV().dumpToFile("out_v.txt");
        //     level.getF().dumpToFile("out_f.txt");
        //     level.getR().dumpToFile("out_r.txt");
        // }

        // std::cout << level << std::endl;

        // relax nu1 times
        res = MultigridEngine::jacobiSeq(level.getV(), level.getF(), level.getR(),
                                         problem.omega, level.h * level.h, problem.nu1, problem.residual_norm, problem.stencilType,
                                         level.stencilFactor, level.stencilValues.get(), false, problem.isPeriodic(),
                                         level.isCalculatedLocally(), 1, level.getMpiDataPtr());

        // update residual before restriction
        res = residualSeq(level.getF(), level.getV(), level.getR(), problem.residual_norm, problem.stencilType,
                          level.stencilFactor, level.stencilValues.get(), false, problem.isPeriodic(),
                          level.isCalculatedLocally(), 0, 0, 0, level.getMpiDataPtr());

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
        // TODO do not update ghosts of coarse grid before gather (size too small)
        MultigridEngine::restrictSeq(level, levelAbove, level.getR(), levelAbove.getF());

        // If MPI is in use but minGridPoints is reached, gather rhs data to process 0 and perform calculations
        // locally only, until we're reaching the threshold level moving downwards again.
        if (problem.useMpi() && problem.getMpiLevelThreshold() == levelAbove.getNum())
        {
            // TODO Gather on lv with loc = minGridPoints
            // example: minGridPoints=4, do this on lv 1 with values of lv 2
            //   rank 0 lv 1: glob=16, loc=4, lv 2: glob=loc=8
            // gather into levelAbove.getF() (reuse on rank 0)

            mpi_util::gather(problem.getMpiComm(), levelAbove.getF());

            // Update ghosts of gathered
            // TODO check Dirichlet
            // TODO only on rank 0
            if (problem.isPeriodic())
                MultigridEngine::updateGhostsSeq(levelAbove.getF(), levelAbove.getMpiDataPtr(), problem.isPeriodic(),
                                                 levelAbove.isCalculatedLocally());

            // do rest on proc 0
        }

        // Advance to coarser levels only on proc 0 after gather
        if (!problem.useMpi() || (problem.useMpi() && levelAbove.isCalculatedLocally() && problem.mpiRank() == 0))
        {
            // start next v-cycle iteration if not at highest level
            if (level.getNum() < problem.maxlevel - 1)
                MultigridEngine::vcycleSeq(problem, levelAbove);
            else
            {
                MultigridEngine::jacobiSeq(levelAbove.getV(), levelAbove.getF(), levelAbove.getR(), problem.omega,
                                           levelAbove.h * levelAbove.h, problem.nu1 + problem.nu2, problem.residual_norm,
                                           problem.stencilType, levelAbove.stencilFactor,
                                           levelAbove.stencilValues.get(), false, problem.isPeriodic(),
                                           levelAbove.isCalculatedLocally(), 1, levelAbove.getMpiDataPtr());

                // printf("post v[0] = %e, f[0] = %e\n", data[level.getNum()+1].getV()[1][1][1], data[level.getNum()+1].getF()[1][1][1]);
            }
        }

        // If MPI is in use but minGridPoints is reached, scatter v data from process 0 to others and continue
        // distributed calulcations.
        std::shared_ptr<Cuboid> tmp;
        if (problem.useMpi() && problem.getMpiLevelThreshold() == levelAbove.getNum())
        {
            // TODO Scatter on lv with loc = minGridPoints
            // example: minGridPoints=4, do this on lv 1 with values of lv 2
            //   rank 0 lv 1: glob=16, loc=4, lv 2: glob=loc=8
            // scatter into levelAbove.getV() (size 8 -> 2), update ghosts of V, then prolongate into level.getR() (size 4)
            // ghost update level.getR() needed before prolongate -> adjust size of level.getR()

            if (problem.mpiRank() == 0)
            {
                // TODO check if tmp buffer is ok
                // levelAbove has bigger size on rank 0, thus calculate using level which has local size as base
                tmp = std::make_shared<Cuboid>(level.getM() / 2, level.getN() / 2, level.getO() / 2,
                                               problem.getGhosts(), problem.getGhosts(), problem.getGhosts());
                mpi_util::scatter(problem.getMpiComm(), levelAbove.getVPtr().get(), *tmp);
                levelAbove.setV(tmp);
            }
            else
                mpi_util::scatter(problem.getMpiComm(), nullptr, levelAbove.getV());
        }

        // prolongate from coarser to finer grid
        // r of this level is reused here and should actually be called e
        MultigridEngine::prolongateSeq(level, levelAbove, level.getR(), levelAbove.getV());

        // correct error
        for (int i = problem.ghosts; i < level.m + problem.ghosts; i++)
            for (int j = problem.ghosts; j < level.n + problem.ghosts; j++)
                for (int k = problem.ghosts; k < level.o + problem.ghosts; k++)
                    level.getV()[i][j][k] += level.getR()[i][j][k];

        // relax nu2 times
        res = MultigridEngine::jacobiSeq(level.getV(), level.getF(), level.getR(),
                                         problem.omega, level.h * level.h, problem.nu2, problem.residual_norm, problem.stencilType,
                                         level.stencilFactor, level.stencilValues.get(), !problem.ignoreTol,
                                         problem.isPeriodic(), level.isCalculatedLocally(), 1, level.getMpiDataPtr());
        // printf("res on level %d, downwards: %.17e\n", level.getNum(), res);
        return res;
    }

    /* Runs V-cycle recursively using ocl */
    double MultigridEngine::vcycle(Problem& problem, Level& level)
    {
        auto& levelAbove = problem.getLevelAt(level.num + 1);
        // printf("level.getNum() = %d, m = %3.d\n", level.getNum(), level.m-2);
        // problem.maxlevel = 3;
        double res;
        int err;
        cl_double zero = 0;

        if (level.getNum() < problem.maxlevel) // if not at highest level
        {
            // reset v to zero for coarser grids (for another possible v-cycle)
            err = clEnqueueFillBuffer(problem.openCLHelper.getCommands(), levelAbove.dVIn, &zero, sizeof(cl_double), 0,
                                      sizeof(double) * levelAbove.mgh * levelAbove.ngh * levelAbove.ogh, 0, NULL,
                                      NULL);
            mgclCheckError(err, "resetting dVIn to 0");
        }

        // relax nu1 times
        res = jacobi(problem, level, problem.nu1, false);

        // update residual before restriction
        res = residual(problem, level, false);
        // printf("res on level.getNum() %d, upwards: %e\n", level.getNum(), res);

        // restrict to coarser grid
        restrict(level, levelAbove, level.getDR(), levelAbove.getDF());

        // start next v-cycle iteration if not at highest level
        if (level.getNum() < problem.maxlevel - 1)
            vcycle(problem, levelAbove);
        else
        {
            jacobi(problem, levelAbove, problem.nu1 + problem.nu2, 0);
        }

        // prolongate from coarser to finer grid
        // r of this level.getNum() is reused here and should actually be called e
        prolongate(level, levelAbove, level.dR, levelAbove.dVIn);

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
    int MultigridEngine::correctError(Problem& problem, cl_mem d_v, cl_mem d_r, int mgh, int ngh, int ogh)
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

    /**
     * @brief Calculates and sets the stencil (i.e. the matrix A) for the current level by applying the
     * Galerkin operator, which is defined as A_2h = R * A_h * P with R being restriction and P being prolongation
     * operators.
     *
     * @param a_h The stencil of the finer grid.
     * @returns VaryingStencil3x3x3 The stencil to be applied on the coarser grid
     */
    VaryingStencil3x3x3 MultigridEngine::galerkin(VaryingStencil3x3x3& a_h)
    {
        // TODO respect problem::ghosts maybe

        // Make sure a_h has two ghosts at each border for periodic bc.
        if (a_h.getGhostsDim1() != 2 || a_h.getGhostsDim3() != 2 || a_h.getGhostsDim3() != 2)
            throw "galerkin: a_h needs to have 2 ghosts at each border for periodic bc!";

        // Get the full-weight restriction stencil S as 3x3x3 stencil with two additional ghosts at each border.
        // The ghosts are needed in order to respect periodic boundary conditions. One ghost per stencil multiplication.
        auto sr = create3dFullWeightRestrictionStencil();
        auto sp = create3dBilinearProlongationStencil();

        // A_2h = R * A_h * P = K * S * A_h * S * K^T, where K is the cutting matrix. We first calculate
        // S * A_h * S and cut out later manually.
        auto sas = sr.multiply(a_h, 2).multiply(sp, 0);

        // std::cout << "a_h[1][1][1][1][1][1] = " << a_h[1][1][1][1][1][1] << std::endl;
        // std::cout << "s[1][1][1][1][1][1] = " << (*s)[1][1][1][1][1][1] << std::endl;
        // std::cout << "sas[1][1][1][1][1][1] = " << sas[1][1][1][1][1][1] << std::endl;

        // Cut stencil from 7x7x7 down to 3x3x3, i.e. copy only selected values to new stencil, skipping ghosts.
        VaryingStencil3x3x3 a_2h(a_h.getDim1() >> 1, a_h.getDim2() >> 1, a_h.getDim3() >> 1, 2, 2, 2);
        // clang-format off
        for (int i = 2, i2 = 1; i < a_2h.getDim1() + 2; i++, i2 += 2)
        for (int j = 2, j2 = 1; j < a_2h.getDim2() + 2; j++, j2 += 2)
        for (int k = 2, k2 = 1; k < a_2h.getDim3() + 2; k++, k2 += 2)
            for (int ii = 0, ii2 = 1; ii < 3; ii++, ii2 += 2)
            for (int jj = 0, jj2 = 1; jj < 3; jj++, jj2 += 2)
            for (int kk = 0, kk2 = 1; kk < 3; kk++, kk2 += 2)
            {
                a_2h[i][j][k][ii][jj][kk] = sas[i2][j2][k2][ii2][jj2][kk2];
            }
        // clang-format on

        a_2h.updateGhosts();

        return a_2h;
    }

    /**
     * @brief Calculates and sets the stencil (i.e. the matrix A) for the current level by applying the
     * Galerkin operator, which is defined as A_2h = R * A_h * P with R being restriction and P being prolongation
     * operators on GPU.
     *
     * @param a_h The stencil of the finer grid.
     * @returns VaryingStencilGpu The stencil to be applied on the coarser grid
     */
    VaryingStencilGpu MultigridEngine::galerkin(VaryingStencilGpu& a_h, cl_program program, cl_command_queue queue, cl_context context)
    {
        // Make sure a_h has two ghosts at each border for periodic bc.
        if (a_h.getGh() != 2)
            throw "galerkin: a_h needs to have 2 ghosts at each border for periodic bc!";

        // Get the full-weight restriction stencil S as 3x3x3 stencil with two additional ghosts at each border.
        // The ghosts are needed in order to respect periodic boundary conditions. One ghost per stencil multiplication.
        auto sr = create3dFullWeightRestrictionStencilGpu(context, queue);
        auto sp = create3dBilinearProlongationStencilGpu(context, queue);

        // A_2h = R * A_h * P = K * S * A_h * S * K^T, where K is the cutting matrix. We first calculate
        // S * A_h * S and cut out later manually.
        auto sas = sr.multiply(a_h, 2, program, queue, context).multiply(sp, 0, program, queue, context);

        // Cut stencil from 7x7x7 down to 3x3x3, i.e. copy only selected values to new stencil, skipping ghosts.
        auto a_2h = sas.cutFromW7ToW3(program, queue, context);
        a_2h.updateGhosts(program, queue);

        return a_2h;
    }
}