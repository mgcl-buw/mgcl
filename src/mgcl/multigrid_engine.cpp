#include "multigrid_engine.hpp"
#include "cuboid.hpp"    // for Cuboid
#include "hypercube.hpp" // for Hypercube6d
#include "level.hpp"     // for Level
#include "mpi_stencil.hpp"
#include "mpi_util.hpp"
#include "opencl_helper.hpp" // for mgclCheckError, OpenCLHelper
#include "problem.hpp"       // for Problem
#include "stencil.hpp"

#include <cstddef> // for size_t, NULL
#include <iostream>
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

        // relax nu1 times
        MultigridEngine::jacobiSeq(level.getV(), level.getF(), level.getR(),
                                   problem.omega, level.h * level.h, problem.nu1, problem.residual_norm, problem.stencilType,
                                   level.stencilFactor, level.stencilValues.get(), false, problem.isPeriodic(),
                                   level.isCalculatedLocally(), problem.getJacobiIterationsPerKernel(), level.getMpiDataPtr());

        // update residual before restriction
        residualSeq(level.getF(), level.getV(), level.getR(), problem.residual_norm, problem.stencilType,
                    level.stencilFactor, level.stencilValues.get(), false, problem.isPeriodic(),
                    level.isCalculatedLocally(), 0, 0, 0, level.getMpiDataPtr());

        // restrict residual as right hand side on coarser grid
        // TODO do not update ghosts of coarse grid before gather (size too small)
        MultigridEngine::restrictSeq(level, levelAbove, level.getR(), levelAbove.getF());

        // If MPI is in use but minGridPoints is reached, gather rhs data to process 0 and perform calculations
        // locally only, until we're reaching the threshold level moving downwards again.
        if (problem.useMpi() && problem.getMpiLevelThreshold() == levelAbove.getNum())
        {
            mpi_util::gather(problem.getMpiComm(), levelAbove.getF());

            // Update ghosts of gathered
            // TODO check Dirichlet
            if (problem.isPeriodic() && problem.mpiRank() == 0)
                MultigridEngine::updateGhostsSeq(levelAbove.getF(), levelAbove.getMpiDataPtr(), problem.isPeriodic(),
                                                 levelAbove.isCalculatedLocally());
        }

        // TODO update ghosts of levelABove.F here when using gh > 1

        // Advance to coarser levels only if
        // 1. not using MPI at all (or on only one process), or
        // 2. coarser level is still calculated distributively, or
        // 3. rank is 0
        if (!problem.useMpi() || !levelAbove.isCalculatedLocally() || problem.mpiRank() == 0)
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
                                           levelAbove.isCalculatedLocally(), problem.getJacobiIterationsPerKernel(),
                                           levelAbove.getMpiDataPtr());

                // printf("post v[0] = %e, f[0] = %e\n", data[level.getNum()+1].getV()[1][1][1], data[level.getNum()+1].getF()[1][1][1]);
            }
        }

        // If MPI is in use but minGridPoints is reached, scatter v data from process 0 to others and continue
        // distributed calulcations.
        if (problem.useMpi() && problem.getMpiLevelThreshold() == levelAbove.getNum())
            mpi_util::scatter_inplace_wgh(problem.getMpiComm(), levelAbove.getV());

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
                                         problem.isPeriodic(), level.isCalculatedLocally(),
                                         problem.getJacobiIterationsPerKernel(), level.getMpiDataPtr());
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

        if (level.getNum() < problem.maxlevel) // if not at highest level
        {
            // reset v to zero for coarser grids (for another possible v-cycle)
            levelAbove.getDVIn().fill(problem.getCommands(), 0.0, false);
        }

        // relax nu1 times
        jacobi(problem, level, problem.nu1, false, problem.getJacobiIterationsPerKernel());

        // update residual before restriction
        residual(problem, level, false);
        // printf("res on level.getNum() %d, upwards: %e\n", level.getNum(), res);

        // restrict to coarser grid
        restrict(level, levelAbove, level.getDR(), levelAbove.getDF());

        // If MPI is in use but minGridPoints is reached, gather rhs data to process 0 and perform calculations
        // locally only, until we're reaching the threshold level moving downwards again.
        if (problem.useMpi() && problem.getMpiLevelThreshold() == levelAbove.getNum())
        {
            mpi_util::gather(problem.getMpiComm(), problem.getCommands(), levelAbove.getDF());

            // Update ghosts of gathered
            // TODO check Dirichlet
            if (problem.isPeriodic() && problem.mpiRank() == 0)
                MultigridEngine::updateGhosts(problem, levelAbove.getDF(),
                                              levelAbove.getMpiDataPtr(), levelAbove.isCalculatedLocally());
        }

        // Advance to coarser levels only if
        // 1. not using MPI at all (or on only one process), or
        // 2. coarser level is still calculated distributively, or
        // 3. rank is 0
        if (!problem.useMpi() || !levelAbove.isCalculatedLocally() || problem.mpiRank() == 0)
        {
            // start next v-cycle iteration if not at highest level
            if (level.getNum() < problem.maxlevel - 1)
                vcycle(problem, levelAbove);
            else
            {
                jacobi(problem, levelAbove, problem.nu1 + problem.nu2, false, problem.getJacobiIterationsPerKernel());
            }
        }

        // If MPI is in use but minGridPoints is reached, scatter v data from process 0 to others and continue
        // distributed calulcations.
        if (problem.useMpi() && problem.getMpiLevelThreshold() == levelAbove.getNum())
            mpi_util::scatter_inplace_wgh(problem.getMpiComm(), problem.getCommands(), levelAbove.getDVIn());

        // prolongate from coarser to finer grid
        // r of this level.getNum() is reused here and should actually be called e
        prolongate(level, levelAbove, level.getDR(), levelAbove.getDVIn());

        // correct error
        correctError(problem, level.getDVIn(), level.getDR(), level.mgh, level.ngh, level.ogh);

        // relax nu2 times
        res = jacobi(problem, level, problem.nu2, !problem.ignoreTol, problem.getJacobiIterationsPerKernel());

        // calculate residual again for the norm TODO in jacobi
        // res = residual(problem, level, 1);
        // printf("res on level.getNum() %d, downwards: %e\n", level.getNum(), res);
        return res;
    }

    /* Starts kernel to correct error, e.g. v = v + e
     * m,n,o is size of ghosted grid */
    int MultigridEngine::correctError(Problem& problem, CuboidGpu& d_v, CuboidGpu& d_r, int mgh, int ngh, int ogh)
    {
        int err;

        // Create the compute kernel from the program
        const char* kernelName = "correct_error";
        cl_kernel kernel = clCreateKernel(problem.openCLHelper.getProgram(), kernelName, &err);
        mgclCheckError(err, "Creating kernel");

        cl_mem dvraw = d_v.getBuffer();
        cl_mem drraw = d_r.getBuffer();

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dvraw);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &drraw);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &problem.ghosts);
        mgclCheckError(err, "Setting kernel arguments");

        // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
        size_t global[3] = {static_cast<size_t>(mgh), static_cast<size_t>(ngh), static_cast<size_t>(ogh)};
        const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(problem.getKernelConfig(), kernelName, 1);
        const size_t local[3] = {
            static_cast<size_t>(mgh > c[0] ? c[0] : mgh),
            static_cast<size_t>(ngh > c[1] ? c[1] : ngh),
            static_cast<size_t>(ogh > c[2] ? c[2] : ogh)};

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
     * @param gh_a2h Amount of ghost cells to apply to the output stencil a_2h. Must be max(2, jacobiItersPerKernel).
     * @param resm Size of resulting stencil's grid. Per default halve of a_h's size.
     * @param resn Size of resulting stencil's grid. Per default halve of a_h's size.
     * @param reso Size of resulting stencil's grid. Per default halve of a_h's size.
     * @returns VaryingStencil The stencil to be applied on the coarser grid
     */
    VaryingStencil MultigridEngine::galerkin(VaryingStencil& a_h, int gh_a2h,
                                             MPILevelData* mpiDataFine, MPILevelData* mpiDataCoarse,
                                             bool periodic, bool forceLocalFine, bool forceLocalCoarse,
                                             bool skipUpdateGhostsCoarse,
                                             int resm, int resn, int reso)
    {
        // TODO respect problem::ghosts maybe

        // Make sure a_h has two ghosts at each border for periodic bc.
        if (a_h.getGhostsM() < 2 || a_h.getGhostsN() < 2 || a_h.getGhostsO() < 2)
            throw "galerkin: a_h needs to have at least 2 ghosts at each border for periodic bc!";

        if (gh_a2h < 2)
            throw "galerkin: gh_a2h must be at least 2.";

        // Get the full-weight restriction stencil S as 3x3x3 stencil with two additional ghosts at each border.
        // The ghosts are needed in order to respect periodic boundary conditions. One ghost per stencil multiplication.
        auto sr = create3dFullWeightRestrictionStencil();
        auto sp = create3dBilinearProlongationStencil();

        // A_2h = R * A_h * P = K * S * A_h * S * K^T, where K is the cutting matrix. We first calculate
        // S * A_h * S and cut out later manually.
        auto sas = sr.multiply(a_h, 2, mpiDataFine, periodic, forceLocalFine)
                       .multiply(sp, 0, mpiDataFine, periodic, forceLocalFine);

        // std::cout << "a_h[1][1][1][1][1][1] = " << a_h[1][1][1][1][1][1] << std::endl;
        // std::cout << "s[1][1][1][1][1][1] = " << (*s)[1][1][1][1][1][1] << std::endl;
        // std::cout << "sas[1][1][1][1][1][1] = " << sas[1][1][1][1][1][1] << std::endl;

        if (resm <= 0)
            resm = a_h.getM() >> 1;
        if (resn <= 0)
            resn = a_h.getN() >> 1;
        if (reso <= 0)
            reso = a_h.getO() >> 1;

        // Cut stencil from 7x7x7 down to 3x3x3, i.e. copy only selected values to new stencil, skipping ghosts.
        VaryingStencil a_2h(resm, resn, reso, 3, gh_a2h, gh_a2h, gh_a2h);
        // clang-format off
        for (int i = gh_a2h, i2 = 1; i < (a_h.getM() >> 1) + gh_a2h; i++, i2 += 2)
        for (int j = gh_a2h, j2 = 1; j < (a_h.getN() >> 1) + gh_a2h; j++, j2 += 2)
        for (int k = gh_a2h, k2 = 1; k < (a_h.getO() >> 1) + gh_a2h; k++, k2 += 2)
            for (int ii = 0, ii2 = 1; ii < 3; ii++, ii2 += 2)
            for (int jj = 0, jj2 = 1; jj < 3; jj++, jj2 += 2)
            for (int kk = 0, kk2 = 1; kk < 3; kk++, kk2 += 2)
            {
                a_2h[ii][jj][kk][i][j][k] = sas[ii2][jj2][kk2][i2][j2][k2];
            }
        // clang-format on

        if (!skipUpdateGhostsCoarse)
            updateGhostsStencilMpi(a_2h, mpiDataCoarse, periodic, forceLocalCoarse);

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
    VaryingStencilGpu MultigridEngine::galerkin(VaryingStencilGpu& a_h, int gh_a2h,
                                                cl_program program, cl_command_queue queue, cl_context context,
                                                MPILevelData* mpiDataFine, MPILevelData* mpiDataCoarse,
                                                bool periodic, bool forceLocalFine, bool forceLocalCoarse,
                                                bool skipUpdateGhostsCoarse,
                                                int resm, int resn, int reso)
    {
        // Make sure a_h has two ghosts at each border for periodic bc.
        if (a_h.getGh() < 2)
            throw "galerkin: a_h needs to have 2 ghosts at each border for periodic bc!";

        if (gh_a2h < 2)
            throw "galerkin: gh_a2h must be at least 2.";

        // Get the full-weight restriction stencil S as 3x3x3 stencil with two additional ghosts at each border.
        // The ghosts are needed in order to respect periodic boundary conditions. One ghost per stencil multiplication.
        auto sr = create3dFullWeightRestrictionStencilGpu(context, queue);
        auto sp = create3dBilinearProlongationStencilGpu(context, queue);

        // A_2h = R * A_h * P = K * S * A_h * S * K^T, where K is the cutting matrix. We first calculate
        // S * A_h * S and cut out later manually.
        auto sas = sr.multiply(a_h, 2, program, queue, context, mpiDataFine, periodic, forceLocalFine)
                       .multiply(sp, 0, program, queue, context, mpiDataFine, periodic, forceLocalFine);

        // Cut stencil from 7x7x7 down to 3x3x3, i.e. copy only selected values to new stencil, skipping ghosts.
        auto a_2h = sas.cutFromW7ToW3(program, queue, context, gh_a2h, resm, resn, reso);

        if (!skipUpdateGhostsCoarse)
            updateGhostsStencilOclMpi(queue, program, a_2h, mpiDataCoarse, periodic, forceLocalCoarse);

        return a_2h;
    }
}