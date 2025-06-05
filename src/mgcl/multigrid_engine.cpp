#include "multigrid_engine.hpp"
#include "blockstencil.hpp"
#include "blockstencil_gpu.hpp"
#include "cuboid.hpp"    // for Cuboid
#include "hypercube.hpp" // for Hypercube6d
#include "level.hpp"     // for Level
#include "matrix.hpp"
#include "mgcl.hpp"
#include "mpi_stencil.hpp"
#include "mpi_util.hpp"
#include "opencl_helper.hpp" // for mgclCheckError, OpenCLHelper
#include "problem.hpp"       // for Problem
#include "profiling_data.hpp"
#include "stencil.hpp"

#include <cassert>
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
                                   level.stencilFactor, level.stencilValues.get(), level.fixedStencil.get(), false, problem.isPeriodic(),
                                   level.isCalculatedLocally(), problem.getJacobiIterationsPerKernel(), level.getMpiDataPtr());

        // update residual before restriction
        residualSeq(level.getF(), level.getV(), level.getR(), problem.residual_norm, problem.stencilType,
                    level.stencilFactor, level.stencilValues.get(), level.fixedStencil.get(), false, problem.isPeriodic(),
                    level.isCalculatedLocally(), 0, 0, 0, level.getMpiDataPtr());

        // level.getV().dumpToFile("v_" + std::to_string(level.getNum()) + ".txt");
        // level.getR().dumpToFile("r_" + std::to_string(level.getNum()) + ".txt");

        // restrict residual as right hand side on coarser grid
        // TODO do not update ghosts of coarse grid before gather (size too small)
        MultigridEngine::restrictSeq(level, levelAbove, level.getR(), levelAbove.getF());

        // levelAbove.getF().dumpToFile("f_" + std::to_string(levelAbove.getNum()) + ".txt");

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
                                           levelAbove.stencilValues.get(), levelAbove.fixedStencil.get(), false, problem.isPeriodic(),
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

        // levelAbove.getV().dumpToFile("v_" + std::to_string(levelAbove.getNum()) + ".txt");
        // level.getR().dumpToFile("r_" + std::to_string(level.getNum()) + ".txt");

        // correct error
        for (int i = problem.ghosts; i < level.m + problem.ghosts; i++)
            for (int j = problem.ghosts; j < level.n + problem.ghosts; j++)
                for (int k = problem.ghosts; k < level.o + problem.ghosts; k++)
                    level.getV()[i][j][k] += level.getR()[i][j][k];

        // relax nu2 times
        res = MultigridEngine::jacobiSeq(level.getV(), level.getF(), level.getR(),
                                         problem.omega, level.h * level.h, problem.nu2, problem.residual_norm, problem.stencilType,
                                         level.stencilFactor, level.stencilValues.get(), level.fixedStencil.get(), !problem.ignoreTol,
                                         problem.isPeriodic(), level.isCalculatedLocally(),
                                         problem.getJacobiIterationsPerKernel(), level.getMpiDataPtr());
        // printf("res on level %d, downwards: %.17e\n", level.getNum(), res);
        return res;
    }

    /* Runs V-cycle recursively and sequentially */
    double MultigridEngine::vcycleSeqBlockstencil(Problem& problem, Level& level)
    {

        auto& levelAbove = problem.getLevelAt(level.num + 1);
        // printf("level.getNum() = %d, m = %3.d\n", level.getNum(), level.m-2);
        // problem.maxlevel = 7;
        double res;

        // reset initial guess of coarser grid
        if (level.getNum() < problem.maxlevel && levelAbove.getVBSPtr() != nullptr)
        {
            levelAbove.getVBS().fill(0);
        }

        // relax nu1 times
        args::JacobiBSSeqArgs jacobi_args1{
            level.getFBS(), level.getVBS(), level.getRBS(),
            problem.residual_norm,
            *level.blockstencil, level.getBlockstencilInvVariant(),
            false, problem.isPeriodic(),
            level.isCalculatedLocally(),
            problem.nu1, problem.jacobi_iterations_per_kernel,
            problem.omega,
            level.getMpiDataPtr()};
        MultigridEngine::jacobiSeq(jacobi_args1);

        // update residual before restriction
        args::ResidualBSSeqArgs residual_args{
            level.getFBS(), level.getVBS(), level.getRBS(),
            problem.residual_norm,
            *level.blockstencil,
            false, problem.isPeriodic(),
            level.isCalculatedLocally(),
            0, 0, 0,
            level.getMpiDataPtr()};
        residualSeq(residual_args);

        // level.getVBS().dumpToFile("vbs_" + std::to_string(level.getNum()) + ".txt");
        // level.getRBS().dumpToFile("rbs_" + std::to_string(level.getNum()) + ".txt");

        // restrict residual as right hand side on coarser grid
        // TODO do not update ghosts of coarse grid before gather (size too small)
        args::RestrictionBSSeqArgs restriction_args{
            level.getRBS(), levelAbove.getFBS(),
            *problem.getRestrictionBlockstencil(),
            problem.isPeriodic(),
            level.isCalculatedLocally(),
            levelAbove.isCalculatedLocally(),
            level.getMpiDataPtr(), levelAbove.getMpiDataPtr()};
        MultigridEngine::restrictSeqBlockstencil(restriction_args);

        // levelAbove.getFBS().dumpToFile("fbs_" + std::to_string(levelAbove.getNum()) + ".txt");

        // If MPI is in use but minGridPoints is reached, gather rhs data to process 0 and perform calculations
        // locally only, until we're reaching the threshold level moving downwards again.
        if (problem.useMpi() && problem.getMpiLevelThreshold() == levelAbove.getNum())
        {
            mpi_util::gather(problem.getMpiComm(), levelAbove.getFBS());

            // Update ghosts of gathered
            // TODO check Dirichlet
            if (problem.isPeriodic() && problem.mpiRank() == 0)
            {
                levelAbove.getFBS().updateGhosts(levelAbove.getMpiDataPtr(), levelAbove.isCalculatedLocally());
            }
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
                MultigridEngine::vcycleSeqBlockstencil(problem, levelAbove);
            else
            {
                args::JacobiBSSeqArgs jacobi_args{
                    levelAbove.getFBS(), levelAbove.getVBS(), levelAbove.getRBS(),
                    problem.residual_norm,
                    *levelAbove.blockstencil, levelAbove.getBlockstencilInvVariant(),
                    false, problem.isPeriodic(),
                    levelAbove.isCalculatedLocally(),
                    problem.nu1 + problem.nu2, problem.jacobi_iterations_per_kernel,
                    problem.omega,
                    levelAbove.getMpiDataPtr()};
                MultigridEngine::jacobiSeq(jacobi_args);

                // printf("post v[0] = %e, f[0] = %e\n", data[level.getNum()+1].getVBS()[1][1][1], data[level.getNum()+1].getFBS()[1][1][1]);
            }
        }

        // If MPI is in use but minGridPoints is reached, scatter v data from process 0 to others and continue
        // distributed calulcations.
        if (problem.useMpi() && problem.getMpiLevelThreshold() == levelAbove.getNum())
            mpi_util::scatter_inplace_wgh(problem.getMpiComm(), levelAbove.getVBS());

        // prolongate from coarser to finer grid
        // r of this level is reused here and should actually be called e
        args::ProlongationBSSeqArgs prolongation_args{
            level.getRBS(), levelAbove.getVBS(),
            *problem.getProlongationBlockstencil(),
            problem.isPeriodic(),
            level.isCalculatedLocally(),
            levelAbove.isCalculatedLocally(),
            level.getMpiDataPtr(), levelAbove.getMpiDataPtr()};
        MultigridEngine::prolongateSeqBlockstencil(prolongation_args);

        // levelAbove.getVBS().dumpToFile("vbs_" + std::to_string(levelAbove.getNum()) + ".txt");
        // level.getRBS().dumpToFile("rbs_" + std::to_string(level.getNum()) + ".txt");

        // correct error
        for (int i = problem.ghosts; i < level.m + problem.ghosts; i++)
            for (int j = problem.ghosts; j < level.n + problem.ghosts; j++)
                for (int k = problem.ghosts; k < level.o + problem.ghosts; k++)
                    for (size_t b = 0; b < level.getVBS().getBlocksize(); b++)
                        level.getVBS()[i][j][k][b] += level.getRBS()[i][j][k][b];

        // relax nu2 times
        args::JacobiBSSeqArgs jacobi_args2{
            level.getFBS(), level.getVBS(), level.getRBS(),
            problem.residual_norm,
            *level.blockstencil, level.getBlockstencilInvVariant(),
            !problem.ignoreTol, problem.isPeriodic(),
            level.isCalculatedLocally(),
            problem.nu2, problem.jacobi_iterations_per_kernel,
            problem.omega,
            level.getMpiDataPtr()};
        res = jacobiSeq(jacobi_args2);

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
            levelAbove.getDVIn().fill(problem.getProgram(), problem.getCommands(), 0.0, false, &problem.getKernelConfig(), problem.getProfilingData());
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
        correctError(level);

        // relax nu2 times
        res = jacobi(problem, level, problem.nu2, !problem.ignoreTol, problem.getJacobiIterationsPerKernel());

        // calculate residual again for the norm TODO in jacobi
        // res = residual(problem, level, 1);
        // printf("res on level.getNum() %d, downwards: %e\n", level.getNum(), res);
        return res;
    }

    /**
     * @brief Starts kernel to correct the error, i.e. v = v + e.
     *
     * @param level
     * @return int
     */
    int MultigridEngine::correctError(Level& level)
    {
        int err;
        auto& problem = *level.problem;

        // Create the compute kernel from the program
        const char* kernelName = "correct_error";
        cl_kernel kernel = clCreateKernel(problem.openCLHelper.getProgram(), kernelName, &err);
        mgclCheckError(err, "Creating kernel");

        cl_mem dvraw = level.getDVIn().getBuffer();
        cl_mem drraw = level.getDR().getBuffer();

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dvraw);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &drraw);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &level.mgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &level.ngh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &level.ogh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &problem.ghosts);
        mgclCheckError(err, "Setting kernel arguments");

        // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
        size_t global[3] = {static_cast<size_t>(level.m), static_cast<size_t>(level.n), static_cast<size_t>(level.o)};
        const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(problem.getKernelConfig(), kernelName, 1);
        const size_t local[3] = {
            static_cast<size_t>(level.m > c[0] ? c[0] : level.m),
            static_cast<size_t>(level.n > c[1] ? c[1] : level.n),
            static_cast<size_t>(level.o > c[2] ? c[2] : level.o)};

        for (int i = 0; i < 3; i++)
            if (global[i] % local[i] != 0)
                global[i] += local[i] - (global[i] % local[i]);

        cl_event ev;

        err = clEnqueueNDRangeKernel(problem.openCLHelper.getCommands(), kernel, 3, NULL, global, local, 0, NULL, &ev);
        mgclCheckError(err, "Enqueueing kernel");

        if (problem.isProfilingEnabled())
        {
            problem.getProfilingData()->addMeasurement(problem.getCommands(), ev, kernelName,
                                                       {global[0], global[1], global[2]},
                                                       {local[0], local[1], local[2]});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        clReleaseKernel(kernel);
        return err;
    }

    /**
     * @brief Calculates and sets the stencil (i.e. the matrix A) for the current level by applying the
     * Galerkin operator, which is defined as A_2h = R * A_h * P with R being restriction and P being prolongation
     * operators. Optimized version that calculated the end result directly, without intermediate stencils.
     * a_h must have up-to-date ghosts.
     *
     * @param a_h The stencil of the finer grid.
     * @param gh_a2h Amount of ghost cells to apply to the output stencil a_2h. Must be max(1, jacobiItersPerKernel).
     * @param resm Size of resulting stencil's grid. Per default halve of a_h's size.
     * @param resn Size of resulting stencil's grid. Per default halve of a_h's size.
     * @param reso Size of resulting stencil's grid. Per default halve of a_h's size.
     * @returns VaryingStencil The stencil to be applied on the coarser grid
     */
    std::unique_ptr<VaryingStencil> MultigridEngine::galerkinOptimized(VaryingStencil& a_h, int gh_a2h,
                                                                       int resm, int resn, int reso)
    {
        // TODO respect problem::ghosts maybe

        // Make sure a_h has two ghosts at each border for periodic bc.
        if (a_h.getGhostsM() < 1 || a_h.getGhostsN() < 1 || a_h.getGhostsO() < 1)
            error("galerkin: a_h needs to have at least 1 ghosts at each border for periodic bc!");

        if (gh_a2h < 1)
            error("galerkin: gh_a2h must be at least 1.");

        // TODO sanity checks on resm, resn, reso?

        // Get the full-weight restriction stencil S as 3x3x3 stencil with two additional ghosts at each border.
        // The ghosts are needed in order to respect periodic boundary conditions. One ghost per stencil multiplication.
        auto r = create3dFullWeightRestrictionStencil();
        auto p = create3dBilinearProlongationStencil();

        auto a_2h = std::make_unique<VaryingStencil>(resm, resn, reso, 3, gh_a2h, gh_a2h, gh_a2h);

        struct Interval
        {
            int start;
            int end;
            std::string toString() { return "[" + std::to_string(start) + "," + std::to_string(end) + "]"; }
        };

        // Returns the intersection of two intervals or [-1,-1] if they don't overlap
        auto intersect = [](Interval a, Interval b) -> Interval
        {
            // Check if intervals overlap
            if (a.start <= b.end && b.start <= a.end)
            {
                // Calculate start and end points of intersection
                int start = (a.start > b.start) ? a.start : b.start;
                int end = (a.end < b.end) ? a.end : b.end;
                return Interval{start, end};
            }
            else
            {
                // Intervals do not overlap
                return Interval{-1, -1};
            }
        };

        struct Point
        {
            int x;
            int y;
            int z;
            std::string toString() { return "(" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + ")"; }
        };

        // Returns the stencil entry indices of the stencil sitting at locationOfStencil that maps to mapsTo.
        // No check is done, if the mapping is possible, i.e. the returned value might be outside of range [0,2].
        // The result is just the difference of the indices plus one, since the stencil entry indices start at 0
        // and not at -1.
        auto stencilEntryThatMapsTo = [](Point locationOfStencil, Point mapsTo) -> Point
        {
            return {mapsTo.x - locationOfStencil.x + 1,
                    mapsTo.y - locationOfStencil.y + 1,
                    mapsTo.z - locationOfStencil.z + 1};
        };

        // Returns the grid point indices that is mapped to by the stencil entry of another point.
        // stencilEntry must be 0-based, hence the substraction by 1.
        auto pointMappedToByStencilEntry = [](Point locationOfStencil, Point stencilEntry) -> Point
        {
            return {locationOfStencil.x + (stencilEntry.x - 1),
                    locationOfStencil.y + (stencilEntry.y - 1),
                    locationOfStencil.z + (stencilEntry.z - 1)};
        };

        // Returns the point on the fine grid that is related to the coarse grid point, respecting ghost cells.
        auto coarseToFine = [](Point p, int ghc, int ghf) -> Point
        {
            return {(p.x - ghc) * 2 + 1 + ghf, (p.y - ghc) * 2 + 1 + ghf, (p.z - ghc) * 2 + 1 + ghf};
        };

        // for each real resulting stencil and stencil entry...
        // (only until a_h >> 1, since on root and on threshold level, a_2h has global size, but only local part can be filled.)
        for (int i = a_2h->getGhostsM(); i < (a_h.getM() >> 1) + a_2h->getGhostsM(); i++)
            for (int j = a_2h->getGhostsN(); j < (a_h.getN() >> 1) + a_2h->getGhostsN(); j++)
                for (int k = a_2h->getGhostsO(); k < (a_h.getO() >> 1) + a_2h->getGhostsO(); k++)
                    for (int ii = 0; ii < a_2h->getWidth(); ii++)
                        for (int jj = 0; jj < a_2h->getWidth(); jj++)
                            for (int kk = 0; kk < a_2h->getWidth(); kk++)
                            {
                                // calculate fine grid point indices
                                Point gp_c = {i, j, k};
                                Point gp_f = coarseToFine(gp_c, a_2h->getGhostsM(), a_h.getGhostsM());
                                Point entry_gpf = coarseToFine(
                                    pointMappedToByStencilEntry(gp_c, {ii, jj, kk}),
                                    a_2h->getGhostsM(), a_h.getGhostsM());

                                // find intersection S_P of neighbouring points for entry_gpf with reach=1 and gp_f with reach=2
                                Interval S_P[3] = {
                                    intersect(Interval{gp_f.x - 2, gp_f.x + 2}, Interval{entry_gpf.x - 1, entry_gpf.x + 1}),
                                    intersect(Interval{gp_f.y - 2, gp_f.y + 2}, Interval{entry_gpf.y - 1, entry_gpf.y + 1}),
                                    intersect(Interval{gp_f.z - 2, gp_f.z + 2}, Interval{entry_gpf.z - 1, entry_gpf.z + 1}),
                                };

                                // Start calc (R*A)*P
                                double res = 0;

                                // for each fine grid point gp_sp in S_P:
                                for (int spi = S_P[0].start; spi <= S_P[0].end; spi++)
                                    for (int spj = S_P[1].start; spj <= S_P[1].end; spj++)
                                        for (int spk = S_P[2].start; spk <= S_P[2].end; spk++)
                                        {
                                            Point gp_sp = {spi, spj, spk};
                                            // tmp_p <- in stencil P located at gp_sp: Find stencil entry entry_p that maps to entry_gpf. Since
                                            // gp_sp is in S_P, it is ensured that the stencil has a stencil entry that maps to entry_gpf.
                                            Point tmp_p_indices = stencilEntryThatMapsTo(gp_sp, entry_gpf);
                                            double tmp_p = p[tmp_p_indices.x][tmp_p_indices.y][tmp_p_indices.z];

                                            // Start calc R*A
                                            // find intersection S_R of neighbouring points for gp_f and gp_sp, both with reach=1
                                            Interval S_R[3] = {
                                                intersect(Interval{gp_f.x - 1, gp_f.x + 1}, Interval{spi - 1, spi + 1}),
                                                intersect(Interval{gp_f.y - 1, gp_f.y + 1}, Interval{spj - 1, spj + 1}),
                                                intersect(Interval{gp_f.z - 1, gp_f.z + 1}, Interval{spk - 1, spk + 1}),
                                            };

                                            double sum = 0;
                                            // for each fine grid point gp_sr in S_R:
                                            for (int sri = S_R[0].start; sri <= S_R[0].end; sri++)
                                                for (int srj = S_R[1].start; srj <= S_R[1].end; srj++)
                                                    for (int srk = S_R[2].start; srk <= S_R[2].end; srk++)
                                                    {
                                                        Point gp_sr = {sri, srj, srk};
                                                        // tmp_r <- in stencil R located at gp_f: Find stencil entry entry_r that maps to gp_sr
                                                        Point tmp_r_indices = stencilEntryThatMapsTo(gp_f, gp_sr);
                                                        double tmp_r = r[tmp_r_indices.x][tmp_r_indices.y][tmp_r_indices.z];
                                                        // tmp_a <- in stencil A located at gp_sr: Find stencil entry that maps to gp_sp
                                                        Point tmp_a_indices = stencilEntryThatMapsTo(gp_sr, gp_sp);

                                                        double tmp_a = a_h[tmp_a_indices.x][tmp_a_indices.y][tmp_a_indices.z][gp_sr.x][gp_sr.y][gp_sr.z];
                                                        // sum <- sum + tmp_r * tmp_a
                                                        sum += tmp_r * tmp_a;
                                                        // End calc R*A
                                                    }

                                            //   res <- res + sum * tmp_p
                                            res += sum * tmp_p;
                                            // End calc (R*A)*P
                                        }

                                // store res in rap
                                (*a_2h)[ii][jj][kk][i][j][k] = res;
                            }

        return a_2h;
    }

    /**
     * @brief Calculates and sets the stencil (i.e. the matrix A) for the current level by applying the
     * Galerkin operator, which is defined as A_2h = R * A_h * P with R being restriction and P being prolongation
     * operators.
     * Handcrafted version to directly calculate A_2h, only valid for 3x3x3 stencils and fixed R and P. This is what
     * Hypre does with rap_type = 0, too.
     * a_h must have up-to-date ghosts. Does not update ghosts of a_2h.
     *
     * @param a_h The stencil of the finer grid.
     * @param gh_a2h Amount of ghost cells to apply to the output stencil a_2h. Must be max(1, jacobiItersPerKernel).
     * @param resm Size of resulting stencil's grid. Per default halve of a_h's size.
     * @param resn Size of resulting stencil's grid. Per default halve of a_h's size.
     * @param reso Size of resulting stencil's grid. Per default halve of a_h's size.
     * @returns VaryingStencil The stencil to be applied on the coarser grid
     */
    std::unique_ptr<VaryingStencil> MultigridEngine::galerkinHandcrafted(VaryingStencil& a_h, int gh_a2h,
                                                                         int resm, int resn, int reso)
    {
        // Make sure a_h has two ghosts at each border for periodic bc.
        if (a_h.getGhostsM() < 1 || a_h.getGhostsN() < 1 || a_h.getGhostsO() < 1)
            error("galerkin: a_h needs to have at least 1 ghosts at each border for periodic bc!");

        if (gh_a2h < 1)
            error("galerkin: gh_a2h must be at least 1.");

        // TODO sanity checks on resm, resn, reso?

        // Get the full-weight restriction stencil S as 3x3x3 stencil with two additional ghosts at each border.
        // The ghosts are needed in order to respect periodic boundary conditions. One ghost per stencil multiplication.
        auto r = create3dFullWeightRestrictionStencil();
        auto p = create3dBilinearProlongationStencil();

        auto a_2h_ptr = std::make_unique<VaryingStencil>(resm, resn, reso, 3, gh_a2h, gh_a2h, gh_a2h);
        auto& a_2h = *a_2h_ptr;

        // for each real coarse grid point and the corresponding fine grid point
        int mc = a_h.getM() >> 1;
        int nc = a_h.getN() >> 1;
        int oc = a_h.getO() >> 1;
        for (int ci = gh_a2h, fi = a_h.getGhostsM() + 1; ci < mc + gh_a2h; ci++, fi += 2)
            for (int cj = gh_a2h, fj = a_h.getGhostsN() + 1; cj < nc + gh_a2h; cj++, fj += 2)
                for (int ck = gh_a2h, fk = a_h.getGhostsO() + 1; ck < oc + gh_a2h; ck++, fk += 2)
                {
                    // front-top-left coefficient
                    a_2h[0][0][0][ci][cj][ck] = ((r[0][0][0] * a_h[0][0][0][fi + -1][fj + -1][fk + -1]) * p[1][1][1] + (r[0][0][0] * a_h[0][0][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][0][0][fi + -1][fj + -1][fk + 0]) * p[1][1][0] + (r[0][0][0] * a_h[0][1][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[0][0][0][fi + -1][fj + 0][fk + -1]) * p[1][0][1] + (r[0][0][0] * a_h[0][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[0][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][0][0][fi + -1][fj + 0][fk + 0]) * p[1][0][0] + (r[0][0][0] * a_h[1][0][0][fi + -1][fj + -1][fk + -1] + r[1][0][0] * a_h[0][0][0][fi + 0][fj + -1][fk + -1]) * p[0][1][1] + (r[0][0][0] * a_h[1][0][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][0][0][fi + -1][fj + -1][fk + 0] + r[1][0][0] * a_h[0][0][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][0][0][fi + 0][fj + -1][fk + 0]) * p[0][1][0] + (r[0][0][0] * a_h[1][1][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[1][0][0][fi + -1][fj + 0][fk + -1] + r[1][0][0] * a_h[0][1][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[0][0][0][fi + 0][fj + 0][fk + -1]) * p[0][0][1] + (r[0][0][0] * a_h[1][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[0][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][0][fi + 0][fj + 0][fk + 0]) * p[0][0][0]);

                    a_2h[0][0][1][ci][cj][ck] = ((r[0][0][0] * a_h[0][0][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][0][0][fi + -1][fj + -1][fk + 0]) * p[1][1][2] + (r[0][0][0] * a_h[0][0][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][0][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][0][0][fi + -1][fj + -1][fk + 1]) * p[1][1][1] + (r[0][0][1] * a_h[0][0][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][0][1][fi + -1][fj + -1][fk + 1]) * p[1][1][0] + (r[0][0][0] * a_h[0][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[0][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][0][0][fi + -1][fj + 0][fk + 0]) * p[1][0][2] + (r[0][0][0] * a_h[0][1][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][1][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][1][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[0][0][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][0][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][0][0][fi + -1][fj + 0][fk + 1]) * p[1][0][1] + (r[0][0][1] * a_h[0][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[0][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][0][1][fi + -1][fj + 0][fk + 1]) * p[1][0][0] + (r[0][0][0] * a_h[1][0][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][0][0][fi + -1][fj + -1][fk + 0] + r[1][0][0] * a_h[0][0][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][0][0][fi + 0][fj + -1][fk + 0]) * p[0][1][2] + (r[0][0][0] * a_h[1][0][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][0][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][0][0][fi + -1][fj + -1][fk + 1] + r[1][0][0] * a_h[0][0][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][0][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][0][0][fi + 0][fj + -1][fk + 1]) * p[0][1][1] + (r[0][0][1] * a_h[1][0][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][0][1][fi + -1][fj + -1][fk + 1] + r[1][0][1] * a_h[0][0][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][0][1][fi + 0][fj + -1][fk + 1]) * p[0][1][0] + (r[0][0][0] * a_h[1][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[0][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][0][fi + 0][fj + 0][fk + 0]) * p[0][0][2] + (r[0][0][0] * a_h[1][1][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[1][0][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][0][fi + -1][fj + 0][fk + 1] + r[1][0][0] * a_h[0][1][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[0][0][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][0][fi + 0][fj + 0][fk + 1]) * p[0][0][1] + (r[0][0][1] * a_h[1][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[0][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][1][fi + 0][fj + 0][fk + 1]) * p[0][0][0]);

                    a_2h[0][0][2][ci][cj][ck] = ((r[0][0][1] * a_h[0][0][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][0][1][fi + -1][fj + -1][fk + 1]) * p[1][1][2] + (r[0][0][2] * a_h[0][0][2][fi + -1][fj + -1][fk + 1]) * p[1][1][1] + (r[0][0][1] * a_h[0][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[0][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][0][1][fi + -1][fj + 0][fk + 1]) * p[1][0][2] + (r[0][0][2] * a_h[0][1][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[0][0][2][fi + -1][fj + 0][fk + 1]) * p[1][0][1] + (r[0][0][1] * a_h[1][0][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][0][1][fi + -1][fj + -1][fk + 1] + r[1][0][1] * a_h[0][0][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][0][1][fi + 0][fj + -1][fk + 1]) * p[0][1][2] + (r[0][0][2] * a_h[1][0][2][fi + -1][fj + -1][fk + 1] + r[1][0][2] * a_h[0][0][2][fi + 0][fj + -1][fk + 1]) * p[0][1][1] + (r[0][0][1] * a_h[1][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[0][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][1][fi + 0][fj + 0][fk + 1]) * p[0][0][2] + (r[0][0][2] * a_h[1][1][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[1][0][2][fi + -1][fj + 0][fk + 1] + r[1][0][2] * a_h[0][1][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[0][0][2][fi + 0][fj + 0][fk + 1]) * p[0][0][1]);

                    a_2h[0][1][0][ci][cj][ck] = ((r[0][0][0] * a_h[0][1][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[0][0][0][fi + -1][fj + 0][fk + -1]) * p[1][2][1] + (r[0][0][0] * a_h[0][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[0][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][0][0][fi + -1][fj + 0][fk + 0]) * p[1][2][0] + (r[0][0][0] * a_h[0][2][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[0][1][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[0][0][0][fi + -1][fj + 1][fk + -1]) * p[1][1][1] + (r[0][0][0] * a_h[0][2][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][2][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[0][1][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][1][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[0][0][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][0][0][fi + -1][fj + 1][fk + 0]) * p[1][1][0] + (r[0][1][0] * a_h[0][2][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[0][1][0][fi + -1][fj + 1][fk + -1]) * p[1][0][1] + (r[0][1][0] * a_h[0][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[0][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][1][0][fi + -1][fj + 1][fk + 0]) * p[1][0][0] + (r[0][0][0] * a_h[1][1][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[1][0][0][fi + -1][fj + 0][fk + -1] + r[1][0][0] * a_h[0][1][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[0][0][0][fi + 0][fj + 0][fk + -1]) * p[0][2][1] + (r[0][0][0] * a_h[1][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[0][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][0][fi + 0][fj + 0][fk + 0]) * p[0][2][0] + (r[0][0][0] * a_h[1][2][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[1][1][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[1][0][0][fi + -1][fj + 1][fk + -1] + r[1][0][0] * a_h[0][2][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[0][1][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[0][0][0][fi + 0][fj + 1][fk + -1]) * p[0][1][1] + (r[0][0][0] * a_h[1][2][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][2][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][1][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][1][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][0][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][0][0][fi + -1][fj + 1][fk + 0] + r[1][0][0] * a_h[0][2][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][2][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][1][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][1][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][0][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][0][0][fi + 0][fj + 1][fk + 0]) * p[0][1][0] + (r[0][1][0] * a_h[1][2][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[1][1][0][fi + -1][fj + 1][fk + -1] + r[1][1][0] * a_h[0][2][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[0][1][0][fi + 0][fj + 1][fk + -1]) * p[0][0][1] + (r[0][1][0] * a_h[1][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[0][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][0][fi + 0][fj + 1][fk + 0]) * p[0][0][0]);

                    a_2h[0][1][1][ci][cj][ck] = ((r[0][0][0] * a_h[0][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[0][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][0][0][fi + -1][fj + 0][fk + 0]) * p[1][2][2] + (r[0][0][0] * a_h[0][1][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][1][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][1][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[0][0][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][0][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][0][0][fi + -1][fj + 0][fk + 1]) * p[1][2][1] + (r[0][0][1] * a_h[0][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[0][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][0][1][fi + -1][fj + 0][fk + 1]) * p[1][2][0] + (r[0][0][0] * a_h[0][2][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][2][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[0][1][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][1][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[0][0][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][0][0][fi + -1][fj + 1][fk + 0]) * p[1][1][2] + (r[0][0][0] * a_h[0][2][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[0][2][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][2][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[0][1][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][1][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][1][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[0][0][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][0][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][0][0][fi + -1][fj + 1][fk + 1]) * p[1][1][1] + (r[0][0][1] * a_h[0][2][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][2][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[0][1][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][1][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[0][0][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][0][1][fi + -1][fj + 1][fk + 1]) * p[1][1][0] + (r[0][1][0] * a_h[0][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[0][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][1][0][fi + -1][fj + 1][fk + 0]) * p[1][0][2] + (r[0][1][0] * a_h[0][2][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][2][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][2][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[0][1][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][1][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][1][0][fi + -1][fj + 1][fk + 1]) * p[1][0][1] + (r[0][1][1] * a_h[0][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[0][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][1][1][fi + -1][fj + 1][fk + 1]) * p[1][0][0] + (r[0][0][0] * a_h[1][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[0][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][0][fi + 0][fj + 0][fk + 0]) * p[0][2][2] + (r[0][0][0] * a_h[1][1][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[1][0][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][0][fi + -1][fj + 0][fk + 1] + r[1][0][0] * a_h[0][1][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[0][0][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][0][fi + 0][fj + 0][fk + 1]) * p[0][2][1] + (r[0][0][1] * a_h[1][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[0][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][1][fi + 0][fj + 0][fk + 1]) * p[0][2][0] + (r[0][0][0] * a_h[1][2][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][2][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][1][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][1][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][0][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][0][0][fi + -1][fj + 1][fk + 0] + r[1][0][0] * a_h[0][2][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][2][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][1][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][1][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][0][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][0][0][fi + 0][fj + 1][fk + 0]) * p[0][1][2] + (r[0][0][0] * a_h[1][2][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][2][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][2][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[1][1][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][1][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][1][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[1][0][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][0][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][0][0][fi + -1][fj + 1][fk + 1] + r[1][0][0] * a_h[0][2][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][2][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][2][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[0][1][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][1][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][1][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[0][0][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][0][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][0][0][fi + 0][fj + 1][fk + 1]) * p[0][1][1] + (r[0][0][1] * a_h[1][2][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][2][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][1][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][1][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][0][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][0][1][fi + -1][fj + 1][fk + 1] + r[1][0][1] * a_h[0][2][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][2][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][1][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][1][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][0][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][0][1][fi + 0][fj + 1][fk + 1]) * p[0][1][0] + (r[0][1][0] * a_h[1][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[0][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][0][fi + 0][fj + 1][fk + 0]) * p[0][0][2] + (r[0][1][0] * a_h[1][2][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[1][1][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][0][fi + -1][fj + 1][fk + 1] + r[1][1][0] * a_h[0][2][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[0][1][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][0][fi + 0][fj + 1][fk + 1]) * p[0][0][1] + (r[0][1][1] * a_h[1][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[0][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][1][fi + 0][fj + 1][fk + 1]) * p[0][0][0]);

                    a_2h[0][1][2][ci][cj][ck] = ((r[0][0][1] * a_h[0][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[0][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][0][1][fi + -1][fj + 0][fk + 1]) * p[1][2][2] + (r[0][0][2] * a_h[0][1][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[0][0][2][fi + -1][fj + 0][fk + 1]) * p[1][2][1] + (r[0][0][1] * a_h[0][2][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[0][2][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[0][1][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][1][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[0][0][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][0][1][fi + -1][fj + 1][fk + 1]) * p[1][1][2] + (r[0][0][2] * a_h[0][2][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[0][1][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[0][0][2][fi + -1][fj + 1][fk + 1]) * p[1][1][1] + (r[0][1][1] * a_h[0][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[0][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][1][1][fi + -1][fj + 1][fk + 1]) * p[1][0][2] + (r[0][1][2] * a_h[0][2][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[0][1][2][fi + -1][fj + 1][fk + 1]) * p[1][0][1] + (r[0][0][1] * a_h[1][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[0][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][1][fi + 0][fj + 0][fk + 1]) * p[0][2][2] + (r[0][0][2] * a_h[1][1][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[1][0][2][fi + -1][fj + 0][fk + 1] + r[1][0][2] * a_h[0][1][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[0][0][2][fi + 0][fj + 0][fk + 1]) * p[0][2][1] + (r[0][0][1] * a_h[1][2][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][2][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][1][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][1][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][0][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][0][1][fi + -1][fj + 1][fk + 1] + r[1][0][1] * a_h[0][2][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][2][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][1][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][1][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][0][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][0][1][fi + 0][fj + 1][fk + 1]) * p[0][1][2] + (r[0][0][2] * a_h[1][2][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[1][1][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[1][0][2][fi + -1][fj + 1][fk + 1] + r[1][0][2] * a_h[0][2][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[0][1][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[0][0][2][fi + 0][fj + 1][fk + 1]) * p[0][1][1] + (r[0][1][1] * a_h[1][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[0][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][1][fi + 0][fj + 1][fk + 1]) * p[0][0][2] + (r[0][1][2] * a_h[1][2][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[1][1][2][fi + -1][fj + 1][fk + 1] + r[1][1][2] * a_h[0][2][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[0][1][2][fi + 0][fj + 1][fk + 1]) * p[0][0][1]);

                    a_2h[0][2][0][ci][cj][ck] = ((r[0][1][0] * a_h[0][2][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[0][1][0][fi + -1][fj + 1][fk + -1]) * p[1][2][1] + (r[0][1][0] * a_h[0][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[0][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][1][0][fi + -1][fj + 1][fk + 0]) * p[1][2][0] + (r[0][2][0] * a_h[0][2][0][fi + -1][fj + 1][fk + -1]) * p[1][1][1] + (r[0][2][0] * a_h[0][2][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][2][0][fi + -1][fj + 1][fk + 0]) * p[1][1][0] + (r[0][1][0] * a_h[1][2][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[1][1][0][fi + -1][fj + 1][fk + -1] + r[1][1][0] * a_h[0][2][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[0][1][0][fi + 0][fj + 1][fk + -1]) * p[0][2][1] + (r[0][1][0] * a_h[1][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[0][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][0][fi + 0][fj + 1][fk + 0]) * p[0][2][0] + (r[0][2][0] * a_h[1][2][0][fi + -1][fj + 1][fk + -1] + r[1][2][0] * a_h[0][2][0][fi + 0][fj + 1][fk + -1]) * p[0][1][1] + (r[0][2][0] * a_h[1][2][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][2][0][fi + -1][fj + 1][fk + 0] + r[1][2][0] * a_h[0][2][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][2][0][fi + 0][fj + 1][fk + 0]) * p[0][1][0]);

                    a_2h[0][2][1][ci][cj][ck] = ((r[0][1][0] * a_h[0][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[0][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][1][0][fi + -1][fj + 1][fk + 0]) * p[1][2][2] + (r[0][1][0] * a_h[0][2][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[0][2][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][2][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[0][1][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][1][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][1][0][fi + -1][fj + 1][fk + 1]) * p[1][2][1] + (r[0][1][1] * a_h[0][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[0][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][1][1][fi + -1][fj + 1][fk + 1]) * p[1][2][0] + (r[0][2][0] * a_h[0][2][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][2][0][fi + -1][fj + 1][fk + 0]) * p[1][1][2] + (r[0][2][0] * a_h[0][2][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[0][2][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][2][0][fi + -1][fj + 1][fk + 1]) * p[1][1][1] + (r[0][2][1] * a_h[0][2][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][2][1][fi + -1][fj + 1][fk + 1]) * p[1][1][0] + (r[0][1][0] * a_h[1][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[0][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][0][fi + 0][fj + 1][fk + 0]) * p[0][2][2] + (r[0][1][0] * a_h[1][2][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[1][1][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][0][fi + -1][fj + 1][fk + 1] + r[1][1][0] * a_h[0][2][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[0][1][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][0][fi + 0][fj + 1][fk + 1]) * p[0][2][1] + (r[0][1][1] * a_h[1][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[0][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][1][fi + 0][fj + 1][fk + 1]) * p[0][2][0] + (r[0][2][0] * a_h[1][2][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][2][0][fi + -1][fj + 1][fk + 0] + r[1][2][0] * a_h[0][2][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][2][0][fi + 0][fj + 1][fk + 0]) * p[0][1][2] + (r[0][2][0] * a_h[1][2][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][2][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][2][0][fi + -1][fj + 1][fk + 1] + r[1][2][0] * a_h[0][2][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][2][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][2][0][fi + 0][fj + 1][fk + 1]) * p[0][1][1] + (r[0][2][1] * a_h[1][2][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][2][1][fi + -1][fj + 1][fk + 1] + r[1][2][1] * a_h[0][2][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][2][1][fi + 0][fj + 1][fk + 1]) * p[0][1][0]);

                    a_2h[0][2][2][ci][cj][ck] = ((r[0][1][1] * a_h[0][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[0][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[0][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][1][1][fi + -1][fj + 1][fk + 1]) * p[1][2][2] + (r[0][1][2] * a_h[0][2][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[0][1][2][fi + -1][fj + 1][fk + 1]) * p[1][2][1] + (r[0][2][1] * a_h[0][2][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[0][2][1][fi + -1][fj + 1][fk + 1]) * p[1][1][2] + (r[0][2][2] * a_h[0][2][2][fi + -1][fj + 1][fk + 1]) * p[1][1][1] + (r[0][1][1] * a_h[1][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[0][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][1][fi + 0][fj + 1][fk + 1]) * p[0][2][2] + (r[0][1][2] * a_h[1][2][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[1][1][2][fi + -1][fj + 1][fk + 1] + r[1][1][2] * a_h[0][2][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[0][1][2][fi + 0][fj + 1][fk + 1]) * p[0][2][1] + (r[0][2][1] * a_h[1][2][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][2][1][fi + -1][fj + 1][fk + 1] + r[1][2][1] * a_h[0][2][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][2][1][fi + 0][fj + 1][fk + 1]) * p[0][1][2] + (r[0][2][2] * a_h[1][2][2][fi + -1][fj + 1][fk + 1] + r[1][2][2] * a_h[0][2][2][fi + 0][fj + 1][fk + 1]) * p[0][1][1]);

                    a_2h[1][0][0][ci][cj][ck] = ((r[0][0][0] * a_h[1][0][0][fi + -1][fj + -1][fk + -1] + r[1][0][0] * a_h[0][0][0][fi + 0][fj + -1][fk + -1]) * p[2][1][1] + (r[0][0][0] * a_h[1][0][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][0][0][fi + -1][fj + -1][fk + 0] + r[1][0][0] * a_h[0][0][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][0][0][fi + 0][fj + -1][fk + 0]) * p[2][1][0] + (r[0][0][0] * a_h[1][1][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[1][0][0][fi + -1][fj + 0][fk + -1] + r[1][0][0] * a_h[0][1][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[0][0][0][fi + 0][fj + 0][fk + -1]) * p[2][0][1] + (r[0][0][0] * a_h[1][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[0][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][0][fi + 0][fj + 0][fk + 0]) * p[2][0][0] + (r[0][0][0] * a_h[2][0][0][fi + -1][fj + -1][fk + -1] + r[1][0][0] * a_h[1][0][0][fi + 0][fj + -1][fk + -1] + r[2][0][0] * a_h[0][0][0][fi + 1][fj + -1][fk + -1]) * p[1][1][1] + (r[0][0][0] * a_h[2][0][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][0][0][fi + -1][fj + -1][fk + 0] + r[1][0][0] * a_h[1][0][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][0][0][fi + 0][fj + -1][fk + 0] + r[2][0][0] * a_h[0][0][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][0][0][fi + 1][fj + -1][fk + 0]) * p[1][1][0] + (r[0][0][0] * a_h[2][1][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[2][0][0][fi + -1][fj + 0][fk + -1] + r[1][0][0] * a_h[1][1][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[1][0][0][fi + 0][fj + 0][fk + -1] + r[2][0][0] * a_h[0][1][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[0][0][0][fi + 1][fj + 0][fk + -1]) * p[1][0][1] + (r[0][0][0] * a_h[2][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[2][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[1][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[1][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[0][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[0][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][0][0][fi + 1][fj + 0][fk + 0]) * p[1][0][0] + (r[1][0][0] * a_h[2][0][0][fi + 0][fj + -1][fk + -1] + r[2][0][0] * a_h[1][0][0][fi + 1][fj + -1][fk + -1]) * p[0][1][1] + (r[1][0][0] * a_h[2][0][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][0][0][fi + 0][fj + -1][fk + 0] + r[2][0][0] * a_h[1][0][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][0][0][fi + 1][fj + -1][fk + 0]) * p[0][1][0] + (r[1][0][0] * a_h[2][1][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[2][0][0][fi + 0][fj + 0][fk + -1] + r[2][0][0] * a_h[1][1][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[1][0][0][fi + 1][fj + 0][fk + -1]) * p[0][0][1] + (r[1][0][0] * a_h[2][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[1][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][0][fi + 1][fj + 0][fk + 0]) * p[0][0][0]);

                    a_2h[1][0][1][ci][cj][ck] = ((r[0][0][0] * a_h[1][0][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][0][0][fi + -1][fj + -1][fk + 0] + r[1][0][0] * a_h[0][0][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][0][0][fi + 0][fj + -1][fk + 0]) * p[2][1][2] + (r[0][0][0] * a_h[1][0][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][0][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][0][0][fi + -1][fj + -1][fk + 1] + r[1][0][0] * a_h[0][0][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][0][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][0][0][fi + 0][fj + -1][fk + 1]) * p[2][1][1] + (r[0][0][1] * a_h[1][0][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][0][1][fi + -1][fj + -1][fk + 1] + r[1][0][1] * a_h[0][0][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][0][1][fi + 0][fj + -1][fk + 1]) * p[2][1][0] + (r[0][0][0] * a_h[1][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[0][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][0][fi + 0][fj + 0][fk + 0]) * p[2][0][2] + (r[0][0][0] * a_h[1][1][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[1][0][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][0][fi + -1][fj + 0][fk + 1] + r[1][0][0] * a_h[0][1][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[0][0][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][0][fi + 0][fj + 0][fk + 1]) * p[2][0][1] + (r[0][0][1] * a_h[1][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[0][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][1][fi + 0][fj + 0][fk + 1]) * p[2][0][0] + (r[0][0][0] * a_h[2][0][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][0][0][fi + -1][fj + -1][fk + 0] + r[1][0][0] * a_h[1][0][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][0][0][fi + 0][fj + -1][fk + 0] + r[2][0][0] * a_h[0][0][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][0][0][fi + 1][fj + -1][fk + 0]) * p[1][1][2] + (r[0][0][0] * a_h[2][0][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][0][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][0][0][fi + -1][fj + -1][fk + 1] + r[1][0][0] * a_h[1][0][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][0][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][0][0][fi + 0][fj + -1][fk + 1] + r[2][0][0] * a_h[0][0][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][0][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][0][0][fi + 1][fj + -1][fk + 1]) * p[1][1][1] + (r[0][0][1] * a_h[2][0][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][0][1][fi + -1][fj + -1][fk + 1] + r[1][0][1] * a_h[1][0][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][0][1][fi + 0][fj + -1][fk + 1] + r[2][0][1] * a_h[0][0][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][0][1][fi + 1][fj + -1][fk + 1]) * p[1][1][0] + (r[0][0][0] * a_h[2][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[2][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[1][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[1][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[0][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[0][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][0][0][fi + 1][fj + 0][fk + 0]) * p[1][0][2] + (r[0][0][0] * a_h[2][1][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][1][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][1][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[2][0][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][0][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][0][0][fi + -1][fj + 0][fk + 1] + r[1][0][0] * a_h[1][1][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][1][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][1][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[1][0][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][0][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][0][0][fi + 0][fj + 0][fk + 1] + r[2][0][0] * a_h[0][1][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][1][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][1][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[0][0][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][0][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][0][0][fi + 1][fj + 0][fk + 1]) * p[1][0][1] + (r[0][0][1] * a_h[2][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[2][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[1][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[1][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[0][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[0][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][0][1][fi + 1][fj + 0][fk + 1]) * p[1][0][0] + (r[1][0][0] * a_h[2][0][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][0][0][fi + 0][fj + -1][fk + 0] + r[2][0][0] * a_h[1][0][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][0][0][fi + 1][fj + -1][fk + 0]) * p[0][1][2] + (r[1][0][0] * a_h[2][0][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][0][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][0][0][fi + 0][fj + -1][fk + 1] + r[2][0][0] * a_h[1][0][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][0][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][0][0][fi + 1][fj + -1][fk + 1]) * p[0][1][1] + (r[1][0][1] * a_h[2][0][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][0][1][fi + 0][fj + -1][fk + 1] + r[2][0][1] * a_h[1][0][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][0][1][fi + 1][fj + -1][fk + 1]) * p[0][1][0] + (r[1][0][0] * a_h[2][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[1][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][0][fi + 1][fj + 0][fk + 0]) * p[0][0][2] + (r[1][0][0] * a_h[2][1][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[2][0][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][0][fi + 0][fj + 0][fk + 1] + r[2][0][0] * a_h[1][1][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[1][0][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][0][fi + 1][fj + 0][fk + 1]) * p[0][0][1] + (r[1][0][1] * a_h[2][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[1][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][1][fi + 1][fj + 0][fk + 1]) * p[0][0][0]);

                    a_2h[1][0][2][ci][cj][ck] = ((r[0][0][1] * a_h[1][0][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][0][1][fi + -1][fj + -1][fk + 1] + r[1][0][1] * a_h[0][0][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][0][1][fi + 0][fj + -1][fk + 1]) * p[2][1][2] + (r[0][0][2] * a_h[1][0][2][fi + -1][fj + -1][fk + 1] + r[1][0][2] * a_h[0][0][2][fi + 0][fj + -1][fk + 1]) * p[2][1][1] + (r[0][0][1] * a_h[1][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[0][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][1][fi + 0][fj + 0][fk + 1]) * p[2][0][2] + (r[0][0][2] * a_h[1][1][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[1][0][2][fi + -1][fj + 0][fk + 1] + r[1][0][2] * a_h[0][1][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[0][0][2][fi + 0][fj + 0][fk + 1]) * p[2][0][1] + (r[0][0][1] * a_h[2][0][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][0][1][fi + -1][fj + -1][fk + 1] + r[1][0][1] * a_h[1][0][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][0][1][fi + 0][fj + -1][fk + 1] + r[2][0][1] * a_h[0][0][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][0][1][fi + 1][fj + -1][fk + 1]) * p[1][1][2] + (r[0][0][2] * a_h[2][0][2][fi + -1][fj + -1][fk + 1] + r[1][0][2] * a_h[1][0][2][fi + 0][fj + -1][fk + 1] + r[2][0][2] * a_h[0][0][2][fi + 1][fj + -1][fk + 1]) * p[1][1][1] + (r[0][0][1] * a_h[2][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[2][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[1][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[1][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[0][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[0][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][0][1][fi + 1][fj + 0][fk + 1]) * p[1][0][2] + (r[0][0][2] * a_h[2][1][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[2][0][2][fi + -1][fj + 0][fk + 1] + r[1][0][2] * a_h[1][1][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[1][0][2][fi + 0][fj + 0][fk + 1] + r[2][0][2] * a_h[0][1][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[0][0][2][fi + 1][fj + 0][fk + 1]) * p[1][0][1] + (r[1][0][1] * a_h[2][0][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][0][1][fi + 0][fj + -1][fk + 1] + r[2][0][1] * a_h[1][0][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][0][1][fi + 1][fj + -1][fk + 1]) * p[0][1][2] + (r[1][0][2] * a_h[2][0][2][fi + 0][fj + -1][fk + 1] + r[2][0][2] * a_h[1][0][2][fi + 1][fj + -1][fk + 1]) * p[0][1][1] + (r[1][0][1] * a_h[2][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[1][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][1][fi + 1][fj + 0][fk + 1]) * p[0][0][2] + (r[1][0][2] * a_h[2][1][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[2][0][2][fi + 0][fj + 0][fk + 1] + r[2][0][2] * a_h[1][1][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[1][0][2][fi + 1][fj + 0][fk + 1]) * p[0][0][1]);

                    a_2h[1][1][0][ci][cj][ck] = ((r[0][0][0] * a_h[1][1][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[1][0][0][fi + -1][fj + 0][fk + -1] + r[1][0][0] * a_h[0][1][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[0][0][0][fi + 0][fj + 0][fk + -1]) * p[2][2][1] + (r[0][0][0] * a_h[1][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[0][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][0][fi + 0][fj + 0][fk + 0]) * p[2][2][0] + (r[0][0][0] * a_h[1][2][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[1][1][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[1][0][0][fi + -1][fj + 1][fk + -1] + r[1][0][0] * a_h[0][2][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[0][1][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[0][0][0][fi + 0][fj + 1][fk + -1]) * p[2][1][1] + (r[0][0][0] * a_h[1][2][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][2][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][1][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][1][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][0][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][0][0][fi + -1][fj + 1][fk + 0] + r[1][0][0] * a_h[0][2][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][2][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][1][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][1][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][0][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][0][0][fi + 0][fj + 1][fk + 0]) * p[2][1][0] + (r[0][1][0] * a_h[1][2][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[1][1][0][fi + -1][fj + 1][fk + -1] + r[1][1][0] * a_h[0][2][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[0][1][0][fi + 0][fj + 1][fk + -1]) * p[2][0][1] + (r[0][1][0] * a_h[1][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[0][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][0][fi + 0][fj + 1][fk + 0]) * p[2][0][0] + (r[0][0][0] * a_h[2][1][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[2][0][0][fi + -1][fj + 0][fk + -1] + r[1][0][0] * a_h[1][1][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[1][0][0][fi + 0][fj + 0][fk + -1] + r[2][0][0] * a_h[0][1][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[0][0][0][fi + 1][fj + 0][fk + -1]) * p[1][2][1] + (r[0][0][0] * a_h[2][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[2][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[1][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[1][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[0][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[0][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][0][0][fi + 1][fj + 0][fk + 0]) * p[1][2][0] + (r[0][0][0] * a_h[2][2][0][fi + -1][fj + -1][fk + -1] + r[0][1][0] * a_h[2][1][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[2][0][0][fi + -1][fj + 1][fk + -1] + r[1][0][0] * a_h[1][2][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[1][1][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[1][0][0][fi + 0][fj + 1][fk + -1] + r[2][0][0] * a_h[0][2][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[0][1][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[0][0][0][fi + 1][fj + 1][fk + -1]) * p[1][1][1] + (r[0][0][0] * a_h[2][2][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][2][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[2][1][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][1][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[2][0][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][0][0][fi + -1][fj + 1][fk + 0] + r[1][0][0] * a_h[1][2][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][2][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[1][1][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][1][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[1][0][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][0][0][fi + 0][fj + 1][fk + 0] + r[2][0][0] * a_h[0][2][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][2][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[0][1][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][1][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[0][0][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][0][0][fi + 1][fj + 1][fk + 0]) * p[1][1][0] + (r[0][1][0] * a_h[2][2][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[2][1][0][fi + -1][fj + 1][fk + -1] + r[1][1][0] * a_h[1][2][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[1][1][0][fi + 0][fj + 1][fk + -1] + r[2][1][0] * a_h[0][2][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[0][1][0][fi + 1][fj + 1][fk + -1]) * p[1][0][1] + (r[0][1][0] * a_h[2][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[2][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[1][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[1][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[0][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[0][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][1][0][fi + 1][fj + 1][fk + 0]) * p[1][0][0] + (r[1][0][0] * a_h[2][1][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[2][0][0][fi + 0][fj + 0][fk + -1] + r[2][0][0] * a_h[1][1][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[1][0][0][fi + 1][fj + 0][fk + -1]) * p[0][2][1] + (r[1][0][0] * a_h[2][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[1][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][0][fi + 1][fj + 0][fk + 0]) * p[0][2][0] + (r[1][0][0] * a_h[2][2][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[2][1][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[2][0][0][fi + 0][fj + 1][fk + -1] + r[2][0][0] * a_h[1][2][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[1][1][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[1][0][0][fi + 1][fj + 1][fk + -1]) * p[0][1][1] + (r[1][0][0] * a_h[2][2][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][2][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][1][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][1][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][0][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][0][0][fi + 0][fj + 1][fk + 0] + r[2][0][0] * a_h[1][2][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][2][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][1][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][1][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][0][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][0][0][fi + 1][fj + 1][fk + 0]) * p[0][1][0] + (r[1][1][0] * a_h[2][2][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[2][1][0][fi + 0][fj + 1][fk + -1] + r[2][1][0] * a_h[1][2][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[1][1][0][fi + 1][fj + 1][fk + -1]) * p[0][0][1] + (r[1][1][0] * a_h[2][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[1][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][0][fi + 1][fj + 1][fk + 0]) * p[0][0][0]);

                    a_2h[1][1][1][ci][cj][ck] = ((r[0][0][0] * a_h[1][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[0][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][0][fi + 0][fj + 0][fk + 0]) * p[2][2][2] + (r[0][0][0] * a_h[1][1][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][1][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[1][0][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][0][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][0][fi + -1][fj + 0][fk + 1] + r[1][0][0] * a_h[0][1][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][1][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[0][0][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][0][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][0][fi + 0][fj + 0][fk + 1]) * p[2][2][1] + (r[0][0][1] * a_h[1][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[0][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][1][fi + 0][fj + 0][fk + 1]) * p[2][2][0] + (r[0][0][0] * a_h[1][2][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][2][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[1][1][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][1][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][0][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][0][0][fi + -1][fj + 1][fk + 0] + r[1][0][0] * a_h[0][2][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][2][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[0][1][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][1][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][0][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][0][0][fi + 0][fj + 1][fk + 0]) * p[2][1][2] + (r[0][0][0] * a_h[1][2][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[1][2][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][2][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[1][1][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][1][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][1][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[1][0][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][0][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][0][0][fi + -1][fj + 1][fk + 1] + r[1][0][0] * a_h[0][2][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[0][2][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][2][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[0][1][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][1][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][1][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[0][0][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][0][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][0][0][fi + 0][fj + 1][fk + 1]) * p[2][1][1] + (r[0][0][1] * a_h[1][2][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][2][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][1][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][1][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][0][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][0][1][fi + -1][fj + 1][fk + 1] + r[1][0][1] * a_h[0][2][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][2][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][1][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][1][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][0][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][0][1][fi + 0][fj + 1][fk + 1]) * p[2][1][0] + (r[0][1][0] * a_h[1][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[0][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][0][fi + 0][fj + 1][fk + 0]) * p[2][0][2] + (r[0][1][0] * a_h[1][2][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[1][1][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][0][fi + -1][fj + 1][fk + 1] + r[1][1][0] * a_h[0][2][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[0][1][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][0][fi + 0][fj + 1][fk + 1]) * p[2][0][1] + (r[0][1][1] * a_h[1][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[0][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][1][fi + 0][fj + 1][fk + 1]) * p[2][0][0] + (r[0][0][0] * a_h[2][1][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][1][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[2][0][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][0][0][fi + -1][fj + 0][fk + 0] + r[1][0][0] * a_h[1][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[1][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[0][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[0][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][0][0][fi + 1][fj + 0][fk + 0]) * p[1][2][2] + (r[0][0][0] * a_h[2][1][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][1][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][1][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[2][0][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][0][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][0][0][fi + -1][fj + 0][fk + 1] + r[1][0][0] * a_h[1][1][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][1][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][1][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[1][0][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][0][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][0][0][fi + 0][fj + 0][fk + 1] + r[2][0][0] * a_h[0][1][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][1][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][1][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[0][0][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][0][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][0][0][fi + 1][fj + 0][fk + 1]) * p[1][2][1] + (r[0][0][1] * a_h[2][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[2][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[1][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[1][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[0][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[0][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][0][1][fi + 1][fj + 0][fk + 1]) * p[1][2][0] + (r[0][0][0] * a_h[2][2][1][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][2][0][fi + -1][fj + -1][fk + 0] + r[0][1][0] * a_h[2][1][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][1][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[2][0][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][0][0][fi + -1][fj + 1][fk + 0] + r[1][0][0] * a_h[1][2][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][2][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[1][1][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][1][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[1][0][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][0][0][fi + 0][fj + 1][fk + 0] + r[2][0][0] * a_h[0][2][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][2][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[0][1][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][1][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[0][0][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][0][0][fi + 1][fj + 1][fk + 0]) * p[1][1][2] + (r[0][0][0] * a_h[2][2][2][fi + -1][fj + -1][fk + -1] + r[0][0][1] * a_h[2][2][1][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][2][0][fi + -1][fj + -1][fk + 1] + r[0][1][0] * a_h[2][1][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][1][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][1][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[2][0][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][0][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][0][0][fi + -1][fj + 1][fk + 1] + r[1][0][0] * a_h[1][2][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[1][2][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][2][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[1][1][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][1][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][1][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[1][0][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][0][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][0][0][fi + 0][fj + 1][fk + 1] + r[2][0][0] * a_h[0][2][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[0][2][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][2][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[0][1][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][1][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][1][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[0][0][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][0][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][0][0][fi + 1][fj + 1][fk + 1]) * p[1][1][1] + (r[0][0][1] * a_h[2][2][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][2][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[2][1][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][1][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[2][0][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][0][1][fi + -1][fj + 1][fk + 1] + r[1][0][1] * a_h[1][2][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][2][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[1][1][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][1][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[1][0][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][0][1][fi + 0][fj + 1][fk + 1] + r[2][0][1] * a_h[0][2][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][2][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[0][1][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][1][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[0][0][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][0][1][fi + 1][fj + 1][fk + 1]) * p[1][1][0] + (r[0][1][0] * a_h[2][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[2][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[1][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[1][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[0][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[0][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][1][0][fi + 1][fj + 1][fk + 0]) * p[1][0][2] + (r[0][1][0] * a_h[2][2][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][2][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][2][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[2][1][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][1][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][1][0][fi + -1][fj + 1][fk + 1] + r[1][1][0] * a_h[1][2][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][2][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][2][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[1][1][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][1][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][1][0][fi + 0][fj + 1][fk + 1] + r[2][1][0] * a_h[0][2][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][2][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][2][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[0][1][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][1][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][1][0][fi + 1][fj + 1][fk + 1]) * p[1][0][1] + (r[0][1][1] * a_h[2][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[2][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[1][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[1][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[0][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[0][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][1][1][fi + 1][fj + 1][fk + 1]) * p[1][0][0] + (r[1][0][0] * a_h[2][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[1][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][0][fi + 1][fj + 0][fk + 0]) * p[0][2][2] + (r[1][0][0] * a_h[2][1][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[2][0][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][0][fi + 0][fj + 0][fk + 1] + r[2][0][0] * a_h[1][1][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[1][0][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][0][fi + 1][fj + 0][fk + 1]) * p[0][2][1] + (r[1][0][1] * a_h[2][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[1][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][1][fi + 1][fj + 0][fk + 1]) * p[0][2][0] + (r[1][0][0] * a_h[2][2][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][2][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][1][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][1][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][0][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][0][0][fi + 0][fj + 1][fk + 0] + r[2][0][0] * a_h[1][2][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][2][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][1][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][1][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][0][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][0][0][fi + 1][fj + 1][fk + 0]) * p[0][1][2] + (r[1][0][0] * a_h[2][2][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][2][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][2][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[2][1][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][1][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][1][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[2][0][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][0][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][0][0][fi + 0][fj + 1][fk + 1] + r[2][0][0] * a_h[1][2][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][2][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][2][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[1][1][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][1][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][1][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[1][0][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][0][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][0][0][fi + 1][fj + 1][fk + 1]) * p[0][1][1] + (r[1][0][1] * a_h[2][2][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][2][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][1][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][1][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][0][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][0][1][fi + 0][fj + 1][fk + 1] + r[2][0][1] * a_h[1][2][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][2][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][1][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][1][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][0][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][0][1][fi + 1][fj + 1][fk + 1]) * p[0][1][0] + (r[1][1][0] * a_h[2][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[1][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][0][fi + 1][fj + 1][fk + 0]) * p[0][0][2] + (r[1][1][0] * a_h[2][2][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[2][1][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][0][fi + 0][fj + 1][fk + 1] + r[2][1][0] * a_h[1][2][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[1][1][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][0][fi + 1][fj + 1][fk + 1]) * p[0][0][1] + (r[1][1][1] * a_h[2][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[1][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][1][fi + 1][fj + 1][fk + 1]) * p[0][0][0]);

                    a_2h[1][1][2][ci][cj][ck] = ((r[0][0][1] * a_h[1][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[0][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][0][1][fi + 0][fj + 0][fk + 1]) * p[2][2][2] + (r[0][0][2] * a_h[1][1][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[1][0][2][fi + -1][fj + 0][fk + 1] + r[1][0][2] * a_h[0][1][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[0][0][2][fi + 0][fj + 0][fk + 1]) * p[2][2][1] + (r[0][0][1] * a_h[1][2][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[1][2][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[1][1][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][1][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][0][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][0][1][fi + -1][fj + 1][fk + 1] + r[1][0][1] * a_h[0][2][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[0][2][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[0][1][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][1][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][0][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][0][1][fi + 0][fj + 1][fk + 1]) * p[2][1][2] + (r[0][0][2] * a_h[1][2][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[1][1][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[1][0][2][fi + -1][fj + 1][fk + 1] + r[1][0][2] * a_h[0][2][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[0][1][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[0][0][2][fi + 0][fj + 1][fk + 1]) * p[2][1][1] + (r[0][1][1] * a_h[1][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[0][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][1][fi + 0][fj + 1][fk + 1]) * p[2][0][2] + (r[0][1][2] * a_h[1][2][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[1][1][2][fi + -1][fj + 1][fk + 1] + r[1][1][2] * a_h[0][2][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[0][1][2][fi + 0][fj + 1][fk + 1]) * p[2][0][1] + (r[0][0][1] * a_h[2][1][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][1][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[2][0][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][0][1][fi + -1][fj + 0][fk + 1] + r[1][0][1] * a_h[1][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[1][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[0][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[0][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][0][1][fi + 1][fj + 0][fk + 1]) * p[1][2][2] + (r[0][0][2] * a_h[2][1][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[2][0][2][fi + -1][fj + 0][fk + 1] + r[1][0][2] * a_h[1][1][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[1][0][2][fi + 0][fj + 0][fk + 1] + r[2][0][2] * a_h[0][1][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[0][0][2][fi + 1][fj + 0][fk + 1]) * p[1][2][1] + (r[0][0][1] * a_h[2][2][2][fi + -1][fj + -1][fk + 0] + r[0][0][2] * a_h[2][2][1][fi + -1][fj + -1][fk + 1] + r[0][1][1] * a_h[2][1][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][1][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[2][0][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][0][1][fi + -1][fj + 1][fk + 1] + r[1][0][1] * a_h[1][2][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[1][2][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[1][1][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][1][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[1][0][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][0][1][fi + 0][fj + 1][fk + 1] + r[2][0][1] * a_h[0][2][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[0][2][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[0][1][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][1][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[0][0][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][0][1][fi + 1][fj + 1][fk + 1]) * p[1][1][2] + (r[0][0][2] * a_h[2][2][2][fi + -1][fj + -1][fk + 1] + r[0][1][2] * a_h[2][1][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[2][0][2][fi + -1][fj + 1][fk + 1] + r[1][0][2] * a_h[1][2][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[1][1][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[1][0][2][fi + 0][fj + 1][fk + 1] + r[2][0][2] * a_h[0][2][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[0][1][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[0][0][2][fi + 1][fj + 1][fk + 1]) * p[1][1][1] + (r[0][1][1] * a_h[2][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[2][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[1][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[1][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[0][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[0][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][1][1][fi + 1][fj + 1][fk + 1]) * p[1][0][2] + (r[0][1][2] * a_h[2][2][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[2][1][2][fi + -1][fj + 1][fk + 1] + r[1][1][2] * a_h[1][2][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[1][1][2][fi + 0][fj + 1][fk + 1] + r[2][1][2] * a_h[0][2][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[0][1][2][fi + 1][fj + 1][fk + 1]) * p[1][0][1] + (r[1][0][1] * a_h[2][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[1][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][1][fi + 1][fj + 0][fk + 1]) * p[0][2][2] + (r[1][0][2] * a_h[2][1][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[2][0][2][fi + 0][fj + 0][fk + 1] + r[2][0][2] * a_h[1][1][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[1][0][2][fi + 1][fj + 0][fk + 1]) * p[0][2][1] + (r[1][0][1] * a_h[2][2][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][2][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][1][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][1][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][0][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][0][1][fi + 0][fj + 1][fk + 1] + r[2][0][1] * a_h[1][2][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][2][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][1][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][1][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][0][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][0][1][fi + 1][fj + 1][fk + 1]) * p[0][1][2] + (r[1][0][2] * a_h[2][2][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[2][1][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[2][0][2][fi + 0][fj + 1][fk + 1] + r[2][0][2] * a_h[1][2][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[1][1][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[1][0][2][fi + 1][fj + 1][fk + 1]) * p[0][1][1] + (r[1][1][1] * a_h[2][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[1][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][1][fi + 1][fj + 1][fk + 1]) * p[0][0][2] + (r[1][1][2] * a_h[2][2][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[2][1][2][fi + 0][fj + 1][fk + 1] + r[2][1][2] * a_h[1][2][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[1][1][2][fi + 1][fj + 1][fk + 1]) * p[0][0][1]);

                    a_2h[1][2][0][ci][cj][ck] = ((r[0][1][0] * a_h[1][2][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[1][1][0][fi + -1][fj + 1][fk + -1] + r[1][1][0] * a_h[0][2][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[0][1][0][fi + 0][fj + 1][fk + -1]) * p[2][2][1] + (r[0][1][0] * a_h[1][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[0][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][0][fi + 0][fj + 1][fk + 0]) * p[2][2][0] + (r[0][2][0] * a_h[1][2][0][fi + -1][fj + 1][fk + -1] + r[1][2][0] * a_h[0][2][0][fi + 0][fj + 1][fk + -1]) * p[2][1][1] + (r[0][2][0] * a_h[1][2][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][2][0][fi + -1][fj + 1][fk + 0] + r[1][2][0] * a_h[0][2][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][2][0][fi + 0][fj + 1][fk + 0]) * p[2][1][0] + (r[0][1][0] * a_h[2][2][0][fi + -1][fj + 0][fk + -1] + r[0][2][0] * a_h[2][1][0][fi + -1][fj + 1][fk + -1] + r[1][1][0] * a_h[1][2][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[1][1][0][fi + 0][fj + 1][fk + -1] + r[2][1][0] * a_h[0][2][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[0][1][0][fi + 1][fj + 1][fk + -1]) * p[1][2][1] + (r[0][1][0] * a_h[2][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[2][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[1][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[1][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[0][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[0][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][1][0][fi + 1][fj + 1][fk + 0]) * p[1][2][0] + (r[0][2][0] * a_h[2][2][0][fi + -1][fj + 1][fk + -1] + r[1][2][0] * a_h[1][2][0][fi + 0][fj + 1][fk + -1] + r[2][2][0] * a_h[0][2][0][fi + 1][fj + 1][fk + -1]) * p[1][1][1] + (r[0][2][0] * a_h[2][2][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][2][0][fi + -1][fj + 1][fk + 0] + r[1][2][0] * a_h[1][2][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][2][0][fi + 0][fj + 1][fk + 0] + r[2][2][0] * a_h[0][2][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][2][0][fi + 1][fj + 1][fk + 0]) * p[1][1][0] + (r[1][1][0] * a_h[2][2][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[2][1][0][fi + 0][fj + 1][fk + -1] + r[2][1][0] * a_h[1][2][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[1][1][0][fi + 1][fj + 1][fk + -1]) * p[0][2][1] + (r[1][1][0] * a_h[2][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[1][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][0][fi + 1][fj + 1][fk + 0]) * p[0][2][0] + (r[1][2][0] * a_h[2][2][0][fi + 0][fj + 1][fk + -1] + r[2][2][0] * a_h[1][2][0][fi + 1][fj + 1][fk + -1]) * p[0][1][1] + (r[1][2][0] * a_h[2][2][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][2][0][fi + 0][fj + 1][fk + 0] + r[2][2][0] * a_h[1][2][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][2][0][fi + 1][fj + 1][fk + 0]) * p[0][1][0]);

                    a_2h[1][2][1][ci][cj][ck] = ((r[0][1][0] * a_h[1][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[1][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[0][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[0][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][0][fi + 0][fj + 1][fk + 0]) * p[2][2][2] + (r[0][1][0] * a_h[1][2][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[1][2][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[1][1][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][1][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][0][fi + -1][fj + 1][fk + 1] + r[1][1][0] * a_h[0][2][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[0][2][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[0][1][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][1][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][0][fi + 0][fj + 1][fk + 1]) * p[2][2][1] + (r[0][1][1] * a_h[1][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[0][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][1][fi + 0][fj + 1][fk + 1]) * p[2][2][0] + (r[0][2][0] * a_h[1][2][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][2][0][fi + -1][fj + 1][fk + 0] + r[1][2][0] * a_h[0][2][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][2][0][fi + 0][fj + 1][fk + 0]) * p[2][1][2] + (r[0][2][0] * a_h[1][2][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[1][2][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][2][0][fi + -1][fj + 1][fk + 1] + r[1][2][0] * a_h[0][2][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[0][2][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][2][0][fi + 0][fj + 1][fk + 1]) * p[2][1][1] + (r[0][2][1] * a_h[1][2][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][2][1][fi + -1][fj + 1][fk + 1] + r[1][2][1] * a_h[0][2][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][2][1][fi + 0][fj + 1][fk + 1]) * p[2][1][0] + (r[0][1][0] * a_h[2][2][1][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][2][0][fi + -1][fj + 0][fk + 0] + r[0][2][0] * a_h[2][1][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][1][0][fi + -1][fj + 1][fk + 0] + r[1][1][0] * a_h[1][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[1][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[0][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[0][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][1][0][fi + 1][fj + 1][fk + 0]) * p[1][2][2] + (r[0][1][0] * a_h[2][2][2][fi + -1][fj + 0][fk + -1] + r[0][1][1] * a_h[2][2][1][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][2][0][fi + -1][fj + 0][fk + 1] + r[0][2][0] * a_h[2][1][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][1][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][1][0][fi + -1][fj + 1][fk + 1] + r[1][1][0] * a_h[1][2][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[1][2][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][2][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[1][1][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][1][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][1][0][fi + 0][fj + 1][fk + 1] + r[2][1][0] * a_h[0][2][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[0][2][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][2][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[0][1][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][1][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][1][0][fi + 1][fj + 1][fk + 1]) * p[1][2][1] + (r[0][1][1] * a_h[2][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[2][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[1][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[1][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[0][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[0][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][1][1][fi + 1][fj + 1][fk + 1]) * p[1][2][0] + (r[0][2][0] * a_h[2][2][1][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][2][0][fi + -1][fj + 1][fk + 0] + r[1][2][0] * a_h[1][2][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][2][0][fi + 0][fj + 1][fk + 0] + r[2][2][0] * a_h[0][2][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][2][0][fi + 1][fj + 1][fk + 0]) * p[1][1][2] + (r[0][2][0] * a_h[2][2][2][fi + -1][fj + 1][fk + -1] + r[0][2][1] * a_h[2][2][1][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][2][0][fi + -1][fj + 1][fk + 1] + r[1][2][0] * a_h[1][2][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[1][2][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][2][0][fi + 0][fj + 1][fk + 1] + r[2][2][0] * a_h[0][2][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[0][2][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][2][0][fi + 1][fj + 1][fk + 1]) * p[1][1][1] + (r[0][2][1] * a_h[2][2][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][2][1][fi + -1][fj + 1][fk + 1] + r[1][2][1] * a_h[1][2][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][2][1][fi + 0][fj + 1][fk + 1] + r[2][2][1] * a_h[0][2][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][2][1][fi + 1][fj + 1][fk + 1]) * p[1][1][0] + (r[1][1][0] * a_h[2][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[1][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][0][fi + 1][fj + 1][fk + 0]) * p[0][2][2] + (r[1][1][0] * a_h[2][2][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[2][1][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][0][fi + 0][fj + 1][fk + 1] + r[2][1][0] * a_h[1][2][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[1][1][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][0][fi + 1][fj + 1][fk + 1]) * p[0][2][1] + (r[1][1][1] * a_h[2][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[1][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][1][fi + 1][fj + 1][fk + 1]) * p[0][2][0] + (r[1][2][0] * a_h[2][2][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][2][0][fi + 0][fj + 1][fk + 0] + r[2][2][0] * a_h[1][2][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][2][0][fi + 1][fj + 1][fk + 0]) * p[0][1][2] + (r[1][2][0] * a_h[2][2][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][2][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][2][0][fi + 0][fj + 1][fk + 1] + r[2][2][0] * a_h[1][2][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][2][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][2][0][fi + 1][fj + 1][fk + 1]) * p[0][1][1] + (r[1][2][1] * a_h[2][2][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][2][1][fi + 0][fj + 1][fk + 1] + r[2][2][1] * a_h[1][2][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][2][1][fi + 1][fj + 1][fk + 1]) * p[0][1][0]);

                    a_2h[1][2][2][ci][cj][ck] = ((r[0][1][1] * a_h[1][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[1][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[1][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[0][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[0][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[0][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][1][1][fi + 0][fj + 1][fk + 1]) * p[2][2][2] + (r[0][1][2] * a_h[1][2][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[1][1][2][fi + -1][fj + 1][fk + 1] + r[1][1][2] * a_h[0][2][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[0][1][2][fi + 0][fj + 1][fk + 1]) * p[2][2][1] + (r[0][2][1] * a_h[1][2][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[1][2][1][fi + -1][fj + 1][fk + 1] + r[1][2][1] * a_h[0][2][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[0][2][1][fi + 0][fj + 1][fk + 1]) * p[2][1][2] + (r[0][2][2] * a_h[1][2][2][fi + -1][fj + 1][fk + 1] + r[1][2][2] * a_h[0][2][2][fi + 0][fj + 1][fk + 1]) * p[2][1][1] + (r[0][1][1] * a_h[2][2][2][fi + -1][fj + 0][fk + 0] + r[0][1][2] * a_h[2][2][1][fi + -1][fj + 0][fk + 1] + r[0][2][1] * a_h[2][1][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][1][1][fi + -1][fj + 1][fk + 1] + r[1][1][1] * a_h[1][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[1][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[1][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[0][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[0][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[0][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][1][1][fi + 1][fj + 1][fk + 1]) * p[1][2][2] + (r[0][1][2] * a_h[2][2][2][fi + -1][fj + 0][fk + 1] + r[0][2][2] * a_h[2][1][2][fi + -1][fj + 1][fk + 1] + r[1][1][2] * a_h[1][2][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[1][1][2][fi + 0][fj + 1][fk + 1] + r[2][1][2] * a_h[0][2][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[0][1][2][fi + 1][fj + 1][fk + 1]) * p[1][2][1] + (r[0][2][1] * a_h[2][2][2][fi + -1][fj + 1][fk + 0] + r[0][2][2] * a_h[2][2][1][fi + -1][fj + 1][fk + 1] + r[1][2][1] * a_h[1][2][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[1][2][1][fi + 0][fj + 1][fk + 1] + r[2][2][1] * a_h[0][2][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[0][2][1][fi + 1][fj + 1][fk + 1]) * p[1][1][2] + (r[0][2][2] * a_h[2][2][2][fi + -1][fj + 1][fk + 1] + r[1][2][2] * a_h[1][2][2][fi + 0][fj + 1][fk + 1] + r[2][2][2] * a_h[0][2][2][fi + 1][fj + 1][fk + 1]) * p[1][1][1] + (r[1][1][1] * a_h[2][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[1][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][1][fi + 1][fj + 1][fk + 1]) * p[0][2][2] + (r[1][1][2] * a_h[2][2][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[2][1][2][fi + 0][fj + 1][fk + 1] + r[2][1][2] * a_h[1][2][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[1][1][2][fi + 1][fj + 1][fk + 1]) * p[0][2][1] + (r[1][2][1] * a_h[2][2][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][2][1][fi + 0][fj + 1][fk + 1] + r[2][2][1] * a_h[1][2][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][2][1][fi + 1][fj + 1][fk + 1]) * p[0][1][2] + (r[1][2][2] * a_h[2][2][2][fi + 0][fj + 1][fk + 1] + r[2][2][2] * a_h[1][2][2][fi + 1][fj + 1][fk + 1]) * p[0][1][1]);

                    a_2h[2][0][0][ci][cj][ck] = ((r[1][0][0] * a_h[2][0][0][fi + 0][fj + -1][fk + -1] + r[2][0][0] * a_h[1][0][0][fi + 1][fj + -1][fk + -1]) * p[2][1][1] + (r[1][0][0] * a_h[2][0][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][0][0][fi + 0][fj + -1][fk + 0] + r[2][0][0] * a_h[1][0][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][0][0][fi + 1][fj + -1][fk + 0]) * p[2][1][0] + (r[1][0][0] * a_h[2][1][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[2][0][0][fi + 0][fj + 0][fk + -1] + r[2][0][0] * a_h[1][1][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[1][0][0][fi + 1][fj + 0][fk + -1]) * p[2][0][1] + (r[1][0][0] * a_h[2][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[1][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][0][fi + 1][fj + 0][fk + 0]) * p[2][0][0] + (r[2][0][0] * a_h[2][0][0][fi + 1][fj + -1][fk + -1]) * p[1][1][1] + (r[2][0][0] * a_h[2][0][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][0][0][fi + 1][fj + -1][fk + 0]) * p[1][1][0] + (r[2][0][0] * a_h[2][1][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[2][0][0][fi + 1][fj + 0][fk + -1]) * p[1][0][1] + (r[2][0][0] * a_h[2][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[2][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][0][0][fi + 1][fj + 0][fk + 0]) * p[1][0][0]);

                    a_2h[2][0][1][ci][cj][ck] = ((r[1][0][0] * a_h[2][0][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][0][0][fi + 0][fj + -1][fk + 0] + r[2][0][0] * a_h[1][0][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][0][0][fi + 1][fj + -1][fk + 0]) * p[2][1][2] + (r[1][0][0] * a_h[2][0][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][0][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][0][0][fi + 0][fj + -1][fk + 1] + r[2][0][0] * a_h[1][0][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][0][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][0][0][fi + 1][fj + -1][fk + 1]) * p[2][1][1] + (r[1][0][1] * a_h[2][0][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][0][1][fi + 0][fj + -1][fk + 1] + r[2][0][1] * a_h[1][0][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][0][1][fi + 1][fj + -1][fk + 1]) * p[2][1][0] + (r[1][0][0] * a_h[2][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[1][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][0][fi + 1][fj + 0][fk + 0]) * p[2][0][2] + (r[1][0][0] * a_h[2][1][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[2][0][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][0][fi + 0][fj + 0][fk + 1] + r[2][0][0] * a_h[1][1][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[1][0][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][0][fi + 1][fj + 0][fk + 1]) * p[2][0][1] + (r[1][0][1] * a_h[2][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[1][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][1][fi + 1][fj + 0][fk + 1]) * p[2][0][0] + (r[2][0][0] * a_h[2][0][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][0][0][fi + 1][fj + -1][fk + 0]) * p[1][1][2] + (r[2][0][0] * a_h[2][0][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][0][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][0][0][fi + 1][fj + -1][fk + 1]) * p[1][1][1] + (r[2][0][1] * a_h[2][0][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][0][1][fi + 1][fj + -1][fk + 1]) * p[1][1][0] + (r[2][0][0] * a_h[2][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[2][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][0][0][fi + 1][fj + 0][fk + 0]) * p[1][0][2] + (r[2][0][0] * a_h[2][1][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][1][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][1][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[2][0][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][0][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][0][0][fi + 1][fj + 0][fk + 1]) * p[1][0][1] + (r[2][0][1] * a_h[2][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[2][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][0][1][fi + 1][fj + 0][fk + 1]) * p[1][0][0]);

                    a_2h[2][0][2][ci][cj][ck] = ((r[1][0][1] * a_h[2][0][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][0][1][fi + 0][fj + -1][fk + 1] + r[2][0][1] * a_h[1][0][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][0][1][fi + 1][fj + -1][fk + 1]) * p[2][1][2] + (r[1][0][2] * a_h[2][0][2][fi + 0][fj + -1][fk + 1] + r[2][0][2] * a_h[1][0][2][fi + 1][fj + -1][fk + 1]) * p[2][1][1] + (r[1][0][1] * a_h[2][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[1][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][1][fi + 1][fj + 0][fk + 1]) * p[2][0][2] + (r[1][0][2] * a_h[2][1][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[2][0][2][fi + 0][fj + 0][fk + 1] + r[2][0][2] * a_h[1][1][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[1][0][2][fi + 1][fj + 0][fk + 1]) * p[2][0][1] + (r[2][0][1] * a_h[2][0][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][0][1][fi + 1][fj + -1][fk + 1]) * p[1][1][2] + (r[2][0][2] * a_h[2][0][2][fi + 1][fj + -1][fk + 1]) * p[1][1][1] + (r[2][0][1] * a_h[2][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[2][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][0][1][fi + 1][fj + 0][fk + 1]) * p[1][0][2] + (r[2][0][2] * a_h[2][1][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[2][0][2][fi + 1][fj + 0][fk + 1]) * p[1][0][1]);

                    a_2h[2][1][0][ci][cj][ck] = ((r[1][0][0] * a_h[2][1][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[2][0][0][fi + 0][fj + 0][fk + -1] + r[2][0][0] * a_h[1][1][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[1][0][0][fi + 1][fj + 0][fk + -1]) * p[2][2][1] + (r[1][0][0] * a_h[2][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[1][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][0][fi + 1][fj + 0][fk + 0]) * p[2][2][0] + (r[1][0][0] * a_h[2][2][0][fi + 0][fj + -1][fk + -1] + r[1][1][0] * a_h[2][1][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[2][0][0][fi + 0][fj + 1][fk + -1] + r[2][0][0] * a_h[1][2][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[1][1][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[1][0][0][fi + 1][fj + 1][fk + -1]) * p[2][1][1] + (r[1][0][0] * a_h[2][2][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][2][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][1][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][1][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][0][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][0][0][fi + 0][fj + 1][fk + 0] + r[2][0][0] * a_h[1][2][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][2][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][1][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][1][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][0][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][0][0][fi + 1][fj + 1][fk + 0]) * p[2][1][0] + (r[1][1][0] * a_h[2][2][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[2][1][0][fi + 0][fj + 1][fk + -1] + r[2][1][0] * a_h[1][2][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[1][1][0][fi + 1][fj + 1][fk + -1]) * p[2][0][1] + (r[1][1][0] * a_h[2][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[1][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][0][fi + 1][fj + 1][fk + 0]) * p[2][0][0] + (r[2][0][0] * a_h[2][1][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[2][0][0][fi + 1][fj + 0][fk + -1]) * p[1][2][1] + (r[2][0][0] * a_h[2][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[2][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][0][0][fi + 1][fj + 0][fk + 0]) * p[1][2][0] + (r[2][0][0] * a_h[2][2][0][fi + 1][fj + -1][fk + -1] + r[2][1][0] * a_h[2][1][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[2][0][0][fi + 1][fj + 1][fk + -1]) * p[1][1][1] + (r[2][0][0] * a_h[2][2][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][2][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[2][1][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][1][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[2][0][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][0][0][fi + 1][fj + 1][fk + 0]) * p[1][1][0] + (r[2][1][0] * a_h[2][2][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[2][1][0][fi + 1][fj + 1][fk + -1]) * p[1][0][1] + (r[2][1][0] * a_h[2][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[2][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][1][0][fi + 1][fj + 1][fk + 0]) * p[1][0][0]);

                    a_2h[2][1][1][ci][cj][ck] = ((r[1][0][0] * a_h[2][1][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][0][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][0][fi + 0][fj + 0][fk + 0] + r[2][0][0] * a_h[1][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][0][fi + 1][fj + 0][fk + 0]) * p[2][2][2] + (r[1][0][0] * a_h[2][1][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][1][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[2][0][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][0][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][0][fi + 0][fj + 0][fk + 1] + r[2][0][0] * a_h[1][1][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][1][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[1][0][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][0][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][0][fi + 1][fj + 0][fk + 1]) * p[2][2][1] + (r[1][0][1] * a_h[2][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[1][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][1][fi + 1][fj + 0][fk + 1]) * p[2][2][0] + (r[1][0][0] * a_h[2][2][1][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][2][0][fi + 0][fj + -1][fk + 0] + r[1][1][0] * a_h[2][1][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][1][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][0][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][0][0][fi + 0][fj + 1][fk + 0] + r[2][0][0] * a_h[1][2][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][2][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[1][1][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][1][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][0][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][0][0][fi + 1][fj + 1][fk + 0]) * p[2][1][2] + (r[1][0][0] * a_h[2][2][2][fi + 0][fj + -1][fk + -1] + r[1][0][1] * a_h[2][2][1][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][2][0][fi + 0][fj + -1][fk + 1] + r[1][1][0] * a_h[2][1][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][1][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][1][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[2][0][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][0][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][0][0][fi + 0][fj + 1][fk + 1] + r[2][0][0] * a_h[1][2][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[1][2][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][2][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[1][1][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][1][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][1][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[1][0][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][0][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][0][0][fi + 1][fj + 1][fk + 1]) * p[2][1][1] + (r[1][0][1] * a_h[2][2][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][2][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][1][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][1][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][0][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][0][1][fi + 0][fj + 1][fk + 1] + r[2][0][1] * a_h[1][2][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][2][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][1][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][1][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][0][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][0][1][fi + 1][fj + 1][fk + 1]) * p[2][1][0] + (r[1][1][0] * a_h[2][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[1][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][0][fi + 1][fj + 1][fk + 0]) * p[2][0][2] + (r[1][1][0] * a_h[2][2][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[2][1][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][0][fi + 0][fj + 1][fk + 1] + r[2][1][0] * a_h[1][2][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[1][1][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][0][fi + 1][fj + 1][fk + 1]) * p[2][0][1] + (r[1][1][1] * a_h[2][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[1][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][1][fi + 1][fj + 1][fk + 1]) * p[2][0][0] + (r[2][0][0] * a_h[2][1][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][1][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[2][0][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][0][0][fi + 1][fj + 0][fk + 0]) * p[1][2][2] + (r[2][0][0] * a_h[2][1][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][1][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][1][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[2][0][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][0][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][0][0][fi + 1][fj + 0][fk + 1]) * p[1][2][1] + (r[2][0][1] * a_h[2][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[2][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][0][1][fi + 1][fj + 0][fk + 1]) * p[1][2][0] + (r[2][0][0] * a_h[2][2][1][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][2][0][fi + 1][fj + -1][fk + 0] + r[2][1][0] * a_h[2][1][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][1][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[2][0][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][0][0][fi + 1][fj + 1][fk + 0]) * p[1][1][2] + (r[2][0][0] * a_h[2][2][2][fi + 1][fj + -1][fk + -1] + r[2][0][1] * a_h[2][2][1][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][2][0][fi + 1][fj + -1][fk + 1] + r[2][1][0] * a_h[2][1][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][1][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][1][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[2][0][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][0][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][0][0][fi + 1][fj + 1][fk + 1]) * p[1][1][1] + (r[2][0][1] * a_h[2][2][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][2][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[2][1][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][1][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[2][0][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][0][1][fi + 1][fj + 1][fk + 1]) * p[1][1][0] + (r[2][1][0] * a_h[2][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[2][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][1][0][fi + 1][fj + 1][fk + 0]) * p[1][0][2] + (r[2][1][0] * a_h[2][2][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][2][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][2][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[2][1][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][1][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][1][0][fi + 1][fj + 1][fk + 1]) * p[1][0][1] + (r[2][1][1] * a_h[2][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[2][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][1][1][fi + 1][fj + 1][fk + 1]) * p[1][0][0]);

                    a_2h[2][1][2][ci][cj][ck] = ((r[1][0][1] * a_h[2][1][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][1][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][0][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][0][1][fi + 0][fj + 0][fk + 1] + r[2][0][1] * a_h[1][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][0][1][fi + 1][fj + 0][fk + 1]) * p[2][2][2] + (r[1][0][2] * a_h[2][1][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[2][0][2][fi + 0][fj + 0][fk + 1] + r[2][0][2] * a_h[1][1][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[1][0][2][fi + 1][fj + 0][fk + 1]) * p[2][2][1] + (r[1][0][1] * a_h[2][2][2][fi + 0][fj + -1][fk + 0] + r[1][0][2] * a_h[2][2][1][fi + 0][fj + -1][fk + 1] + r[1][1][1] * a_h[2][1][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][1][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][0][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][0][1][fi + 0][fj + 1][fk + 1] + r[2][0][1] * a_h[1][2][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[1][2][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[1][1][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][1][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][0][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][0][1][fi + 1][fj + 1][fk + 1]) * p[2][1][2] + (r[1][0][2] * a_h[2][2][2][fi + 0][fj + -1][fk + 1] + r[1][1][2] * a_h[2][1][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[2][0][2][fi + 0][fj + 1][fk + 1] + r[2][0][2] * a_h[1][2][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[1][1][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[1][0][2][fi + 1][fj + 1][fk + 1]) * p[2][1][1] + (r[1][1][1] * a_h[2][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[1][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][1][fi + 1][fj + 1][fk + 1]) * p[2][0][2] + (r[1][1][2] * a_h[2][2][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[2][1][2][fi + 0][fj + 1][fk + 1] + r[2][1][2] * a_h[1][2][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[1][1][2][fi + 1][fj + 1][fk + 1]) * p[2][0][1] + (r[2][0][1] * a_h[2][1][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][1][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[2][0][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][0][1][fi + 1][fj + 0][fk + 1]) * p[1][2][2] + (r[2][0][2] * a_h[2][1][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[2][0][2][fi + 1][fj + 0][fk + 1]) * p[1][2][1] + (r[2][0][1] * a_h[2][2][2][fi + 1][fj + -1][fk + 0] + r[2][0][2] * a_h[2][2][1][fi + 1][fj + -1][fk + 1] + r[2][1][1] * a_h[2][1][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][1][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[2][0][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][0][1][fi + 1][fj + 1][fk + 1]) * p[1][1][2] + (r[2][0][2] * a_h[2][2][2][fi + 1][fj + -1][fk + 1] + r[2][1][2] * a_h[2][1][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[2][0][2][fi + 1][fj + 1][fk + 1]) * p[1][1][1] + (r[2][1][1] * a_h[2][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[2][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][1][1][fi + 1][fj + 1][fk + 1]) * p[1][0][2] + (r[2][1][2] * a_h[2][2][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[2][1][2][fi + 1][fj + 1][fk + 1]) * p[1][0][1]);

                    a_2h[2][2][0][ci][cj][ck] = ((r[1][1][0] * a_h[2][2][0][fi + 0][fj + 0][fk + -1] + r[1][2][0] * a_h[2][1][0][fi + 0][fj + 1][fk + -1] + r[2][1][0] * a_h[1][2][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[1][1][0][fi + 1][fj + 1][fk + -1]) * p[2][2][1] + (r[1][1][0] * a_h[2][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[1][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][0][fi + 1][fj + 1][fk + 0]) * p[2][2][0] + (r[1][2][0] * a_h[2][2][0][fi + 0][fj + 1][fk + -1] + r[2][2][0] * a_h[1][2][0][fi + 1][fj + 1][fk + -1]) * p[2][1][1] + (r[1][2][0] * a_h[2][2][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][2][0][fi + 0][fj + 1][fk + 0] + r[2][2][0] * a_h[1][2][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][2][0][fi + 1][fj + 1][fk + 0]) * p[2][1][0] + (r[2][1][0] * a_h[2][2][0][fi + 1][fj + 0][fk + -1] + r[2][2][0] * a_h[2][1][0][fi + 1][fj + 1][fk + -1]) * p[1][2][1] + (r[2][1][0] * a_h[2][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[2][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][1][0][fi + 1][fj + 1][fk + 0]) * p[1][2][0] + (r[2][2][0] * a_h[2][2][0][fi + 1][fj + 1][fk + -1]) * p[1][1][1] + (r[2][2][0] * a_h[2][2][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][2][0][fi + 1][fj + 1][fk + 0]) * p[1][1][0]);

                    a_2h[2][2][1][ci][cj][ck] = ((r[1][1][0] * a_h[2][2][1][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][0][fi + 0][fj + 0][fk + 0] + r[1][2][0] * a_h[2][1][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][0][fi + 0][fj + 1][fk + 0] + r[2][1][0] * a_h[1][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[1][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][0][fi + 1][fj + 1][fk + 0]) * p[2][2][2] + (r[1][1][0] * a_h[2][2][2][fi + 0][fj + 0][fk + -1] + r[1][1][1] * a_h[2][2][1][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][0][fi + 0][fj + 0][fk + 1] + r[1][2][0] * a_h[2][1][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][1][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][0][fi + 0][fj + 1][fk + 1] + r[2][1][0] * a_h[1][2][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[1][2][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[1][1][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][1][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][0][fi + 1][fj + 1][fk + 1]) * p[2][2][1] + (r[1][1][1] * a_h[2][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[1][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][1][fi + 1][fj + 1][fk + 1]) * p[2][2][0] + (r[1][2][0] * a_h[2][2][1][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][2][0][fi + 0][fj + 1][fk + 0] + r[2][2][0] * a_h[1][2][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][2][0][fi + 1][fj + 1][fk + 0]) * p[2][1][2] + (r[1][2][0] * a_h[2][2][2][fi + 0][fj + 1][fk + -1] + r[1][2][1] * a_h[2][2][1][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][2][0][fi + 0][fj + 1][fk + 1] + r[2][2][0] * a_h[1][2][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[1][2][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][2][0][fi + 1][fj + 1][fk + 1]) * p[2][1][1] + (r[1][2][1] * a_h[2][2][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][2][1][fi + 0][fj + 1][fk + 1] + r[2][2][1] * a_h[1][2][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][2][1][fi + 1][fj + 1][fk + 1]) * p[2][1][0] + (r[2][1][0] * a_h[2][2][1][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][2][0][fi + 1][fj + 0][fk + 0] + r[2][2][0] * a_h[2][1][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][1][0][fi + 1][fj + 1][fk + 0]) * p[1][2][2] + (r[2][1][0] * a_h[2][2][2][fi + 1][fj + 0][fk + -1] + r[2][1][1] * a_h[2][2][1][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][2][0][fi + 1][fj + 0][fk + 1] + r[2][2][0] * a_h[2][1][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][1][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][1][0][fi + 1][fj + 1][fk + 1]) * p[1][2][1] + (r[2][1][1] * a_h[2][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[2][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][1][1][fi + 1][fj + 1][fk + 1]) * p[1][2][0] + (r[2][2][0] * a_h[2][2][1][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][2][0][fi + 1][fj + 1][fk + 0]) * p[1][1][2] + (r[2][2][0] * a_h[2][2][2][fi + 1][fj + 1][fk + -1] + r[2][2][1] * a_h[2][2][1][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][2][0][fi + 1][fj + 1][fk + 1]) * p[1][1][1] + (r[2][2][1] * a_h[2][2][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][2][1][fi + 1][fj + 1][fk + 1]) * p[1][1][0]);

                    a_2h[2][2][2][ci][cj][ck] = ((r[1][1][1] * a_h[2][2][2][fi + 0][fj + 0][fk + 0] + r[1][1][2] * a_h[2][2][1][fi + 0][fj + 0][fk + 1] + r[1][2][1] * a_h[2][1][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][1][1][fi + 0][fj + 1][fk + 1] + r[2][1][1] * a_h[1][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[1][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[1][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][1][1][fi + 1][fj + 1][fk + 1]) * p[2][2][2] + (r[1][1][2] * a_h[2][2][2][fi + 0][fj + 0][fk + 1] + r[1][2][2] * a_h[2][1][2][fi + 0][fj + 1][fk + 1] + r[2][1][2] * a_h[1][2][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[1][1][2][fi + 1][fj + 1][fk + 1]) * p[2][2][1] + (r[1][2][1] * a_h[2][2][2][fi + 0][fj + 1][fk + 0] + r[1][2][2] * a_h[2][2][1][fi + 0][fj + 1][fk + 1] + r[2][2][1] * a_h[1][2][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[1][2][1][fi + 1][fj + 1][fk + 1]) * p[2][1][2] + (r[1][2][2] * a_h[2][2][2][fi + 0][fj + 1][fk + 1] + r[2][2][2] * a_h[1][2][2][fi + 1][fj + 1][fk + 1]) * p[2][1][1] + (r[2][1][1] * a_h[2][2][2][fi + 1][fj + 0][fk + 0] + r[2][1][2] * a_h[2][2][1][fi + 1][fj + 0][fk + 1] + r[2][2][1] * a_h[2][1][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][1][1][fi + 1][fj + 1][fk + 1]) * p[1][2][2] + (r[2][1][2] * a_h[2][2][2][fi + 1][fj + 0][fk + 1] + r[2][2][2] * a_h[2][1][2][fi + 1][fj + 1][fk + 1]) * p[1][2][1] + (r[2][2][1] * a_h[2][2][2][fi + 1][fj + 1][fk + 0] + r[2][2][2] * a_h[2][2][1][fi + 1][fj + 1][fk + 1]) * p[1][1][2] + (r[2][2][2] * a_h[2][2][2][fi + 1][fj + 1][fk + 1]) * p[1][1][1]);
                }

        return a_2h_ptr;
    }

    std::unique_ptr<VaryingStencilGpu> MultigridEngine::galerkinOptimized(VaryingStencilGpu& a_h, int gh_a2h,
                                                                          int resm, int resn, int reso,
                                                                          cl_program program, cl_command_queue queue, cl_context context,
                                                                          conf::KernelConfig* kernelConfig, ProfilingData* pd)
    {
        // Make sure a_h has two ghosts at each border for periodic bc.
        if (a_h.getGh() < 1 || a_h.getGh() < 1 || a_h.getGh() < 1)
            error("galerkin: a_h needs to have at least 1 ghosts at each border for periodic bc!");

        if (gh_a2h < 1)
            error("galerkin: gh_a2h must be at least 1.");

        // TODO sanity checks on resm, resn, reso?

        // Get the full-weight restriction stencil S as 3x3x3 stencil with two additional ghosts at each border.
        // The ghosts are needed in order to respect periodic boundary conditions. One ghost per stencil multiplication.
        auto r = create3dFullWeightRestrictionStencilGpu(context, queue, program);
        auto p = create3dBilinearProlongationStencilGpu(context, queue, program);

        auto a_2h = std::make_unique<VaryingStencilGpu>(resm, resn, reso, 3, gh_a2h, context, queue, program);

        int err;

        // Create the compute kernel from the program
        const char* kernelName = "galerkin";
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        mgclCheckError(err, "Creating kernel");

        cl_mem a_h_raw = a_h.getBuf();
        cl_mem a_2h_raw = a_2h->getBuf();
        cl_mem r_raw = r.getBuf();
        cl_mem p_raw = p.getBuf();

        int mgh_f = a_h.getMgh();
        int ngh_f = a_h.getNgh();
        int ogh_f = a_h.getOgh();
        int m_c_loc = a_h.getM() >> 1;
        int n_c_loc = a_h.getN() >> 1;
        int o_c_loc = a_h.getO() >> 1;
        int gh_f = a_h.getGh();
        int gh_c = a_2h->getGh();

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &a_h_raw);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &a_2h_raw);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &r_raw);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &p_raw);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh_f);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh_f);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh_f);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m_c_loc);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n_c_loc);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o_c_loc);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &resm);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &resn);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &reso);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &gh_f);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &gh_c);
        mgclCheckError(err, "Setting kernel arguments");

        // one work-item per local real coarse grid point.
        size_t global = (a_h.getM() >> 1) * (a_h.getN() >> 1) * (a_h.getO() >> 1) * 27;
        size_t local = 128;

        // Apply kernel config, if available
        if (kernelConfig)
        {
            const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(*kernelConfig, kernelName, global);
            local = static_cast<size_t>(global > c[0] ? c[0] : global);
        }

        // pad global size to fit multiple of local size
        if (global % local != 0)
            global += local - (global % local);

        cl_event ev;

        // enqueue kernel
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, &ev);
        mgclCheckError(err, "Enqueueing update ghosts of varying stencil kernel");

        if (pd != nullptr)
        {
            pd->addMeasurement(queue, ev, kernelName,
                               {global, 0, 0},
                               {local, 1, 1});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        err = clReleaseKernel(kernel);
        mgclCheckError(err, "Releasing update ghosts of varying stencil kernel");

        return a_2h;
    }

    std::unique_ptr<VaryingStencilGpu> MultigridEngine::galerkinHandcrafted(VaryingStencilGpu& a_h, int gh_a2h,
                                                                            int resm, int resn, int reso,
                                                                            cl_program program, cl_command_queue queue, cl_context context,
                                                                            conf::KernelConfig* kernelConfig, ProfilingData* pd)
    {
        // Make sure a_h has two ghosts at each border for periodic bc.
        if (a_h.getGh() < 1 || a_h.getGh() < 1 || a_h.getGh() < 1)
            error("galerkin: a_h needs to have at least 1 ghosts at each border for periodic bc!");

        if (gh_a2h < 1)
            error("galerkin: gh_a2h must be at least 1.");

        // TODO sanity checks on resm, resn, reso?

        // Get the full-weight restriction stencil S as 3x3x3 stencil with two additional ghosts at each border.
        // The ghosts are needed in order to respect periodic boundary conditions. One ghost per stencil multiplication.
        auto r = create3dFullWeightRestrictionStencilGpu(context, queue, program);
        auto p = create3dBilinearProlongationStencilGpu(context, queue, program);

        auto a_2h = std::make_unique<VaryingStencilGpu>(resm, resn, reso, 3, gh_a2h, context, queue, program);

        int err;

        // Create the compute kernel from the program
        const char* kernelName = "galerkin_handcrafted";
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        mgclCheckError(err, "Creating kernel");

        cl_mem a_h_raw = a_h.getBuf();
        cl_mem a_2h_raw = a_2h->getBuf();
        cl_mem r_raw = r.getBuf();
        cl_mem p_raw = p.getBuf();

        int mgh_f = a_h.getMgh();
        int ngh_f = a_h.getNgh();
        int ogh_f = a_h.getOgh();
        int m_c_loc = a_h.getM() >> 1;
        int n_c_loc = a_h.getN() >> 1;
        int o_c_loc = a_h.getO() >> 1;
        int gh_f = a_h.getGh();
        int gh_c = a_2h->getGh();

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &a_h_raw);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &a_2h_raw);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &r_raw);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &p_raw);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh_f);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh_f);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh_f);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m_c_loc);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n_c_loc);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o_c_loc);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &resm);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &resn);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &reso);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &gh_f);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &gh_c);
        mgclCheckError(err, "Setting kernel arguments");

        // one work-item per local real coarse grid point.
        size_t global = (a_h.getM() >> 1) * (a_h.getN() >> 1) * (a_h.getO() >> 1);
        size_t local = 128;

        // Apply kernel config, if available
        if (kernelConfig)
        {
            const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(*kernelConfig, kernelName, global);
            local = static_cast<size_t>(global > c[0] ? c[0] : global);
        }

        // pad global size to fit multiple of local size
        if (global % local != 0)
            global += local - (global % local);

        cl_event ev;

        // enqueue kernel
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, &ev);
        mgclCheckError(err, "Enqueueing update ghosts of varying stencil kernel");

        if (pd != nullptr)
        {
            pd->addMeasurement(queue, ev, kernelName,
                               {global, 0, 0},
                               {local, 1, 1});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        err = clReleaseKernel(kernel);
        mgclCheckError(err, "Releasing update ghosts of varying stencil kernel");

        return a_2h;
    }

    std::unique_ptr<FixedStencil> MultigridEngine::galerkinOptimized(FixedStencil& a_h)
    {
        // Get the full-weight restriction stencil S as 3x3x3 stencil with two additional ghosts at each border.
        // The ghosts are needed in order to respect periodic boundary conditions. One ghost per stencil multiplication.
        auto r = create3dFullWeightRestrictionStencil();
        auto p = create3dBilinearProlongationStencil();

        auto a_2h = std::make_unique<FixedStencil>(3);

        struct Interval
        {
            int start;
            int end;
            std::string toString() { return "[" + std::to_string(start) + "," + std::to_string(end) + "]"; }
        };

        // Returns the intersection of two intervals or [-1,-1] if they don't overlap
        auto intersect = [](Interval a, Interval b) -> Interval
        {
            // Check if intervals overlap
            if (a.start <= b.end && b.start <= a.end)
            {
                // Calculate start and end points of intersection
                int start = (a.start > b.start) ? a.start : b.start;
                int end = (a.end < b.end) ? a.end : b.end;
                return Interval{start, end};
            }
            else
            {
                // Intervals do not overlap
                return Interval{-1, -1};
            }
        };

        struct Point
        {
            int x;
            int y;
            int z;
            std::string toString() { return "(" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + ")"; }
        };

        // Returns the stencil entry indices of the stencil sitting at locationOfStencil that maps to mapsTo.
        // No check is done, if the mapping is possible, i.e. the returned value might be outside of range [0,2].
        // The result is just the difference of the indices plus one, since the stencil entry indices start at 0
        // and not at -1.
        auto stencilEntryThatMapsTo = [](Point locationOfStencil, Point mapsTo) -> Point
        {
            return {mapsTo.x - locationOfStencil.x + 1,
                    mapsTo.y - locationOfStencil.y + 1,
                    mapsTo.z - locationOfStencil.z + 1};
        };

        // Returns the grid point indices that is mapped to by the stencil entry of another point.
        // stencilEntry must be 0-based, hence the substraction by 1.
        auto pointMappedToByStencilEntry = [](Point locationOfStencil, Point stencilEntry) -> Point
        {
            return {locationOfStencil.x + (stencilEntry.x - 1),
                    locationOfStencil.y + (stencilEntry.y - 1),
                    locationOfStencil.z + (stencilEntry.z - 1)};
        };

        // Returns the point on the fine grid that is related to the coarse grid point, respecting ghost cells.
        auto coarseToFine = [](Point p, int ghc, int ghf) -> Point
        {
            return {(p.x - ghc) * 2 + 1 + ghf, (p.y - ghc) * 2 + 1 + ghf, (p.z - ghc) * 2 + 1 + ghf};
        };

        // No need to iterate each grid point, like for VaryingStencil. Just do for one, result will be the same for
        // all grid points.
        // Also, for the same reason, no intersections need to be build. But still we need the correct coefficients.
        // TODO This for sure can be optimized, but for now we just set the coarse grid point to index 1,1,1
        int i = 1;
        int j = 1;
        int k = 1;
        for (int ii = 0; ii < a_2h->getWidth(); ii++)
            for (int jj = 0; jj < a_2h->getWidth(); jj++)
                for (int kk = 0; kk < a_2h->getWidth(); kk++)
                {
                    // calculate fine grid point indices
                    Point gp_c = {i, j, k};
                    Point gp_f = coarseToFine(gp_c, a_2h->getGhostsM(), a_h.getGhostsM());
                    Point entry_gpf = coarseToFine(
                        pointMappedToByStencilEntry(gp_c, {ii, jj, kk}),
                        a_2h->getGhostsM(), a_h.getGhostsM());

                    // find intersection S_P of neighbouring points for entry_gpf with reach=1 and gp_f with reach=2
                    Interval S_P[3] = {
                        intersect(Interval{gp_f.x - 2, gp_f.x + 2}, Interval{entry_gpf.x - 1, entry_gpf.x + 1}),
                        intersect(Interval{gp_f.y - 2, gp_f.y + 2}, Interval{entry_gpf.y - 1, entry_gpf.y + 1}),
                        intersect(Interval{gp_f.z - 2, gp_f.z + 2}, Interval{entry_gpf.z - 1, entry_gpf.z + 1}),
                    };

                    // Start calc (R*A)*P
                    double res = 0;

                    // for each fine grid point gp_sp in S_P:
                    for (int spi = S_P[0].start; spi <= S_P[0].end; spi++)
                        for (int spj = S_P[1].start; spj <= S_P[1].end; spj++)
                            for (int spk = S_P[2].start; spk <= S_P[2].end; spk++)
                            {
                                Point gp_sp = {spi, spj, spk};
                                // tmp_p <- in stencil P located at gp_sp: Find stencil entry entry_p that maps to entry_gpf. Since
                                // gp_sp is in S_P, it is ensured that the stencil has a stencil entry that maps to entry_gpf.
                                Point tmp_p_indices = stencilEntryThatMapsTo(gp_sp, entry_gpf);
                                double tmp_p = p[tmp_p_indices.x][tmp_p_indices.y][tmp_p_indices.z];

                                // Start calc R*A
                                // find intersection S_R of neighbouring points for gp_f and gp_sp, both with reach=1
                                Interval S_R[3] = {
                                    intersect(Interval{gp_f.x - 1, gp_f.x + 1}, Interval{spi - 1, spi + 1}),
                                    intersect(Interval{gp_f.y - 1, gp_f.y + 1}, Interval{spj - 1, spj + 1}),
                                    intersect(Interval{gp_f.z - 1, gp_f.z + 1}, Interval{spk - 1, spk + 1}),
                                };

                                double sum = 0;
                                // for each fine grid point gp_sr in S_R:
                                for (int sri = S_R[0].start; sri <= S_R[0].end; sri++)
                                    for (int srj = S_R[1].start; srj <= S_R[1].end; srj++)
                                        for (int srk = S_R[2].start; srk <= S_R[2].end; srk++)
                                        {
                                            Point gp_sr = {sri, srj, srk};
                                            // tmp_r <- in stencil R located at gp_f: Find stencil entry entry_r that maps to gp_sr
                                            Point tmp_r_indices = stencilEntryThatMapsTo(gp_f, gp_sr);
                                            double tmp_r = r[tmp_r_indices.x][tmp_r_indices.y][tmp_r_indices.z];
                                            // tmp_a <- in stencil A located at gp_sr: Find stencil entry that maps to gp_sp
                                            Point tmp_a_indices = stencilEntryThatMapsTo(gp_sr, gp_sp);

                                            double tmp_a = a_h[tmp_a_indices.x][tmp_a_indices.y][tmp_a_indices.z];
                                            // sum <- sum + tmp_r * tmp_a
                                            sum += tmp_r * tmp_a;
                                            // End calc R*A
                                        }

                                //   res <- res + sum * tmp_p
                                res += sum * tmp_p;
                                // End calc (R*A)*P
                            }

                    // store res in rap
                    (*a_2h)[ii][jj][kk] = res;
                }

        return a_2h;
    }

    std::unique_ptr<FixedStencilGpu> MultigridEngine::galerkinOptimized(FixedStencilGpu& a_h,
                                                                        cl_program program, cl_command_queue queue, cl_context context,
                                                                        conf::KernelConfig* kernelConfig, ProfilingData* pd)
    {
        // Get the full-weight restriction stencil S as 3x3x3 stencil with two additional ghosts at each border.
        // The ghosts are needed in order to respect periodic boundary conditions. One ghost per stencil multiplication.
        auto r = create3dFullWeightRestrictionStencilGpu(context, queue, program);
        auto p = create3dBilinearProlongationStencilGpu(context, queue, program);

        auto a_2h = std::make_unique<FixedStencilGpu>(3, context, queue, program);

        int err;

        // Create the compute kernel from the program
        const char* kernelName = "galerkin_fixed_stencil";
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        mgclCheckError(err, "Creating galerkin_fixed_stencil kernel");

        cl_mem a_h_raw = a_h.getBuf();
        cl_mem a_2h_raw = a_2h->getBuf();
        cl_mem r_raw = r.getBuf();
        cl_mem p_raw = p.getBuf();

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &a_h_raw);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &a_2h_raw);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &r_raw);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &p_raw);
        mgclCheckError(err, "Setting kernel arguments");

        // one work-item per local real coarse grid point.
        size_t global = 27;
        size_t local = 32;

        // Apply kernel config, if available
        if (kernelConfig)
        {
            const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(*kernelConfig, kernelName, global);
            local = static_cast<size_t>(global > c[0] ? c[0] : global);
        }

        // pad global size to fit multiple of local size
        if (global % local != 0)
            global += local - (global % local);

        cl_event ev;

        // enqueue kernel
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, &ev);
        mgclCheckError(err, "Enqueueing galerkin_fixed_stencil kernel");

        if (pd != nullptr)
        {
            pd->addMeasurement(queue, ev, kernelName,
                               {global, 0, 0},
                               {local, 1, 1});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        err = clReleaseKernel(kernel);
        mgclCheckError(err, "Releasing galerkin_fixed_stencil kernel");

        return a_2h;
    }

    /**
     * @brief Calculates and sets the stencil (i.e. the matrix A) for the current level by applying the
     * Galerkin operator, which is defined as A_2h = R * A_h * P with R being restriction and P being prolongation
     * operators. Optimized version that calculated the end result directly, without intermediate stencils.
     * a_h must have up-to-date ghosts.
     *
     * @param a_h The stencil of the finer grid.
     * @param gh_a2h Amount of ghost cells to apply to the output stencil a_2h. Must be max(1, jacobiItersPerKernel).
     * @param resm Size of resulting stencil's grid. Per default halve of a_h's size.
     * @param resn Size of resulting stencil's grid. Per default halve of a_h's size.
     * @param reso Size of resulting stencil's grid. Per default halve of a_h's size.
     * @returns VaryingStencil The stencil to be applied on the coarser grid
     */
    std::unique_ptr<Blockstencil> MultigridEngine::galerkinOptimized(
        Blockstencil& a_h, FixedBlockstencil& r, FixedBlockstencil& p,
        int gh_a2h,
        int resm, int resn, int reso)
    {
        // TODO respect problem::ghosts maybe
        int blocksize = a_h.getBlocksize();

        // Make sure a_h has two ghosts at each border for periodic bc.
        if (a_h.getGhostsM() < 1 || a_h.getGhostsN() < 1 || a_h.getGhostsO() < 1)
            error("galerkin: a_h needs to have at least 1 ghosts at each border for periodic bc!");

        if (gh_a2h < 1)
            error("galerkin: gh_a2h must be at least 1.");

        // TODO sanity checks on resm, resn, reso?

        auto a_2h = std::make_unique<Blockstencil>(resm, resn, reso, 3, blocksize, gh_a2h, gh_a2h, gh_a2h);

        struct Interval
        {
            int start;
            int end;
            std::string toString() { return "[" + std::to_string(start) + "," + std::to_string(end) + "]"; }
        };

        // Returns the intersection of two intervals or [-1,-1] if they don't overlap
        auto intersect = [](Interval a, Interval b) -> Interval
        {
            // Check if intervals overlap
            if (a.start <= b.end && b.start <= a.end)
            {
                // Calculate start and end points of intersection
                int start = (a.start > b.start) ? a.start : b.start;
                int end = (a.end < b.end) ? a.end : b.end;
                return Interval{start, end};
            }
            else
            {
                // Intervals do not overlap
                return Interval{-1, -1};
            }
        };

        struct Point
        {
            int x;
            int y;
            int z;
            std::string toString() { return "(" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + ")"; }
        };

        // Returns the stencil entry indices of the stencil sitting at locationOfStencil that maps to mapsTo.
        // No check is done, if the mapping is possible, i.e. the returned value might be outside of range [0,2].
        // The result is just the difference of the indices plus one, since the stencil entry indices start at 0
        // and not at -1.
        auto stencilEntryThatMapsTo = [](Point locationOfStencil, Point mapsTo) -> Point
        {
            return {mapsTo.x - locationOfStencil.x + 1,
                    mapsTo.y - locationOfStencil.y + 1,
                    mapsTo.z - locationOfStencil.z + 1};
        };

        // Returns the grid point indices that is mapped to by the stencil entry of another point.
        // stencilEntry must be 0-based, hence the substraction by 1.
        auto pointMappedToByStencilEntry = [](Point locationOfStencil, Point stencilEntry) -> Point
        {
            return {locationOfStencil.x + (stencilEntry.x - 1),
                    locationOfStencil.y + (stencilEntry.y - 1),
                    locationOfStencil.z + (stencilEntry.z - 1)};
        };

        // Returns the point on the fine grid that is related to the coarse grid point, respecting ghost cells.
        auto coarseToFine = [](Point p, int ghc, int ghf) -> Point
        {
            return {(p.x - ghc) * 2 + 1 + ghf, (p.y - ghc) * 2 + 1 + ghf, (p.z - ghc) * 2 + 1 + ghf};
        };

        // for each real resulting stencil and stencil entry...
        // (only until a_h >> 1, since on root and on threshold level, a_2h has global size, but only local part can be filled.)
        for (int i = a_2h->getGhostsM(); i < (a_h.getM() >> 1) + a_2h->getGhostsM(); i++)
            for (int j = a_2h->getGhostsN(); j < (a_h.getN() >> 1) + a_2h->getGhostsN(); j++)
                for (int k = a_2h->getGhostsO(); k < (a_h.getO() >> 1) + a_2h->getGhostsO(); k++)
                    for (int ii = 0; ii < a_2h->getWidth(); ii++)
                        for (int jj = 0; jj < a_2h->getWidth(); jj++)
                            for (int kk = 0; kk < a_2h->getWidth(); kk++)
                            {
                                // calculate fine grid point indices
                                Point gp_c = {i, j, k};
                                Point gp_f = coarseToFine(gp_c, a_2h->getGhostsM(), a_h.getGhostsM());
                                Point entry_gpf = coarseToFine(
                                    pointMappedToByStencilEntry(gp_c, {ii, jj, kk}),
                                    a_2h->getGhostsM(), a_h.getGhostsM());

                                // find intersection S_P of neighbouring points for entry_gpf with reach=1 and gp_f with reach=2
                                Interval S_P[3] = {
                                    intersect(Interval{gp_f.x - 2, gp_f.x + 2}, Interval{entry_gpf.x - 1, entry_gpf.x + 1}),
                                    intersect(Interval{gp_f.y - 2, gp_f.y + 2}, Interval{entry_gpf.y - 1, entry_gpf.y + 1}),
                                    intersect(Interval{gp_f.z - 2, gp_f.z + 2}, Interval{entry_gpf.z - 1, entry_gpf.z + 1}),
                                };

                                // Start calc (R*A)*P
                                Matrix res(blocksize, blocksize);

                                // for each fine grid point gp_sp in S_P:
                                for (int spi = S_P[0].start; spi <= S_P[0].end; spi++)
                                    for (int spj = S_P[1].start; spj <= S_P[1].end; spj++)
                                        for (int spk = S_P[2].start; spk <= S_P[2].end; spk++)
                                        {
                                            Point gp_sp = {spi, spj, spk};
                                            // tmp_p <- in stencil P located at gp_sp: Find stencil entry entry_p that maps to entry_gpf. Since
                                            // gp_sp is in S_P, it is ensured that the stencil has a stencil entry that maps to entry_gpf.
                                            Point tmp_p_indices = stencilEntryThatMapsTo(gp_sp, entry_gpf);

                                            // Start calc R*A
                                            // find intersection S_R of neighbouring points for gp_f and gp_sp, both with reach=1
                                            Interval S_R[3] = {
                                                intersect(Interval{gp_f.x - 1, gp_f.x + 1}, Interval{spi - 1, spi + 1}),
                                                intersect(Interval{gp_f.y - 1, gp_f.y + 1}, Interval{spj - 1, spj + 1}),
                                                intersect(Interval{gp_f.z - 1, gp_f.z + 1}, Interval{spk - 1, spk + 1}),
                                            };

                                            // double sum = 0;
                                            Matrix sum(blocksize, blocksize);
                                            // for each fine grid point gp_sr in S_R:
                                            for (int sri = S_R[0].start; sri <= S_R[0].end; sri++)
                                                for (int srj = S_R[1].start; srj <= S_R[1].end; srj++)
                                                    for (int srk = S_R[2].start; srk <= S_R[2].end; srk++)
                                                    {
                                                        Point gp_sr = {sri, srj, srk};
                                                        // tmp_r <- in stencil R located at gp_f: Find stencil entry entry_r that maps to gp_sr
                                                        Point tmp_r_indices = stencilEntryThatMapsTo(gp_f, gp_sr);
                                                        // tmp_a <- in stencil A located at gp_sr: Find stencil entry that maps to gp_sp
                                                        Point tmp_a_indices = stencilEntryThatMapsTo(gp_sr, gp_sp);

                                                        // sum <- sum + tmp_r * tmp_a

                                                        //  calculate r * a first
                                                        //  TODO fuse loops?
                                                        Matrix ra(blocksize, blocksize);
                                                        for (size_t bi = 0; bi < blocksize; bi++)
                                                            for (size_t bj = 0; bj < blocksize; bj++)
                                                                for (size_t bk = 0; bk < blocksize; bk++)
                                                                {
                                                                    ra[bi][bj] += r[bi][bk][tmp_r_indices.x][tmp_r_indices.y][tmp_r_indices.z] * a_h[bk][bj][tmp_a_indices.x][tmp_a_indices.y][tmp_a_indices.z][gp_sr.x][gp_sr.y][gp_sr.z];
                                                                }

                                                        // calc sum += ra
                                                        for (size_t bi = 0; bi < blocksize; bi++)
                                                            for (size_t bj = 0; bj < blocksize; bj++)
                                                            {
                                                                sum[bi][bj] += ra[bi][bj];
                                                            }
                                                        // End calc R*A
                                                    }

                                            //   res <- res + sum * tmp_p
                                            Matrix sum_mult_p(blocksize, blocksize);
                                            for (size_t bi = 0; bi < blocksize; bi++)
                                                for (size_t bj = 0; bj < blocksize; bj++)
                                                    for (size_t bk = 0; bk < blocksize; bk++)
                                                    {
                                                        sum_mult_p[bi][bj] += sum[bi][bk] * p[bk][bj][tmp_p_indices.x][tmp_p_indices.y][tmp_p_indices.z];
                                                    }

                                            for (size_t bi = 0; bi < blocksize; bi++)
                                                for (size_t bj = 0; bj < blocksize; bj++)
                                                {
                                                    res[bi][bj] += sum_mult_p[bi][bj];
                                                }
                                            // End calc (R*A)*P
                                        }

                                // store res in rap
                                for (size_t bi = 0; bi < blocksize; bi++)
                                    for (size_t bj = 0; bj < blocksize; bj++)
                                    {
                                        (*a_2h)[bi][bj][ii][jj][kk][i][j][k] = res[bi][bj];
                                    }
                            }

        return a_2h;
    }

    std::unique_ptr<BlockstencilGpu> MultigridEngine::galerkinOptimized(BlockstencilGpu& a_h, FixedBlockstencilGpu& r, FixedBlockstencilGpu& p,
                                                                        int gh_a2h,
                                                                        int resm, int resn, int reso,
                                                                        cl_program program, cl_command_queue queue, cl_context context,
                                                                        conf::KernelConfig* kernelConfig, ProfilingData* pd)
    {
        // Make sure a_h has two ghosts at each border for periodic bc.
        if (a_h.getGh() < 1 || a_h.getGh() < 1 || a_h.getGh() < 1)
            error("galerkin: a_h needs to have at least 1 ghosts at each border for periodic bc!");

        if (gh_a2h < 1)
            error("galerkin: gh_a2h must be at least 1.");

        // TODO sanity checks on resm, resn, reso?
        size_t blocksize = a_h.getBlocksize();

        auto a_2h = std::make_unique<BlockstencilGpu>(resm, resn, reso, 3, blocksize, gh_a2h, context, queue, program);

        int err;

        // Create the compute kernel from the program
        const char* kernelName = "galerkin_blockstencil";
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        mgclCheckError(err, "Creating kernel");

        cl_mem a_h_raw = a_h.getBuf();
        cl_mem a_2h_raw = a_2h->getBuf();
        cl_mem r_raw = r.getBuf().getBuf();
        cl_mem p_raw = p.getBuf().getBuf();

        int mgh_f = a_h.getMgh();
        int ngh_f = a_h.getNgh();
        int ogh_f = a_h.getOgh();
        int m_c_loc = a_h.getM() >> 1;
        int n_c_loc = a_h.getN() >> 1;
        int o_c_loc = a_h.getO() >> 1;
        int gh_f = a_h.getGh();
        int gh_c = a_2h->getGh();

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &a_h_raw);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &a_2h_raw);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &r_raw);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &p_raw);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh_f);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh_f);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh_f);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m_c_loc);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n_c_loc);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o_c_loc);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &resm);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &resn);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &reso);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &gh_f);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &gh_c);
        mgclCheckError(err, "Setting kernel arguments");

        // one work-item per local real coarse grid point and coefficient.
        size_t global = (a_h.getM() >> 1) * (a_h.getN() >> 1) * (a_h.getO() >> 1) * 27;
        size_t local = 128;

        // Apply kernel config, if available
        if (kernelConfig)
        {
            const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(*kernelConfig, kernelName, global);
            local = static_cast<size_t>(global > c[0] ? c[0] : global);
        }

        // pad global size to fit multiple of local size
        if (global % local != 0)
            global += local - (global % local);

        cl_event ev;

        // enqueue kernel
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, &ev);
        mgclCheckError(err, "Enqueueing galerkin_blockstencil kernel");

        if (pd != nullptr)
        {
            pd->addMeasurement(queue, ev, kernelName,
                               {global, 0, 0},
                               {local, 1, 1});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        err = clReleaseKernel(kernel);
        mgclCheckError(err, "Releasing galerkin_blockstencil kernel");

        return a_2h;
    }

}