/**
 * @date 21.03.2025
 * This file contains code for benchmarking various versions of the residual kernel for blockstencils.
 * A main concern is the layout, i.e.
 *   [mx][my][cx][cy][cz][gpx][gpy][gpz] (block first) vs.
 *   [cx][cy][cz][mx][my][gpx][gpy][gpz] (coeffs first), where
 * mx,my: Matrix indices, cx,cy,cz: coeffs indices, gpx,gpy,gpz: grid point indices
 *
 * To remain flexbility and reproducability, no actual production code is used, but instead the driver functions
 * are copied here.
 *
 */

#include "bench_util.hpp"
#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <CL/cl.h>
#include <catch2/catch_message.hpp>
#include <chrono>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
using namespace std::chrono_literals;

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/problem.hpp"
#include "cli_args.hpp"

namespace mgcl_bench_residual_blockstencil
{
    enum class KernelVersion
    {
        COEFFS_FIRST_V_GP_FIRST,    // [cx][cy][cz][mx][my][gpx][gpy][gpz] for coeffs, [m][gpx][gpy][gpz] for v, f, r
        COEFFS_FIRST_V_BLOCK_FIRST, // [cx][cy][cz][mx][my][gpx][gpy][gpz] for coeffs, [gpx][gpy][gpz][m] for v, f, r
        BLOCK_FIRST_V_GP_FIRST,     // [mx][my][cx][cy][cz][gpx][gpy][gpz] for coeffs, [m][gpx][gpy][gpz] for v, f, r
        BLOCK_FIRST_V_BLOCK_FIRST   // [mx][my][cx][cy][cz][gpx][gpy][gpz] for coeffs, [gpx][gpy][gpz][m] for v, f, r
    };

    using size_t3 = struct
    {
        int x, y, z;
    };

    using ResidualArgsBlockStencil = struct
    {
        bool return_residual;
        int mgh;
        int ngh;
        int ogh;
        double h2;
        int ghosts;
        int blocksize; // Size of matrix in 1 dim, i.e. matrix has blocksize x blocksize elements

        cl_program program;
        cl_command_queue commands;
        size_t3 wgsize;

        mgcl::BufferGpu& c_dVIn;
        mgcl::BufferGpu& c_dF;
        mgcl::BufferGpu& c_dR;
        mgcl::BufferGpu& c_blockStencil;

        mgcl::ProfilingData* pd;

        int moff;
        int noff;
        int ooff;

        KernelVersion kernelVersion;
    };

    /* Calculates the residual using OpenCL.
     * Doesn't creates ocl buffers and doesn't copy data from host to device and vice versa
     * v, f and r must be of size [m][n][o] for periodic boundary condition.
     * m, n and o must be the dimensions of ghosted grid.
     * If return_residual is true, the residual's 2-norm or inf-norm will be read back from device and returned, else -1.
     * It's not really performant to do so because we have to wait for all kernels to complete and
     * reading a buffer to host is slow.
     * v needs to have updated ghost cells if the problem is periodic!
     * moff, noff and ooff can be used to change the size of the grid that the residual shall be calculated for.
     *   Per default only real cells are considered (moff = 0), but with e.g. moff = -1, the first ghost cell border is
     *   considered, too. Analogously, with moff = 1 the outermost set of real cells is ignored. The calculation
     *   of the boundaries is e.g. istart = v.ghosts_m + moff.
     *
     * Some things are excluded. E.g. there is no ghost update after calculating the residual and no residual norm
     * will be returned, as the main interest of this benchmark is the speed of the actual residual kernel.
     */
    double residual(ResidualArgsBlockStencil& args)
    {
        int err;
        double res = 0.0;
        int m = args.mgh - 2 * args.ghosts;
        int n = args.ngh - 2 * args.ghosts;
        int o = args.ogh - 2 * args.ghosts;
        double h2inv = 1.0 / args.h2;

        // check if off is too small (i.e. start < 0)
        // TODO refactor to use GPUCuboid and check against v.getargs.ghosts
        if (args.moff <= -args.ghosts || args.noff <= -args.ghosts || args.ooff <= -args.ghosts)
            error("args.moff, args.noff and args.ooff must not be <= -args.ghosts");

        // check if off is too large (i.e. start > end)
        if (args.moff * 2 >= m || args.noff * 2 >= n || args.ooff * 2 >= o)
            error("2*args.moff, 2*args.noff and 2*args.ooff must not be >= m, n or o");

        // Create the compute kernel from the program
        const char* kernelName;
        // This is only for selecting the kernel to benchmark. Not in productive code.
        if (args.kernelVersion == KernelVersion::COEFFS_FIRST_V_BLOCK_FIRST)
        {
            kernelName = "residual_27point_blockstencil_coeffs_first_v_block_first";
        }
        else if (args.kernelVersion == KernelVersion::COEFFS_FIRST_V_GP_FIRST)
        {
            kernelName = "residual_27point_blockstencil_coeffs_first_v_gp_first";
        }
        else if (args.kernelVersion == KernelVersion::BLOCK_FIRST_V_BLOCK_FIRST)
        {
            kernelName = "residual_27point_blockstencil_block_first_v_block_first";
        }
        else if (args.kernelVersion == KernelVersion::BLOCK_FIRST_V_GP_FIRST)
        {
            kernelName = "residual_27point_blockstencil_block_first_v_gp_first";
        }

        cl_event ev;

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(args.program, kernelName, &err);
        mgcl::mgclCheckError(err, "Creating kernel");

        cl_mem dVIn = args.c_dVIn.getBuf();
        cl_mem dF = args.c_dF.getBuf();
        cl_mem dR = args.c_dR.getBuf();
        cl_mem svbuf = args.c_blockStencil.getBuf();

        // Assumption: sv grid has equal ghost amount as v
        int svmgh = args.mgh;
        int svngh = args.ngh;
        int svogh = args.ogh;
        int svgh = args.ghosts;

        int svGridSize = svmgh * svngh * svogh;
        int svGridSizeBlock = svGridSize * args.blocksize * args.blocksize;

        if (args.kernelVersion == KernelVersion::BLOCK_FIRST_V_BLOCK_FIRST ||
            args.kernelVersion == KernelVersion::BLOCK_FIRST_V_GP_FIRST)
        {
            svGridSizeBlock = 27 * svGridSize;
        }

        // assign kernel arguments
        int pos = 0;
        {
            err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dVIn);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dF);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dR);
            err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &svbuf);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.mgh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.ngh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.ogh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svmgh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svngh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svogh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.ghosts);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svgh);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svGridSize);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svGridSizeBlock);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.blocksize);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.moff);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.noff);
            err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.ooff);
        }

        mgcl::mgclCheckError(err, "Setting residual kernel arguments");

        // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
        size_t global = args.mgh * args.ngh * args.ogh;
        // const auto& c = mgcl::conf::getWorkGroupSizeForKernelAndWiCount(problem.getKernelConfig(), kernelName, global);
        size_t local = args.wgsize.x; // c[0];

        if (global % local != 0)
            global += local - (global % local);

        err = clEnqueueNDRangeKernel(args.commands, kernel, 1, NULL, &global, &local, 0, NULL, &ev);
        mgcl::mgclCheckError(err, "Enqueueing residual kernel");

        if (args.pd)
        {
            args.pd->addMeasurement(args.commands, ev, kernelName,
                                    {global, 0, 0},
                                    {local, 1, 1});
        }
        mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        // mgcl::mgclCheckError(clFinish(args.commands), "clFinish");

        // if (problem.isPeriodic())
        // {
        //     err = MultigridEngine::updateargs.ghosts(problem, level.getDR(), level.getMpiDataPtr(),
        //                                              level.isCalculatedLocally());
        //     mgcl::mgclCheckError(err, "Updating args.ghosts of r");
        // }

        // calculate residual's 2-norm. Square elements on device and sum up on host
        // if (return_residual)
        // {
        //     if (problem.residual_norm == MGCL_L2)
        //     {
        //         // calculate 2-Norm
        //         auto& dRsquares = level.getDRsq();
        //         dRsquares.fill(problem.getProgram(), problem.getCommands(), 0.0, false, &problem.getKernelConfig(), problem.getProfilingData()); // reset to zero

        //         // Create the compute kernel from the program
        //         const char* kernelName = "residual_squared";
        //         cl_kernel kernel_square = clCreateKernel(problem.openCLHelper.getProgram(), kernelName, &err);
        //         mgcl::mgclCheckError(err, "Creating residual squared kernel");

        //         pos = 0;
        //         err = clSetKernelArg(kernel_square, pos, sizeof(cl_mem), &dR);
        //         err |= clSetKernelArg(kernel_square, ++pos, sizeof(cl_mem), &dRsquares);
        //         err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &level.m);
        //         err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &level.n);
        //         err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &level.o);
        //         err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &args.ghosts);
        //         mgcl::mgclCheckError(err, "Setting residual squared kernel arguments");

        //         size_t global = level.getM() * level.getN() * level.getO();
        //         const auto& c_sq = conf::getWorkGroupSizeForKernelAndWiCount(problem.getKernelConfig(), kernelName, global);
        //         size_t local_sq = c_sq[0];

        //         if (global % local_sq != 0)
        //             global += local_sq - (global % local_sq);

        //         cl_event ev;
        //         err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel_square, 1, NULL, &global, &local_sq, 0, NULL, &ev);
        //         mgcl::mgclCheckError(err, "Enqueueing residual squared kernel");

        //         if (problem.isProfilingEnabled())
        //         {
        //             problem.getProfilingData()->addMeasurement(problem.getCommands(), ev, kernelName,
        //                                                        {global, 0, 0},
        //                                                        {local_sq, 1, 1});
        //         }
        //         mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        //         // sum up residual squares
        //         res = sqrt(util::sum(dRsquares, problem.getProgram(), problem.getCommands(), true, &problem.getKernelConfig(), problem.getProfilingData()));

        //         clReleaseKernel(kernel_square);
        //     }
        //     else
        //     {
        //         // calculate Infinity-Norm
        //         res = util::max_abs(level.getDR(), problem.getProgram(), problem.getCommands(), true, &problem.getKernelConfig(), problem.getProfilingData());
        //     }
        // }

        clReleaseKernel(kernel); // TODO maybe clFinish before release?
        return res;
    }

    // Benchs the various residual fixed stencil kernel versions
    TEST_CASE("residualBlockstencilLayouts")
    {
        using std::min;

        if (CLI_ARGS::grids.size() == 0 && (CLI_ARGS::gridsMin.size() == 0 || CLI_ARGS::gridsMax.size() == 0))
            throw "Need to specify at least one local grid size, e.g. using --grids 4,8,16 or --gridsMin 4,4,4 AND --gridsMax 32,32,32";

        // build grids to be tested from CLI args
        std::vector<std::vector<int>> gridsTBT;
        for (auto N : CLI_ARGS::grids)
            gridsTBT.push_back({N, N, N});
        if (CLI_ARGS::gridsMin.size() > 0 && CLI_ARGS::gridsMax.size() > 0)
            for (int m = CLI_ARGS::gridsMin[0]; m <= CLI_ARGS::gridsMax[0]; m *= 2)
                for (int n = CLI_ARGS::gridsMin[1]; n <= CLI_ARGS::gridsMax[1]; n *= 2)
                    for (int o = CLI_ARGS::gridsMin[2]; o <= CLI_ARGS::gridsMax[2]; o *= 2)
                        gridsTBT.push_back({m, n, o});

        std::vector<bench_util::Result> results;

        // Create dummy problem to initialize OpenCL
        auto v_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        auto f_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
        mgcl::Problem p(1, 1, 1, f_dummy, v_dummy);
        p.setKernelFile("blockstencil_kernels.cl");
        if (CLI_ARGS::useBinaryFile)
        {
            p.setBinaryFile("blockstencil_kernels.bin");
        }
        p.setUseOpencl(true);
        p.setDeviceType(CL_DEVICE_TYPE_GPU);
        p.setProfilingEnabled(CLI_ARGS::enableKernelProfiling);
        p.init();

        int ghosts = 1;
        int blocksize = 3;
        int blocksize2 = blocksize * blocksize;

        for (auto gr : gridsTBT)
        {
            int m = gr[0];
            int n = gr[1];
            int o = gr[2];
            int mgh = m + 2 * ghosts;
            int ngh = n + 2 * ghosts;
            int ogh = o + 2 * ghosts;

            // TODO
            std::vector<double> vin(mgh * ngh * ogh * blocksize);
            std::vector<double> f(mgh * ngh * ogh * blocksize);
            std::vector<double> bs(mgh * ngh * ogh * blocksize2 * 27);
            for (int i = 0; i < vin.size(); i++)
            {
                vin[i] = i;
                f[i] = i;
            }
            for (int i = 0; i < bs.size(); i++)
            {
                bs[i] = 0.1 * (double)i;
            }
            mgcl::BufferGpu d_vin(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, vin);
            mgcl::BufferGpu d_f(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, f);
            mgcl::BufferGpu d_r(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, f); // careful, init with f
            mgcl::BufferGpu d_bs(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, bs);

            d_r.fill(p.getProgram(), p.getCommands(), 0.0, true, nullptr, nullptr);

            // auto v_in = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            // auto f_in = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            // auto r_in = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            // // v_in->fill1dIndex(true);
            // // f_in->fill1dIndex(true);
            // v_in->fillRandom();
            // f_in->fillRandom();

            // mgcl::CuboidGpu c_dVIn(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, *v_in);
            // mgcl::CuboidGpu c_dF(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, *f_in);
            // mgcl::CuboidGpu c_dR(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, *r_in);

            ResidualArgsBlockStencil args{
                .return_residual = false,
                .mgh = m + 2 * ghosts,
                .ngh = n + 2 * ghosts,
                .ogh = o + 2 * ghosts,
                .h2 = 1.0 / ((double)m * m),
                .ghosts = ghosts,
                .blocksize = blocksize,
                .program = p.getProgram(),
                .commands = p.getCommands(),
                .wgsize = {128, 1, 1},
                .c_dVIn = d_vin,
                .c_dF = d_f,
                .c_dR = d_r,
                .c_blockStencil = d_bs,
                .pd = p.getProfilingData(),
                .moff = 0,
                .noff = 0,
                .ooff = 0,
                .kernelVersion = KernelVersion::COEFFS_FIRST_V_BLOCK_FIRST,
            };

            ankerl::nanobench::Bench bench;
            bench.timeUnit(1ms, "ms")
                .epochs(CLI_ARGS::bench_epochs)
                .epochIterations(CLI_ARGS::bench_iterations)
                .relative(false);

            std::unique_ptr<std::vector<double>> r_out_bs_coeffs_first_v_block_first = nullptr;
            std::unique_ptr<std::vector<double>> r_out_bs_coeffs_first_v_gp_first = nullptr;
            // std::unique_ptr<mgcl::Cuboid> r_out_global_coeffs_indices_precalc = nullptr;
            // std::unique_ptr<mgcl::Cuboid> r_out_global_coeffs_without_ghosts = nullptr;
            if (CLI_ARGS::checkResults)
            {
                bench.epochs(1).epochIterations(1);
            }

            {
                // reset r to zero
                d_r.fill(p.getProgram(), p.getCommands(), 0.0, true, nullptr, nullptr);

                args.kernelVersion = KernelVersion::COEFFS_FIRST_V_BLOCK_FIRST;
                std::string name = std::string("residual_blockstencil_coeff_first_v_block_first_")
                                       .append(std::to_string(m))
                                       .append("_")
                                       .append(std::to_string(n))
                                       .append("_")
                                       .append(std::to_string(o));

                bench.run(std::string(name).c_str(), [&] { //
                    residual(args);
                    p.finish();
                });

                bench_util::Result res;
                res.name = name;
                res.minTime = bench_util::getMinTime(bench, name);
                res.medianTime = bench_util::getMedianTime(bench, name);
                res.avgTime = bench_util::getAvgTime(bench, name);
                res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                res.m = m;
                res.n = n;
                res.o = o;
                results.push_back(res);

                if (CLI_ARGS::checkResults)
                {
                    r_out_bs_coeffs_first_v_block_first = args.c_dR.read(args.commands, nullptr, true);
                }
            }

            {
                // reset r to zero
                d_r.fill(p.getProgram(), p.getCommands(), 0.0, true, nullptr, nullptr);

                args.kernelVersion = KernelVersion::COEFFS_FIRST_V_GP_FIRST;
                std::string name = std::string("residual_blockstencil_coeff_first_v_gp_first_")
                                       .append(std::to_string(m))
                                       .append("_")
                                       .append(std::to_string(n))
                                       .append("_")
                                       .append(std::to_string(o));

                bench.run(std::string(name).c_str(), [&] { //
                    residual(args);
                    p.finish();
                });

                bench_util::Result res;
                res.name = name;
                res.minTime = bench_util::getMinTime(bench, name);
                res.medianTime = bench_util::getMedianTime(bench, name);
                res.avgTime = bench_util::getAvgTime(bench, name);
                res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                res.m = m;
                res.n = n;
                res.o = o;
                results.push_back(res);

                if (CLI_ARGS::checkResults)
                {
                    r_out_bs_coeffs_first_v_gp_first = args.c_dR.read(args.commands, nullptr, true);
                }
            }

            {
                // reset r to zero
                d_r.fill(p.getProgram(), p.getCommands(), 0.0, true, nullptr, nullptr);

                args.kernelVersion = KernelVersion::BLOCK_FIRST_V_GP_FIRST;
                std::string name = std::string("residual_blockstencil_block_first_v_gp_first_")
                                       .append(std::to_string(m))
                                       .append("_")
                                       .append(std::to_string(n))
                                       .append("_")
                                       .append(std::to_string(o));

                bench.run(std::string(name).c_str(), [&] { //
                    residual(args);
                    p.finish();
                });

                bench_util::Result res;
                res.name = name;
                res.minTime = bench_util::getMinTime(bench, name);
                res.medianTime = bench_util::getMedianTime(bench, name);
                res.avgTime = bench_util::getAvgTime(bench, name);
                res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                res.m = m;
                res.n = n;
                res.o = o;
                results.push_back(res);

                if (CLI_ARGS::checkResults)
                {
                    r_out_bs_coeffs_first_v_gp_first = args.c_dR.read(args.commands, nullptr, true);
                }
            }

            {
                // reset r to zero
                d_r.fill(p.getProgram(), p.getCommands(), 0.0, true, nullptr, nullptr);

                args.kernelVersion = KernelVersion::BLOCK_FIRST_V_BLOCK_FIRST;
                std::string name = std::string("residual_blockstencil_block_first_v_block_first_")
                                       .append(std::to_string(m))
                                       .append("_")
                                       .append(std::to_string(n))
                                       .append("_")
                                       .append(std::to_string(o));

                bench.run(std::string(name).c_str(), [&] { //
                    residual(args);
                    p.finish();
                });

                bench_util::Result res;
                res.name = name;
                res.minTime = bench_util::getMinTime(bench, name);
                res.medianTime = bench_util::getMedianTime(bench, name);
                res.avgTime = bench_util::getAvgTime(bench, name);
                res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                res.m = m;
                res.n = n;
                res.o = o;
                results.push_back(res);

                if (CLI_ARGS::checkResults)
                {
                    r_out_bs_coeffs_first_v_gp_first = args.c_dR.read(args.commands, nullptr, true);
                }
            }

            // Check results for kernels that it is valid for
            if (CLI_ARGS::checkResults)
            {
                REQUIRE(r_out_bs_coeffs_first_v_block_first);
                REQUIRE(r_out_bs_coeffs_first_v_gp_first);

                for (int i = 0; i < r_out_bs_coeffs_first_v_block_first->size(); i++)
                {
                    REQUIRE((*r_out_bs_coeffs_first_v_block_first)[i] == (*r_out_bs_coeffs_first_v_gp_first)[i]);
                }
            }
        }

        bench_util::printCsvFormat(results);

        if (CLI_ARGS::enableKernelProfiling)
        {
            p.getProfilingData()->printBestTimingsPerKernel();
        }
    }
}