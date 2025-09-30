/**
 * @date 12.12.2024
 * This file contains code for benchmarking various versions of the residual kernel
 * when using a FixedStencil, i.e. 27pt stencil with fixed but choosable coefficients for all grid-points.
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

namespace mgcl_bench_residual_varying
{
    enum class KernelVersion
    {
        COEFFS_FIRST_1D,
        GPS_FIRST_1D,
        COEFFS_FIRST_3D_M0,
        COEFFS_FIRST_3D_O0,
        GPS_FIRST_3D_O0,
        GPS_FIRST_1D_RESTRUCTURED,
        GPS_FIRST_1D_REAL_ONLY,
        GPS_FIRST_1D_2WI_PER_GP,
        GPS_FIRST_1D_4WI_PER_GP,
        COEFFS_FIRST_1D_4WI_PER_GP,
        COEFFS_FIRST_1D_2WI_PER_GP_SHMEM_SPREAD,
        COEFFS_FIRST_1D_4WI_PER_GP_SHMEM_SPREAD,

        REMOVED_V,
        // COEFFS_WITHOUT_GHOSTS // not needed, using COEFFS_FIRST for this
        FOUR_GP_PER_WI,
        COEFF_INDICES_PRECALC
    };

    using size_t3 = struct
    {
        size_t x, y, z;
    };

    using ResidualArgs = struct
    {
        bool return_residual;
        int mgh;
        int ngh;
        int ogh;
        double h2;
        int ghosts;
        mgcl::MGCL_STENCIL stencilType;

        cl_program program;
        cl_command_queue commands;
        size_t3 wgsize;

        mgcl::CuboidGpu& c_dVIn;
        mgcl::CuboidGpu& c_dF;
        mgcl::CuboidGpu& c_dR;
        mgcl::VaryingStencilGpu* c_stencilValues;
        mgcl::FixedStencilGpu* c_fixedStencil;

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
    double residual(ResidualArgs& args)
    {
        if (args.kernelVersion == KernelVersion::COEFF_INDICES_PRECALC && args.mgh != 66)
        {
            throw "COEFF_INDICES_PRECALC needs grid 64^3 with 1 ghosts";
        }

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

        if (args.c_dVIn.getGhostsM() < 1 || args.c_dVIn.getGhostsN() < 1 || args.c_dVIn.getGhostsO() < 1)
        {
            error("args.c_dVIn must have at least 1 ghost cell in each dimension");
        }

        if (args.c_dF.getGhostsM() < 1 || args.c_dF.getGhostsN() < 1 || args.c_dF.getGhostsO() < 1)
        {
            error("args.c_dF must have at least 1 ghost cell in each dimension");
        }

        if (args.c_dR.getGhostsM() < 1 || args.c_dR.getGhostsN() < 1 || args.c_dR.getGhostsO() < 1)
        {
            error("args.c_dR must have at least 1 ghost cell in each dimension");
        }

        // This is only for selecting the kernel to benchmark. Not in productive code.
        std::string kernelName = "";
        if (args.kernelVersion == KernelVersion::COEFFS_FIRST_1D)
        {
            kernelName = "residual_27point_varying_stencil_coeffs_first_1d";
        }
        else if (args.kernelVersion == KernelVersion::REMOVED_V)
        {
            kernelName = "residual_27point_varying_stencil_no_v";
        }
        else if (args.kernelVersion == KernelVersion::FOUR_GP_PER_WI)
        {
            kernelName = "residual_27point_varying_stencil_coeffs_first_4_gps_per_thread";
        }
        else if (args.kernelVersion == KernelVersion::COEFF_INDICES_PRECALC)
        {
            kernelName = "residual_27point_varying_stencil_coeffs_first_indices_precalc_64";
        }
        else if (args.kernelVersion == KernelVersion::COEFFS_FIRST_3D_M0)
        {
            kernelName = "residual_27point_varying_stencil_coeffs_first_3d_m0";
        }
        else if (args.kernelVersion == KernelVersion::COEFFS_FIRST_3D_O0)
        {
            kernelName = "residual_27point_varying_stencil_coeffs_first_3d_o0";
        }
        else if (args.kernelVersion == KernelVersion::GPS_FIRST_1D)
        {
            kernelName = "residual_27point_varying_stencil_gps_first_1d";
        }
        else if (args.kernelVersion == KernelVersion::GPS_FIRST_3D_O0)
        {
            kernelName = "residual_27point_varying_stencil_gps_first_3d_o0";
        }
        else if (args.kernelVersion == KernelVersion::GPS_FIRST_1D_RESTRUCTURED)
        {
            kernelName = "residual_27point_varying_stencil_1d_gps_first_restructured";
        }
        else if (args.kernelVersion == KernelVersion::GPS_FIRST_1D_REAL_ONLY)
        {
            kernelName = "residual_27point_varying_stencil_1d_gps_first_real_only";
        }
        else if (args.kernelVersion == KernelVersion::GPS_FIRST_1D_2WI_PER_GP)
        {
            kernelName = "residual_27point_varying_stencil_1d_mult_wi_per_cell_2";
        }
        else if (args.kernelVersion == KernelVersion::GPS_FIRST_1D_4WI_PER_GP)
        {
            kernelName = "residual_27point_varying_stencil_1d_mult_wi_per_cell_4";
        }
        else if (args.kernelVersion == KernelVersion::COEFFS_FIRST_1D_4WI_PER_GP)
        {
            kernelName = "residual_27point_varying_stencil_1d_mult_wi_per_cell_4_sv_spread";
        }
        else if (args.kernelVersion == KernelVersion::COEFFS_FIRST_1D_4WI_PER_GP_SHMEM_SPREAD)
        {
            kernelName = "residual_27point_varying_stencil_1d_mult_wi_per_cell_4_sv_spread_shmem_spread";
        }
        else if (args.kernelVersion == KernelVersion::COEFFS_FIRST_1D_2WI_PER_GP_SHMEM_SPREAD)
        {
            kernelName = "residual_27point_varying_stencil_1d_mult_wi_per_cell_2_sv_spread_shmem_spread";
        }

        cl_event ev;

        // Create the compute kernel from the program
        cl_kernel kernel = clCreateKernel(args.program, kernelName.c_str(), &err);
        mgcl::mgclCheckError(err, "Creating kernel");

        cl_mem dVIn = args.c_dVIn.getBuffer();
        cl_mem dF = args.c_dF.getBuffer();
        cl_mem dR = args.c_dR.getBuffer();

        // args.c_dVIn.dumpToFile(args.commands, "dVIn.txt");
        // args.c_dF.dumpToFile(args.commands, "dF.txt");
        // args.c_dR.dumpToFile(args.commands, "dR.txt");

        bool isMultipleWisPerGp = (args.kernelVersion == KernelVersion::COEFFS_FIRST_1D_2WI_PER_GP_SHMEM_SPREAD ||
                                   args.kernelVersion == KernelVersion::COEFFS_FIRST_1D_4WI_PER_GP_SHMEM_SPREAD ||
                                   args.kernelVersion == KernelVersion::COEFFS_FIRST_1D_4WI_PER_GP ||
                                   args.kernelVersion == KernelVersion::GPS_FIRST_1D_4WI_PER_GP ||
                                   args.kernelVersion == KernelVersion::GPS_FIRST_1D_2WI_PER_GP);

        // assign kernel arguments
        int pos = 0;
        if (args.stencilType == mgcl::MGCL_VARYING)
        {
            // if (args.kernelVersion == KernelVersion::COEFFS_FIRST_1D ||
            //     args.kernelVersion == KernelVersion::REMOVED_V ||
            //     args.kernelVersion == KernelVersion::FOUR_GP_PER_WI ||
            //     args.kernelVersion == KernelVersion::COEFF_INDICES_PRECALC)
            {
                auto svbuf = args.c_stencilValues->getBuf();
                int svgh = args.c_stencilValues->getGh();
                int svmgh = args.c_stencilValues->getMgh();
                int svngh = args.c_stencilValues->getNgh();
                int svogh = args.c_stencilValues->getOgh();
                int svGridSize = svmgh * svngh * svogh;

                err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dVIn);
                err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dF);
                err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dR);
                err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &svbuf);

                if (isMultipleWisPerGp)
                {
                    int locmem_size = args.wgsize.x; // TODO check. Possibly less needed, but need to adjsut kernels then
                    // if (args.kernelVersion == KernelVersion::COEFFS_FIRST_1D_2WI_PER_GP_SHMEM_SPREAD ||
                    //     args.kernelVersion == KernelVersion::GPS_FIRST_1D_2WI_PER_GP)
                    // {
                    //     locmem_size = args.wgsize.x / 2;
                    // }
                    // else
                    // {
                    //     locmem_size = (args.wgsize.x / 4) * 3;
                    // }

                    err |= clSetKernelArg(kernel, ++pos, sizeof(double) * locmem_size, NULL);
                }

                if (args.kernelVersion == KernelVersion::GPS_FIRST_1D_REAL_ONLY)
                {
                    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
                    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
                    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
                }
                else
                {
                    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.mgh);
                    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.ngh);
                    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.ogh);
                }
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svmgh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svngh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svogh);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.ghosts);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svgh);
                if (args.kernelVersion != KernelVersion::COEFF_INDICES_PRECALC)
                {
                    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svGridSize);
                }
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.moff);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.noff);
                err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.ooff);

                if (isMultipleWisPerGp)
                {
                    int wiPerGridPoint = 4;
                    if (args.kernelVersion == KernelVersion::GPS_FIRST_1D_2WI_PER_GP || args.kernelVersion == KernelVersion::COEFFS_FIRST_1D_2WI_PER_GP_SHMEM_SPREAD)
                    {
                        wiPerGridPoint = 2;
                    }
                    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &wiPerGridPoint);
                }
            }
        }

        mgcl::mgclCheckError(err, "Setting residual kernel arguments");

        size_t global[3];
        size_t local[3];
        bool is3dKernel = (args.kernelVersion == KernelVersion::COEFFS_FIRST_3D_M0 ||
                           args.kernelVersion == KernelVersion::COEFFS_FIRST_3D_O0 ||
                           args.kernelVersion == KernelVersion::GPS_FIRST_3D_O0);
        if (is3dKernel)
        {
            global[0] = static_cast<size_t>(args.mgh);
            global[1] = static_cast<size_t>(args.ngh);
            global[2] = static_cast<size_t>(args.ogh);
            local[0] = args.wgsize.x;
            local[1] = args.wgsize.y;
            local[2] = args.wgsize.z;
            // decrease wg size for bigger grids
            // if (mgh >= 32 && ngh >= 32 && ogh >= 32)
            //     local[2] = 16;

            for (int i = 0; i < 3; i++)
                if (global[i] % local[i] != 0)
                {
                    // printf("padding global size %d from %ld to ", i, global[i]);
                    global[i] += local[i] - (global[i] % local[i]);
                    // printf("%ld (multiple of %ld)\n", global[i], local[i]);
                }
        }
        else
        {
            // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
            global[0] = args.mgh * args.ngh * args.ogh;
            if (args.kernelVersion == KernelVersion::FOUR_GP_PER_WI)
            {
                global[0] /= 4;
            }
            if (args.kernelVersion == KernelVersion::GPS_FIRST_1D_REAL_ONLY)
            {
                global[0] = m * n * o;
            }
            if (isMultipleWisPerGp)
            {
                if (args.kernelVersion == KernelVersion::GPS_FIRST_1D_2WI_PER_GP || args.kernelVersion == KernelVersion::COEFFS_FIRST_1D_2WI_PER_GP_SHMEM_SPREAD)
                {
                    global[0] *= 2;
                }
                else
                {
                    global[0] *= 4;
                }
            }

            // const auto& c = mgcl::conf::getWorkGroupSizeForKernelAndWiCount(problem.getKernelConfig(), kernelName, global);
            local[0] = args.wgsize.x; // c[0];

            if (global[0] % local[0] != 0)
                global[0] += local[0] - (global[0] % local[0]);

            global[1] = 1;
            global[2] = 1;
            local[1] = 1;
            local[2] = 1;
        }

        err = clEnqueueNDRangeKernel(args.commands, kernel, is3dKernel ? 3 : 1, NULL, global, local, 0, NULL, &ev);
        mgcl::mgclCheckError(err, "Enqueueing residual kernel");

        if (args.pd)
        {
            args.pd->addMeasurement(args.commands, ev, kernelName,
                                    {global[0], global[1], global[2]},
                                    {local[0], local[1], local[2]});
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
    TEST_CASE("residualVaryingKernelVersions")
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
        p.setKernelFile("residual_kernels_1d_vs_3d.cl");
        if (CLI_ARGS::useBinaryFile)
        {
            p.setBinaryFile("residualVaryingKernelVersions.bin");
        }
        p.setUseOpencl(true);
        p.setDeviceType(CL_DEVICE_TYPE_GPU);
        p.setProfilingEnabled(CLI_ARGS::enableKernelProfiling);
        p.init();

        int ghosts = 1;

        for (auto gr : gridsTBT)
        {
            int m = gr[0];
            int n = gr[1];
            int o = gr[2];

            auto v_in = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            auto f_in = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            auto r_in = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            // v_in->fill1dIndex(true);
            // f_in->fill1dIndex(true);
            v_in->fillRandom();
            f_in->fillRandom();

            mgcl::CuboidGpu c_dVIn(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, *v_in);
            mgcl::CuboidGpu c_dF(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, *f_in);
            mgcl::CuboidGpu c_dR(p.getContext(), CL_MEM_COPY_HOST_PTR | CL_MEM_READ_WRITE, *r_in);

            auto vs_in = std::make_shared<mgcl::VaryingStencil>(m, n, o, 3, ghosts, ghosts, ghosts);
            vs_in->fill1dIndex(false);
            auto c_varyingStencil = std::make_shared<mgcl::VaryingStencilGpu>(m, n, o, 3, ghosts, p.getContext(), p.getCommands(), p.getProgram());
            c_varyingStencil->fill(*vs_in, p.getCommands(), true);

            ResidualArgs args{
                .return_residual = false,
                .mgh = m + 2 * ghosts,
                .ngh = n + 2 * ghosts,
                .ogh = o + 2 * ghosts,
                .h2 = 1.0 / ((double)m * m),
                .ghosts = ghosts,
                .stencilType = mgcl::MGCL_VARYING,
                .program = p.getProgram(),
                .commands = p.getCommands(),
                .wgsize = {128, 1, 1},
                .c_dVIn = c_dVIn,
                .c_dF = c_dF,
                .c_dR = c_dR,
                .c_stencilValues = c_varyingStencil.get(),
                .c_fixedStencil = nullptr,
                .pd = p.getProfilingData(),
                .moff = 0,
                .noff = 0,
                .ooff = 0,
                .kernelVersion = KernelVersion::COEFFS_FIRST_1D,
            };

            ankerl::nanobench::Bench bench;
            bench.timeUnit(1ms, "ms")
                .epochs(CLI_ARGS::bench_epochs)
                .epochIterations(CLI_ARGS::bench_iterations)
                .relative(false);

            std::unique_ptr<mgcl::Cuboid> r_out_global_coeffs_first_1d = nullptr;
            std::unique_ptr<mgcl::Cuboid> r_out_global_coeffs_first_3d_m0 = nullptr;
            std::unique_ptr<mgcl::Cuboid> r_out_global_coeffs_first_3d_o0 = nullptr;
            std::unique_ptr<mgcl::Cuboid> r_out_global_gps_first_1d_restructured = nullptr;
            std::unique_ptr<mgcl::Cuboid> r_out_global_gps_first_1d_real_only = nullptr;
            std::unique_ptr<mgcl::Cuboid> r_out_global_gps_first_1d_2wi_per_gp = nullptr;
            std::unique_ptr<mgcl::Cuboid> r_out_global_gps_first_1d_4wi_per_gp = nullptr;
            std::unique_ptr<mgcl::Cuboid> r_out_global_gps_first_1d_4wi_per_gp_shmem_spread = nullptr;
            std::unique_ptr<mgcl::Cuboid> r_out_global_coeffs_first_1d_4wi_per_gp = nullptr;
            std::unique_ptr<mgcl::Cuboid> r_out_global_coeffs_first_1d_4wi_per_gp_shmem_spread = nullptr;
            std::unique_ptr<mgcl::Cuboid> r_out_global_coeffs_first_1d_2wi_per_gp_shmem_spread = nullptr;
            // std::unique_ptr<mgcl::Cuboid> r_out_global_coeffs_indices_precalc = nullptr;
            // std::unique_ptr<mgcl::Cuboid> r_out_global_coeffs_without_ghosts = nullptr;
            std::unique_ptr<mgcl::Cuboid> r_out_global_coeffs_4_gp_per_thread = nullptr;
            // std::unique_ptr<mgcl::Cuboid> r_out_global_gps_first_1d = nullptr;
            std::unique_ptr<mgcl::Cuboid> r_out_global_gps_first_3d_m0 = nullptr;
            if (CLI_ARGS::checkResults)
            {
                bench.epochs(1).epochIterations(1);
            }

            {
                args.kernelVersion = KernelVersion::COEFFS_FIRST_1D;
                std::string name = std::string("residual_varying_stencil_coeffs_first_1d_")
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
                    r_out_global_coeffs_first_1d = std::make_unique<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
                    args.c_dR.read(args.commands, r_out_global_coeffs_first_1d.get(), true);
                }
            }

            {
                // result check will be wrong since we just use stencilValues with [coeffs][m,n,o] layout! For timings,
                // only the access pattern matters and we don't run into illegal memory accesses since the overall size
                // is the same.
                args.kernelVersion = KernelVersion::GPS_FIRST_1D;
                std::string name = std::string("residual_varying_stencil_gps_first_1d_")
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

                // if (CLI_ARGS::checkResults)
                // {
                //     r_out_global_gps_first_1d = std::make_unique<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
                //     args.c_dR.read(args.commands, r_out_global_gps_first_1d.get(), true);
                // }
            }

            if (m == 64)
            {
                args.kernelVersion = KernelVersion::COEFF_INDICES_PRECALC;
                std::string name = std::string("residual_varying_stencil_coeff_indices_precalc_")
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

                // if (CLI_ARGS::checkResults)
                // {
                //     r_out_global_coeffs_first = std::make_unique<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
                //     args.c_dR.read(args.commands, r_out_global_coeffs_first.get(), true);
                // }
            }

            {
                args.kernelVersion = KernelVersion::FOUR_GP_PER_WI;
                std::string name = std::string("residual_varying_stencil_coeffs_first_4_gps_per_thread")
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
                    r_out_global_coeffs_4_gp_per_thread = std::make_unique<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
                    args.c_dR.read(args.commands, r_out_global_coeffs_4_gp_per_thread.get(), true);
                }
            }

            {
                args.kernelVersion = KernelVersion::GPS_FIRST_1D_RESTRUCTURED;
                std::string name = std::string("residual_varying_stencil_gps_first_1d_restructured_")
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
                    r_out_global_gps_first_1d_restructured = std::make_unique<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
                    args.c_dR.read(args.commands, r_out_global_gps_first_1d_restructured.get(), true);
                }
            }

            {

                args.kernelVersion = KernelVersion::GPS_FIRST_1D_REAL_ONLY;
                std::string name = std::string("residual_varying_stencil_gps_first_1d_real_only_")
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
                    r_out_global_gps_first_1d_real_only = std::make_unique<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
                    args.c_dR.read(args.commands, r_out_global_gps_first_1d_real_only.get(), true);
                }
            }

            {
                args.kernelVersion = KernelVersion::GPS_FIRST_1D_2WI_PER_GP;
                std::string name = std::string("residual_varying_stencil_gps_first_1d_2wi_per_gp_")
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
                    r_out_global_gps_first_1d_2wi_per_gp = std::make_unique<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
                    args.c_dR.read(args.commands, r_out_global_gps_first_1d_2wi_per_gp.get(), true);
                }
            }

            {
                args.kernelVersion = KernelVersion::GPS_FIRST_1D_4WI_PER_GP;
                std::string name = std::string("residual_varying_stencil_gps_first_1d_4wi_per_gp_")
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
                    r_out_global_gps_first_1d_4wi_per_gp = std::make_unique<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
                    args.c_dR.read(args.commands, r_out_global_gps_first_1d_4wi_per_gp.get(), true);
                }
            }

            {
                args.kernelVersion = KernelVersion::COEFFS_FIRST_1D_4WI_PER_GP;
                std::string name = std::string("residual_varying_stencil_coeffs_first_1d_4wi_per_gp_")
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
                    r_out_global_coeffs_first_1d_4wi_per_gp = std::make_unique<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
                    args.c_dR.read(args.commands, r_out_global_coeffs_first_1d_4wi_per_gp.get(), true);
                }
            }

            {
                args.kernelVersion = KernelVersion::COEFFS_FIRST_1D_4WI_PER_GP_SHMEM_SPREAD;
                std::string name = std::string("residual_varying_stencil_coeffs_first_1d_4wi_per_gp_shmem_spread_")
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
                    r_out_global_coeffs_first_1d_4wi_per_gp_shmem_spread = std::make_unique<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
                    args.c_dR.read(args.commands, r_out_global_coeffs_first_1d_4wi_per_gp_shmem_spread.get(), true);
                }
            }

            {
                args.kernelVersion = KernelVersion::COEFFS_FIRST_1D_2WI_PER_GP_SHMEM_SPREAD;
                std::string name = std::string("residual_varying_stencil_coeffs_first_1d_2wi_per_gp_shmem_spread_")
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
                    r_out_global_coeffs_first_1d_2wi_per_gp_shmem_spread = std::make_unique<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
                    args.c_dR.read(args.commands, r_out_global_coeffs_first_1d_2wi_per_gp_shmem_spread.get(), true);
                }
            }

            // {
            //     args.kernelVersion = KernelVersion::REMOVED_V;
            //     std::string name = std::string("residual_varying_no_v_")
            //                            .append(std::to_string(m))
            //                            .append("_")
            //                            .append(std::to_string(n))
            //                            .append("_")
            //                            .append(std::to_string(o));

            //     bench.run(std::string(name).c_str(), [&] { //
            //         residual(args);
            //         p.finish();
            //     });

            //     bench_util::Result res;
            //     res.name = name;
            //     res.minTime = bench_util::getMinTime(bench, name);
            //     res.medianTime = bench_util::getMedianTime(bench, name);
            //     res.avgTime = bench_util::getAvgTime(bench, name);
            //     res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            //     res.m = m;
            //     res.n = n;
            //     res.o = o;
            //     results.push_back(res);

            //     // No result checking for this kernel
            //     // if (CLI_ARGS::checkResults)
            //     // {
            //     //     r_out_global_coeffs_first = std::make_unique<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            //     //     args.c_dR.read(args.commands, r_out_global_coeffs_first.get(), true);
            //     // }
            // }

            // // IMPORTANT: Do this test last as it needs to free stencil values and reallocate.
            // {
            //     // actually coeffs without ghosts
            //     args.kernelVersion = KernelVersion::COEFFS_FIRST;

            //     // free ghosted stencil values
            //     vs_in.reset();
            //     c_varyingStencil.reset();

            //     vs_in = std::make_shared<mgcl::VaryingStencil>(m, n, o, 3, 0, 0, 0);
            //     vs_in->fill1dIndex(false);
            //     c_varyingStencil = std::make_shared<mgcl::VaryingStencilGpu>(m, n, o, 3, 0, p.getContext(), p.getCommands(), p.getProgram());
            //     c_varyingStencil->fill(*vs_in, p.getCommands(), true);

            //     std::string name = std::string("residual_varying_stencil_coeffs_without_ghosts_")
            //                            .append(std::to_string(m))
            //                            .append("_")
            //                            .append(std::to_string(n))
            //                            .append("_")
            //                            .append(std::to_string(o));

            //     bench.run(std::string(name).c_str(), [&] { //
            //         residual(args);
            //         p.finish();
            //     });

            //     bench_util::Result res;
            //     res.name = name;
            //     res.minTime = bench_util::getMinTime(bench, name);
            //     res.medianTime = bench_util::getMedianTime(bench, name);
            //     res.avgTime = bench_util::getAvgTime(bench, name);
            //     res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            //     res.m = m;
            //     res.n = n;
            //     res.o = o;
            //     results.push_back(res);

            //     // No result checking for this kernel. Would need to manually write coefficients to the same as ghosted one
            //     // if (CLI_ARGS::checkResults)
            //     // {
            //     //     r_out_global_coeffs_without_ghosts = std::make_unique<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            //     //     args.c_dR.read(args.commands, r_out_global_coeffs_without_ghosts.get(), true);
            //     // }
            // }

            std::vector<std::vector<size_t>> wg_sizes = {{4, 4, 4}, {1, 1, 16}, {1, 1, 32}, {1, 1, 64}, {16, 1, 1}, {32, 1, 1}, {64, 1, 1}};
            {
                args.kernelVersion = KernelVersion::COEFFS_FIRST_3D_M0;
                // args.wgsize = {16, 16, 4};
                for (auto ws : wg_sizes)
                {
                    args.wgsize = {ws[0], ws[1], ws[2]};
                    std::string name = std::string("residual_varying_stencil_coeffs_first_3d_m0_")
                                           .append(std::to_string(m))
                                           .append("_")
                                           .append(std::to_string(n))
                                           .append("_")
                                           .append(std::to_string(o))
                                           .append("_wg")
                                           .append(std::to_string(ws[0]))
                                           .append("x")
                                           .append(std::to_string(ws[1]))
                                           .append("x")
                                           .append(std::to_string(ws[2]));

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
                        r_out_global_coeffs_first_3d_m0 = std::make_unique<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
                        args.c_dR.read(args.commands, r_out_global_coeffs_first_3d_m0.get(), true);
                    }
                }
            }

            {
                args.kernelVersion = KernelVersion::COEFFS_FIRST_3D_O0;
                // args.wgsize = {16, 16, 4};
                for (auto ws : wg_sizes)
                {
                    args.wgsize = {ws[0], ws[1], ws[2]};
                    std::string name = std::string("residual_varying_stencil_coeffs_first_3d_o0_")
                                           .append(std::to_string(m))
                                           .append("_")
                                           .append(std::to_string(n))
                                           .append("_")
                                           .append(std::to_string(o))
                                           .append("_wg")
                                           .append(std::to_string(ws[0]))
                                           .append("x")
                                           .append(std::to_string(ws[1]))
                                           .append("x")
                                           .append(std::to_string(ws[2]));

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
                        r_out_global_coeffs_first_3d_o0 = std::make_unique<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
                        args.c_dR.read(args.commands, r_out_global_coeffs_first_3d_o0.get(), true);
                    }
                }

                {
                    // result check will be wrong since we just use stencilValues with [coeffs][m,n,o] layout! For timings,
                    // only the access pattern matters and we don't run into illegal memory accesses since the overall size
                    // is the same.
                    args.kernelVersion = KernelVersion::GPS_FIRST_3D_O0;
                    std::string name = std::string("residual_27point_varying_stencil_gps_first_3d_o0")
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
                        r_out_global_gps_first_3d_m0 = std::make_unique<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
                        args.c_dR.read(args.commands, r_out_global_gps_first_3d_m0.get(), true);
                    }
                }
            }

            // Check results for kernels that it is valid for
            if (CLI_ARGS::checkResults)
            {
                // coeffs first have different results than gps first, since we don't actually convert stenciValues layout
                REQUIRE(r_out_global_coeffs_first_1d->isEqual(*r_out_global_coeffs_4_gp_per_thread));
                REQUIRE(r_out_global_coeffs_first_1d->isEqual(*r_out_global_coeffs_first_3d_m0));
                REQUIRE(r_out_global_coeffs_first_1d->isEqual(*r_out_global_coeffs_first_3d_o0));
                REQUIRE(r_out_global_coeffs_first_1d->isEqual(*r_out_global_coeffs_first_1d_4wi_per_gp));
                REQUIRE(r_out_global_coeffs_first_1d->isEqual(*r_out_global_coeffs_first_1d_4wi_per_gp_shmem_spread));
                REQUIRE(r_out_global_coeffs_first_1d->isEqual(*r_out_global_coeffs_first_1d_2wi_per_gp_shmem_spread));

                // r_out_global_coeffs_first_1d->dumpToFile("r_out_global_coeffs_first_1d.txt");
                // r_out_global_coeffs_first_1d_2wi_per_gp_shmem_spread->dumpToFile("r_out_global_coeffs_first_1d_2wi_per_gp_shmem_spread.txt");

                REQUIRE(r_out_global_gps_first_3d_m0->isEqual(*r_out_global_gps_first_1d_restructured));
                REQUIRE(r_out_global_gps_first_3d_m0->isEqual(*r_out_global_gps_first_1d_real_only));
                REQUIRE(r_out_global_gps_first_3d_m0->isEqual(*r_out_global_gps_first_1d_2wi_per_gp));
                REQUIRE(r_out_global_gps_first_3d_m0->isEqual(*r_out_global_gps_first_1d_4wi_per_gp));
            }
        }

        bench_util::printCsvFormat(results);

        if (CLI_ARGS::enableKernelProfiling)
        {
            p.getProfilingData()->printBestTimingsPerKernel();
        }
    }
}