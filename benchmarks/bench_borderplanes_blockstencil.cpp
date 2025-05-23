/**
 * @date 27.03.2025
 * This file contains code for benchmarking extract_border_planes for block cuboids.
 * A main concern is the layout, i.e.
    GP_FIRST,    // [m][gpx][gpy][gpz] for v, f, r
    BLOCK_FIRST, // [gpx][gpy][gpz][m] for v, f, r
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

namespace mgcl_bench_borderplanes_blockstencil
{
    enum class KernelVersion
    {
        GP_FIRST,    // [m][gpx][gpy][gpz] for v, f, r
        BLOCK_FIRST, // [gpx][gpy][gpz][m] for v, f, r
    };

    using size_t3 = struct
    {
        int x, y, z;
    };

    using ExtractBorderPlanesArgsBlock = struct
    {
        bool return_residual;
        int m;
        int n;
        int o;
        int mgh;
        int ngh;
        int ogh;
        int ghosts_m;
        int ghosts_n;
        int ghosts_o;
        int blocksize; // Size of matrix in 1 dim, i.e. matrix has blocksize x blocksize elements

        cl_program program;
        cl_command_queue commands;
        cl_context context;
        size_t3 wgsize;

        mgcl::BufferGpu& d_source;
        mgcl::BufferGpu& d_target;
        // std::vector<double>* h_target;

        mgcl::ProfilingData* pd;
        // mgcl::conf::KernelConfig* conf;

        KernelVersion kernelVersion;
    };

    /**
     * @brief Extracts the border planes of the cuboid. Does not read back result.
     *
     * @param commands OpenCL command queue
     * @param program OpenCL program
     * @param d_target CuboidGpu that data gets extracted into. If nullptr, a new CuboidGpu is created temporarily.
     * @param h_target Cuboid that data gets extracted into. If target is nullptr, a new Cuboid is created and returned.
     * @return std::unique_ptr<Cuboid>
     */
    void extractBorderPlanes(ExtractBorderPlanesArgsBlock& args)
    {
        // Plane sizes
        int yz = args.ngh * args.ogh;
        int xz = args.mgh * args.ogh;
        int xy = args.mgh * args.ngh;
        int ressize = 2 * yz * args.ghosts_m + 2 * xz * args.ghosts_n + 2 * xy * args.ghosts_o;

        if (args.ghosts_m > args.m || args.ghosts_n > args.n || args.ghosts_o > args.o)
            error("CuboidGpu::extractBorderPlanes: Only defined for ghosts <= m, n, o");

        // // Create return buffer, if not provided
        // std::unique_ptr<std::vector<double>> ret = nullptr;
        // std::vector<double>* retraw = h_target;
        // if (h_target == nullptr)
        // {
        //     ret = std::make_unique<std::vector<double>>(ressize);
        //     retraw = ret.get();
        // }

        // // Create device target buffer, if not provided
        // bool createdDTarget = false;
        // if (d_target == nullptr)
        // {
        //     d_target = new BufferGpu(context, CL_MEM_READ_WRITE, ressize);
        //     d_target->write(commands, *retraw, true);
        //     createdDTarget = true;
        // }

        int err;

        // Create the compute kernel from the program
        const char* kernelName = "";
        if (args.kernelVersion == KernelVersion::GP_FIRST)
        {
            kernelName = "extract_border_planes_gp_first";
        }
        else if (args.kernelVersion == KernelVersion::BLOCK_FIRST)
        {
            kernelName = "extract_border_planes_block_first";
        }
        else
        {
            error("Unknown kernel version");
        }

        cl_kernel kernel = clCreateKernel(args.program, kernelName, &err);
        mgcl::mgclCheckError(err, "clCreateKernel");

        // assign kernel arguments
        cl_mem d_source_buffer = args.d_source.getBuf();
        cl_mem d_target_buffer = args.d_target.getBuf();
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &d_source_buffer);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &d_target_buffer);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.mgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.ngh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.ogh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.ghosts_m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.ghosts_n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &args.ghosts_o);
        mgcl::mgclCheckError(err, "Setting kernel arguments");

        // one work-item per ghost cell (excluding real cells). Pad global sizes to fit to local sizes
        size_t global = ressize;
        size_t local = 32;
        // Apply kernel config, if available
        // if (conf)
        // {
        //     const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(*conf, kernelName, global);
        //     local = c[0];
        // }

        if (global % local != 0)
            global += local - (global % local);

        cl_event ev;

        // enqueue kernel
        err = clEnqueueNDRangeKernel(args.commands, kernel, 1, NULL, &global, &local, 0, NULL, &ev);
        mgcl::mgclCheckError(err, "Enqueueing extract_border_planes kernel");

        if (args.pd != nullptr)
        {
            args.pd->addMeasurement(args.commands, ev, kernelName,
                                    {global, 0, 0},
                                    {local, 1, 1});
        }
        mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        mgcl::mgclCheckError(clFinish(args.commands), "clFinish");

        err = clReleaseKernel(kernel);
        mgcl::mgclCheckError(err, "Releasing extract_border_planes kernel");

        // // Read into h_target
        // d_target->read(commands, retraw->data(), true, ressize);

        // if (createdDTarget)
        //     delete d_target;

        // return ret;
    }

    // Benchs the various residual fixed stencil kernel versions
    TEST_CASE("borderPlanesBlockstencilCuboidLayouts")
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

            mgcl::BufferGpu d_source(p.getContext(), CL_MEM_READ_WRITE, mgh * ngh * ogh * blocksize);
            mgcl::BufferGpu d_target(p.getContext(), CL_MEM_READ_WRITE, mgh * ngh * ogh * blocksize);

            d_source.fill(p.getProgram(), p.getCommands(), 0.0, true, nullptr, nullptr);

            ExtractBorderPlanesArgsBlock args{
                .return_residual = false,
                .m = m,
                .n = n,
                .o = o,
                .mgh = m + 2 * ghosts,
                .ngh = n + 2 * ghosts,
                .ogh = o + 2 * ghosts,
                .ghosts_m = ghosts,
                .ghosts_n = ghosts,
                .ghosts_o = ghosts,
                .blocksize = blocksize,
                .program = p.getProgram(),
                .commands = p.getCommands(),
                .context = p.getContext(),
                .wgsize = {128, 1, 1},
                .d_source = d_source,
                .d_target = d_target,
                .pd = p.getProfilingData(),
                .kernelVersion = KernelVersion::BLOCK_FIRST,
            };

            ankerl::nanobench::Bench bench;
            bench.timeUnit(1us, "us")
                .epochs(CLI_ARGS::bench_epochs)
                .epochIterations(CLI_ARGS::bench_iterations)
                .relative(false);

            // checkResults does not make sense for this test yet
            // std::unique_ptr<std::vector<double>> r_out_bs_coeffs_first_v_block_first = nullptr;
            // std::unique_ptr<std::vector<double>> r_out_bs_coeffs_first_v_gp_first = nullptr;
            // std::unique_ptr<std::vector<double>> r_out_bs_block_first_v_block_first = nullptr;
            // std::unique_ptr<std::vector<double>> r_out_bs_block_first_v_gp_first = nullptr;
            // if (CLI_ARGS::checkResults)
            // {
            //     bench.epochs(1).epochIterations(1);
            // }

            {
                args.kernelVersion = KernelVersion::BLOCK_FIRST;
                std::string name = std::string("extractBorderPlanes_blockstencil_cuboid_block_first_")
                                       .append(std::to_string(m))
                                       .append("_")
                                       .append(std::to_string(n))
                                       .append("_")
                                       .append(std::to_string(o));

                bench.run(std::string(name).c_str(), [&] { //
                    extractBorderPlanes(args);
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
                //     r_out_bs_coeffs_first_v_block_first = args.c_dR.read(args.commands, nullptr, true);
                // }
            }

            {
                args.kernelVersion = KernelVersion::GP_FIRST;
                std::string name = std::string("extractBorderPlanes_blockstencil_cuboid_gp_first_")
                                       .append(std::to_string(m))
                                       .append("_")
                                       .append(std::to_string(n))
                                       .append("_")
                                       .append(std::to_string(o));

                bench.run(std::string(name).c_str(), [&] { //
                    extractBorderPlanes(args);
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
                //     r_out_bs_coeffs_first_v_block_first = args.c_dR.read(args.commands, nullptr, true);
                // }
            }
        }
        bench_util::printCsvFormat(results);

        if (CLI_ARGS::enableKernelProfiling)
        {
            p.getProfilingData()->printBestTimingsPerKernel();
        }
    }
}