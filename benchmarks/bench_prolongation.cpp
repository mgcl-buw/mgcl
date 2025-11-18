#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"

#include <chrono>
#include <iostream>
#include <vector>
using namespace std::chrono_literals;

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/mpi_util.hpp"
#include "../src/mgcl/multigrid_engine.hpp"
#include "../src/mgcl/opencl_helper.hpp"
#include "../src/mgcl/problem.hpp"
#include "bench_util.hpp"
#include "cli_args.hpp"

namespace mgcl_bench_prolongation
{
    enum class KernelVersion
    {
        THREE_D,
        THREE_D_8WI_PER_GP
    };

    using size_t3 = struct
    {
        int x, y, z;
    };

    // Regular jacobi method like in production code, but with mpi stuff removed. I.e. only single gpu
    int prolongate(mgcl::Problem& problem, mgcl::CuboidGpu& fine, mgcl::CuboidGpu& coarse, KernelVersion kernelVersion, std::vector<size_t> wgsizes)
    {
        int err;

        bool is3d = true;

        // Create the compute kernel from the program
        std::string kernelName;
        if (kernelVersion == KernelVersion::THREE_D)
            kernelName = "prolongate_to_fine";
        else if (kernelVersion == KernelVersion::THREE_D_8WI_PER_GP)
            kernelName = "prolongate_to_fine_8wi_per_gp";
        cl_kernel kernel = clCreateKernel(problem.getOpenCLHelper().getProgram(), kernelName.c_str(), &err);
        mgcl::mgclCheckError(err, "clCreateKernel");

        cl_mem buf_fine = fine.getBuffer();
        cl_mem buf_coarse = coarse.getBuffer();

        int ngh_vals_coarse = coarse.getNgh();
        int ogh_vals_coarse = coarse.getOgh();

        int ghosts = problem.getGhosts();
        int fine_mgh = fine.getMgh();
        int fine_ngh = fine.getNgh();
        int fine_ogh = fine.getOgh();
        int coarse_mgh = coarse.getMgh();
        int coarse_ngh = coarse.getNgh();
        int coarse_ogh = coarse.getOgh();

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &buf_fine);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &buf_coarse);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &fine_mgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &fine_ngh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &fine_ogh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh_vals_coarse);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh_vals_coarse);
        mgcl::mgclCheckError(err, "Setting kernel arguments");

        size_t global[3] = {static_cast<size_t>(coarse_ogh), static_cast<size_t>(coarse_ngh), static_cast<size_t>(coarse_mgh)};
        size_t const local[3] = {wgsizes[0], wgsizes[1], wgsizes[2]};

        if (!is3d)
        {
            global[0] = static_cast<size_t>(coarse_ogh * coarse_ngh * coarse_mgh);
            global[1] = 1;
            global[2] = 1;
        }

        if (kernelVersion == KernelVersion::THREE_D_8WI_PER_GP)
        {
            // launch for real points only
            global[0] = static_cast<size_t>(coarse.getO() * 8);
            global[1] = static_cast<size_t>(coarse.getN());
            global[2] = static_cast<size_t>(coarse.getM());
        }

        for (int i = 0; i < (is3d ? 3 : 1); i++)
            if (global[i] % local[i] != 0)
                global[i] += local[i] - (global[i] % local[i]);

        cl_event ev;

        // enqueue kernel
        err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel, (is3d ? 3 : 1), NULL, global, local, 0, NULL, &ev);
        mgcl::mgclCheckError(err, "Enqueueing update_ghosts_periodic kernel");

        if (problem.isProfilingEnabled())
        {
            problem.getProfilingData()->addMeasurement(problem.getCommands(), ev, kernelName,
                                                       {global[0], global[1], global[2]},
                                                       {local[0], local[1], local[2]});
        }
        mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        err = clReleaseKernel(kernel);
        mgcl::mgclCheckError(err, "Releasing update_ghosts_periodic kernel");

        return err;
    }

    // Benchs the ghost update of CuboidGpu for different workgroup sizes.
    TEST_CASE("benchProlongation")
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

        std::vector<bench_util::ResultMpi> results;

        int ghosts = 1;
        int periodic = 1;
        std::stringstream kernelProfilesStream;

        for (auto stepsPerIter : CLI_ARGS::jacobiStepsPerIter)
            if (stepsPerIter > ghosts)
                throw "stepsPerIter must be <= ghosts. Not supported yet.";

        // check if mpi is initialized
        int isInitialized = 0;
        MPI_Initialized(&isInitialized);
        REQUIRE(isInitialized);

        MPI_Comm mpi_comm = MPI_COMM_WORLD;

        // check number of processes
        int mpi_size = -1;
        MPI_Comm_size(mpi_comm, &mpi_size);
        // REQUIRE(mpi_size == 8);

        /* MPI variables */
        int mpi_rank;
        int mpi_dims[3] = {0, 0, 0};
        int mpi_periods[3] = {periodic, periodic, periodic};
        int mpi_coords[3];

        /* Initialize cartesian process grid */
        MPI_Comm_size(mpi_comm, &mpi_size);
        MPI_Dims_create(mpi_size, 3, mpi_dims);
        MPI_Cart_create(mpi_comm, 3, mpi_dims, mpi_periods, 1, &mpi_comm);
        MPI_Comm_rank(mpi_comm, &mpi_rank);
        MPI_Cart_coords(mpi_comm, mpi_rank, 3, mpi_coords);

        for (auto gr : gridsTBT)
        {
            int ml = gr[0];
            int nl = gr[1];
            int ol = gr[2];
            int mglob = ml * mpi_dims[0];
            int nglob = nl * mpi_dims[1];
            int oglob = ol * mpi_dims[2];

            CAPTURE(ml, nl, ol, mglob, nglob, oglob);

            // print coords and boundaries per rank
            // if (mpi_rank == 0)
            //     std::cout << "rank;coords[0];coords[1];coords[2];ms;me;ns;ne;os;oe" << std::endl;

            // for (int i = 0; i < mpi_size; i++)
            // {
            //     MPI_Barrier(mpi_comm);
            //     if (mpi_rank == i)
            //     {
            //         std::cout << mpi_rank << ";" << mpi_coords[0] << ";" << mpi_coords[1] << ";" << mpi_coords[2] << ";"
            //                   << m_start << ";" << m_end << ";"
            //                   << n_start << ";" << n_end << ";"
            //                   << o_start << ";" << o_end << std::endl;
            //     }
            // }

            REQUIRE(ml > 0);
            REQUIRE(ml <= mglob);
            REQUIRE(nl > 0);
            REQUIRE(nl <= nglob);
            REQUIRE(ol > 0);
            REQUIRE(ol <= oglob);

            auto v_in = std::make_shared<mgcl::Cuboid>(ml, nl, ol, 0, 0, 0);
            auto f_in = std::make_shared<mgcl::Cuboid>(ml, nl, ol, 0, 0, 0);
            v_in->fill1dIndex(true);
            f_in->fill1dIndex(true);
            // v_in->fillRandom();
            // f_in->fillRandom();

            // Create dummy problem to initialize OpenCL
            mgcl::Problem p(ml, nl, ol, f_in, v_in, mglob, nglob, oglob);
            p.setSilent(true);
            p.setKernelFile("kernel_optimizations.cl");
            if (CLI_ARGS::useBinaryFile)
            {
                p.setBinaryFile("kernel_optimizations.bin");
            }
            p.setUseOpencl(true);
            p.setGhosts(ghosts);
            p.setStencilType(mgcl::MGCL_LAPLACE_7POINT);
            p.setDeviceType(CL_DEVICE_TYPE_GPU);
            p.setDeviceStrategy(mgcl::OCL_DEVICE_STRATEGY::DISTRIBUTE_EVENLY);
            p.setProfilingEnabled(CLI_ARGS::enableKernelProfiling);
            p.setMpiComm(mpi_comm);

            // auto& conf = p.getKernelConfig();
            // // Jacobi kernels
            // conf["jacobi_iter_27point_varying_stencil_1d_update_step_only"] = mgcl::conf::KernelWorkgroupSizes{{1, {32, 1, 1}}};
            p.init();

            if (CLI_ARGS::enableKernelProfiling)
                p.getProfilingData()->getMeasurements().clear();

            auto& lv0 = p.getLevelAt(0);
            auto& lv1 = p.getLevelAt(1);

            auto& fine = lv0.getDVIn();
            auto& coarse = lv1.getDVIn();

            ankerl::nanobench::Bench bench;
            bench.timeUnit(1ms, "ms")
                .epochs(CLI_ARGS::bench_epochs)
                .epochIterations(CLI_ARGS::bench_iterations)
                .relative(false);

            if (mpi_rank > 0)
                bench.output(nullptr);

            if (CLI_ARGS::checkResults)
            {
                bench.epochs(1).epochIterations(1);
            }

            std::unique_ptr<mgcl::Cuboid> fine_3d_prod = nullptr;
            std::unique_ptr<mgcl::Cuboid> fine_3d_1wi_per_gp = nullptr;
            std::unique_ptr<mgcl::Cuboid> fine_3d_8wi_per_gp = nullptr;

            if (CLI_ARGS::checkResults)
            {
                // run code that is in production for result checking
                fine.fill(p.getProgram(), p.getCommands(), 0, false, nullptr, nullptr);
                coarse.fill1dIndex(p.getProgram(), p.getCommands(), true, false, nullptr, nullptr);
                fine_3d_prod = fine.read(p.getCommands(), nullptr, false);
                auto h_coarse = coarse.read(p.getCommands(), nullptr, true);

                mgcl::MultigridEngine::prolongateSeq(lv0, lv1, *fine_3d_prod, *h_coarse);
            }

            // std::vector<std::vector<size_t>> wg_sizes_1d = {{4, 1, 1}, {8, 1, 1}, {32, 1, 1}, {64, 1, 1}, {128, 1, 1}, {256, 1, 1}};
            // for (auto wg : wg_sizes_1d)
            // {
            //     lv0.getDVIn().fill(p.getProgram(), p.getCommands(), 0.0, false, nullptr, nullptr);
            //     lv0.getDVIn().fill1dIndex(p.getProgram(), p.getCommands(), true, true, nullptr, nullptr);

            //     std::string name = std::string("ghost_update_1d_")
            //                            .append(std::to_string(mglob))
            //                            .append("_")
            //                            .append(std::to_string(nglob))
            //                            .append("_")
            //                            .append(std::to_string(oglob))
            //                            .append("_wg")
            //                            .append(std::to_string(wg[0]))
            //                            .append("x")
            //                            .append(std::to_string(wg[1]))
            //                            .append("x")
            //                            .append(std::to_string(wg[2]));

            //     bench.run(std::string(name).c_str(), [&] { //
            //         updateGhosts(p, lv0.getDVIn(), KernelVersion::ONE_D, wg);
            //         p.finish();
            //     });

            //     bench_util::ResultMpi res;
            //     res.name = name;
            //     res.minTime = bench_util::getMinTime(bench, name);
            //     res.medianTime = bench_util::getMedianTime(bench, name);
            //     res.avgTime = bench_util::getAvgTime(bench, name);
            //     res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            //     res.m = ml;
            //     res.n = nl;
            //     res.o = ol;
            //     res.mglob = mglob;
            //     res.nglob = nglob;
            //     res.oglob = oglob;
            //     res.gpus = mpi_size;
            //     res.LT = -1;
            //     results.push_back(res);

            //     // if (CLI_ARGS::checkResults)
            //     // {
            //     //     v_out_default = std::make_unique<mgcl::Cuboid>(ml, nl, ol, ghosts, ghosts, ghosts);
            //     //     lv0.getDVIn().read(p.getCommands(), v_out_default.get(), true);
            //     // }
            // }

            std::vector<std::vector<size_t>> wg_sizes_3d = {{4, 4, 4}, {8, 4, 4}, {8, 8, 4}, {8, 8, 8}, {32, 1, 1}, {64, 1, 1}, {128, 1, 1}};
            // std::vector<std::vector<size_t>> wg_sizes_3d = {{4, 4, 4}};
            for (auto wg : wg_sizes_3d)
            {
                {
                    fine.fill(p.getProgram(), p.getCommands(), 0, false, nullptr, nullptr);
                    coarse.fill1dIndex(p.getProgram(), p.getCommands(), true, false, nullptr, nullptr);

                    std::string name = std::string("prolongate_1wi_per_gp_")
                                           .append(std::to_string(mglob))
                                           .append("_")
                                           .append(std::to_string(nglob))
                                           .append("_")
                                           .append(std::to_string(oglob))
                                           .append("_wg")
                                           .append(std::to_string(wg[0]))
                                           .append("x")
                                           .append(std::to_string(wg[1]))
                                           .append("x")
                                           .append(std::to_string(wg[2]));

                    bench.run(std::string(name).c_str(), [&] { //
                        prolongate(p, fine, coarse, KernelVersion::THREE_D, wg);
                        p.finish();
                    });

                    bench_util::ResultMpi res;
                    res.name = name;
                    res.minTime = bench_util::getMinTime(bench, name);
                    res.medianTime = bench_util::getMedianTime(bench, name);
                    res.avgTime = bench_util::getAvgTime(bench, name);
                    res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                    res.m = ml;
                    res.n = nl;
                    res.o = ol;
                    res.mglob = mglob;
                    res.nglob = nglob;
                    res.oglob = oglob;
                    res.gpus = mpi_size;
                    res.LT = -1;
                    results.push_back(res);

                    if (CLI_ARGS::checkResults)
                    {
                        fine_3d_1wi_per_gp = std::make_unique<mgcl::Cuboid>(ml, nl, ol, ghosts, ghosts, ghosts);
                        lv0.getDVIn().read(p.getCommands(), fine_3d_1wi_per_gp.get(), true);
                    }
                }
            }

            for (auto wg : wg_sizes_3d)
            {
                fine.fill(p.getProgram(), p.getCommands(), 0, false, nullptr, nullptr);
                coarse.fill1dIndex(p.getProgram(), p.getCommands(), true, false, nullptr, nullptr);

                std::string name = std::string("prolongate_8wi_per_gp_")
                                       .append(std::to_string(mglob))
                                       .append("_")
                                       .append(std::to_string(nglob))
                                       .append("_")
                                       .append(std::to_string(oglob))
                                       .append("_wg")
                                       .append(std::to_string(wg[0]))
                                       .append("x")
                                       .append(std::to_string(wg[1]))
                                       .append("x")
                                       .append(std::to_string(wg[2]));

                bench.run(std::string(name).c_str(), [&] { //
                    prolongate(p, fine, coarse, KernelVersion::THREE_D_8WI_PER_GP, wg);
                    p.finish();
                });

                bench_util::ResultMpi res;
                res.name = name;
                res.minTime = bench_util::getMinTime(bench, name);
                res.medianTime = bench_util::getMedianTime(bench, name);
                res.avgTime = bench_util::getAvgTime(bench, name);
                res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                res.m = ml;
                res.n = nl;
                res.o = ol;
                res.mglob = mglob;
                res.nglob = nglob;
                res.oglob = oglob;
                res.gpus = mpi_size;
                res.LT = -1;
                results.push_back(res);

                if (CLI_ARGS::checkResults)
                {
                    fine_3d_8wi_per_gp = std::make_unique<mgcl::Cuboid>(ml, nl, ol, ghosts, ghosts, ghosts);
                    lv0.getDVIn().read(p.getCommands(), fine_3d_8wi_per_gp.get(), true);
                }
            }

            // Check results for kernels that it is valid for
            if (CLI_ARGS::checkResults)
            {
                fine_3d_prod->dumpToFile("fine3dprod.txt");
                // fine_3d_1wi_per_gp->dumpToFile("fine3d1wi_per_gp.txt");
                fine_3d_8wi_per_gp->dumpToFile("fine3d8wi_per_gp.txt");

                REQUIRE(fine_3d_prod->isEqual(*fine_3d_1wi_per_gp));
                REQUIRE(fine_3d_prod->isEqual(*fine_3d_8wi_per_gp));
            }

            if (CLI_ARGS::enableKernelProfiling)
            {
                // p.getProfilingData()->printBestTimingsPerKernel(kernelProfilesStream);
                p.getProfilingData()->printBestTimingsPerKernelAsCsv(kernelProfilesStream);
            }

            MPI_Barrier(mpi_comm);
            bench_util::printCsvFormat(results, mpi_comm, mpi_rank);
            MPI_Barrier(mpi_comm);

            if (CLI_ARGS::enableKernelProfiling)
            {
                kernelProfilesStream << "rank: " << mpi_rank << std::endl;
                std::cout << kernelProfilesStream.str() << std::endl;
            }
            MPI_Barrier(mpi_comm);
        }
    }
}