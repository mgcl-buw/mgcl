#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"

#include <CL/cl.h>
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <vector>
using namespace std::chrono_literals;

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/mpi_util.hpp"
#include "../src/mgcl/multigrid_engine.hpp"
#include "../src/mgcl/opencl_helper.hpp"
#include "../src/mgcl/problem.hpp"
#include "bench_util.hpp"
#include "cli_args.hpp"

// Runs ghost update using MPI and OCL, as it happens between Jacobi iterations. I.e. when using MPI,
// the gpu buffer is first read, then ghosts are updated sequentially, then the data is copied back to the gpu.
// Timings will be collected per node and printed by rank at the end.
// Run with e.g.: mpiexec -n 4 benchmarks bench_ghost_update_mpi_ocl
TEST_CASE("bench_ghost_update_mpi_ocl")
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

    // Check if mpi is initialized
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REQUIRE(isInitialized);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    // Check number of processes
    int mpi_size = -1;
    MPI_Comm_size(mpi_comm, &mpi_size);
    // REQUIRE(mpi_size == 1);

    int periodic = 1;

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

    if (mpi_rank == 0)
    {
        std::cout << "Testing the following grid sizes" << std::endl;
        for (auto gr : gridsTBT)
        {
            int m = gr[0];
            int n = gr[1];
            int o = gr[2];
            std::cout << "  local size: " << m << "," << n << "," << o << ", global size: "
                      << m * mpi_dims[0] << "," << n * mpi_dims[1] << "," << o * mpi_dims[2] << std::endl;
        }
    }
    MPI_Barrier(mpi_comm);

    int maxGhosts = 3;
    std::vector<bench_util::ResultGhostUpdateMpi> results;

    bool printedGpu = false;
    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];
        int mglob = m * mpi_dims[0];
        int nglob = n * mpi_dims[1];
        int oglob = o * mpi_dims[2];
        double hm = 1.0 / (double)mglob;
        double hn = 1.0 / (double)nglob;
        double ho = 1.0 / (double)oglob;

        mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
        mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_7POINT;

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ms, "ms")
            .epochs(CLI_ARGS::bench_epochs)
            .epochIterations(CLI_ARGS::bench_iterations)
            // .minEpochTime(100ms)
            .relative(CLI_ARGS::jacobiIters.size() > 1);

        if (mpi_rank > 0)
            bench.output(nullptr);

        for (int ghosts = 1; ghosts <= maxGhosts; ghosts++)
        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            auto r = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            v->fillRandom();
            f->fillRandom();

            auto p = std::make_shared<mgcl::Problem>(m, n, o, f, v, mglob, nglob, oglob);
            p->setMpiComm(mpi_comm);
            p->setGhosts(ghosts);
            p->setGhostsIn(ghosts);
            p->setUseOpencl(true);
            p->setDeviceType(CL_DEVICE_TYPE_GPU);
            p->setSilent(true);
            if (p->getStencilType() == mgcl::MGCL_VARYING)
                p->getStencilValues()->fillRandom();
            p->init();

            auto& buf = p->getLevelAt(0).getDVIn();
            auto h_buf = buf.read(p->getCommands(), nullptr, true);
            auto mpiLevelData = p->getLevelAt(0).getMpiDataPtr();

            if (!printedGpu)
            {
                for (int i = 0; i < mpi_size; i++)
                {
                    MPI_Barrier(mpi_comm);
                    if (i == mpi_rank)
                    {
                        std::cout << "on rank " << mpi_rank << ", GPU info: ";
                        p->getOpenCLHelper().outputDeviceInfo();
                    }
                }
                printedGpu = true;
            }

            {
                std::string name = std::string("ghupdate_ocl_mpi_planes_N_")
                                       .append(std::to_string(m))
                                       .append("_")
                                       .append(std::to_string(n))
                                       .append("_")
                                       .append(std::to_string(o))
                                       .append("_gh")
                                       .append(std::to_string(ghosts));
                bench.run(std::string(name).c_str(), [&] { //
                    MPI_Barrier(mpi_comm);
                    mgcl::MultigridEngine::updateGhosts(*p, buf, mpiLevelData, false);
                    p->getOpenCLHelper().finish();
                    MPI_Barrier(mpi_comm);
                });

                // std::cout << "rank " << mpi_rank << " done" << std::endl;
                MPI_Barrier(mpi_comm);

                bench_util::ResultGhostUpdateMpi res;
                res.name = name;
                res.minTime = bench_util::getMinTime(bench, name);
                res.medianTime = bench_util::getMedianTime(bench, name);
                res.avgTime = bench_util::getAvgTime(bench, name);
                res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                res.mloc = m;
                res.nloc = n;
                res.oloc = o;
                res.mglob = mglob;
                res.nglob = nglob;
                res.oglob = oglob;
                res.ghosts = ghosts;
                res.gpus = mpi_size;
                results.push_back(res);
            }

            {
                // old ghost update copying the entire grid to host
                std::string name = std::string("ghupdate_ocl_mpi_entiregrid_N_")
                                       .append(std::to_string(m))
                                       .append("_")
                                       .append(std::to_string(n))
                                       .append("_")
                                       .append(std::to_string(o))
                                       .append("_gh")
                                       .append(std::to_string(ghosts));
                bench.run(std::string(name).c_str(), [&] { //
                    MPI_Barrier(mpi_comm);
                    buf.read(p->getCommands(), h_buf.get(), true);
                    mgcl::MultigridEngine::updateGhostsSeq(*h_buf, mpiLevelData, periodic, false);
                    buf.write(p->getCommands(), *h_buf, true);
                    p->getOpenCLHelper().finish();
                    MPI_Barrier(mpi_comm);
                });

                // std::cout << "rank " << mpi_rank << " done" << std::endl;
                MPI_Barrier(mpi_comm);

                bench_util::ResultGhostUpdateMpi res;
                res.name = name;
                res.minTime = bench_util::getMinTime(bench, name);
                res.medianTime = bench_util::getMedianTime(bench, name);
                res.avgTime = bench_util::getAvgTime(bench, name);
                res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                res.mloc = m;
                res.nloc = n;
                res.oloc = o;
                res.mglob = mglob;
                res.nglob = nglob;
                res.oglob = oglob;
                res.ghosts = ghosts;
                res.gpus = mpi_size;
                results.push_back(res);
            }
        }
    }

    bench_util::printCsvFormat(results, mpi_comm, mpi_rank);

    MPI_Barrier(mpi_comm);
}

namespace mgclBenchGhostUpdateSplitFused
{
    using namespace mgcl;
    struct Args
    {
        CuboidGpu& d_buf;                    // buffer to update ghosts of
        BufferGpu& dPlanesBuf;               // GPU buffer that holds the border planes
        std::vector<double>& hPlanesBufSend; // host buffer for sending data via MPI
        std::vector<double>& hPlanesBufRecv; // host buffer for receiving data via MPI

        cl_program program;
        cl_command_queue queue;
        cl_context context;

        MPILevelData& mpiData;

        conf::KernelConfig* kernelConfig;
        ProfilingData* pd;
    };

    void sendBorderPlanesInterleaved(int mgh, int ngh, int ogh, int ghosts_m, int ghosts_n, int ghosts_o,
                                     int stencilWidth, BufferGpu& dPlanesBuf,
                                     cl_command_queue queue,
                                     std::vector<double>& sbuf, std::vector<double>& rbuf, MPILevelData& mpiData)
    {
        // Sizes of planes
        int yz = ngh * ogh;
        int xz = mgh * ogh;
        int xy = mgh * ngh;

        // Size of planes times amount of ghosts in that direction, i.e. number of grid points that are sent in that
        // direction
        int yzgh = yz * ghosts_m;
        int xzgh = xz * ghosts_n;
        int xygh = xy * ghosts_o;

        int stencilSize = stencilWidth * stencilWidth * stencilWidth;

        int m = mgh - 2 * ghosts_m;
        int n = ngh - 2 * ghosts_n;
        int o = ogh - 2 * ghosts_o;

        // int base_yz_front = 0;
        int base_yz_back = yzgh * stencilSize;
        int base_xz_top = (2 * yzgh) * stencilSize;
        int base_xz_bottom = (2 * yzgh + xzgh) * stencilSize;
        int base_xy_left = (2 * yzgh + 2 * xzgh) * stencilSize;
        int base_xy_right = (2 * yzgh + 2 * xzgh + xygh) * stencilSize;

        int numElementsFrontBack = yzgh * stencilSize * 2;
        int numElementsTopBottom = xzgh * stencilSize * 2;
        int numElementsLeftRight = xygh * stencilSize * 2;

        // Send planes to neighbors
        int myid, err;
        MPI_Comm_rank(mpiData.comm, &myid);

        auto sbuf_raw = sbuf.data();

        cl_event evRead1;
        cl_event evRead2;
        cl_event evRead3;

        // read front and back, wait for it and enqueue read of top and bottom
        err = clEnqueueReadBuffer(queue, dPlanesBuf.getBuf(), CL_FALSE, 0, sizeof(double) * numElementsFrontBack, sbuf_raw, 0, nullptr, &evRead1);
        mgclCheckError(err, "clEnqueueReadBuffer 1");
        err = clEnqueueReadBuffer(queue, dPlanesBuf.getBuf(), CL_FALSE, sizeof(double) * base_xz_top, sizeof(double) * numElementsTopBottom, &sbuf_raw[base_xz_top], 0, nullptr, &evRead2);
        mgclCheckError(err, "clEnqueueReadBuffer 2");
        err = clEnqueueReadBuffer(queue, dPlanesBuf.getBuf(), CL_FALSE, sizeof(double) * base_xy_left, sizeof(double) * numElementsLeftRight, &sbuf_raw[base_xy_left], 0, nullptr, &evRead3);
        mgclCheckError(err, "clEnqueueReadBuffer 3");
        // mgclCheckError(clFinish(queue), "clFinish");

        // halt execution until first read is finished
        mgclCheckError(clWaitForEvents(1, &evRead1), "clWaitForEvents evRead1");

        // Send front planes to the back
        err = MPI_Sendrecv(static_cast<void*>(sbuf.data()), yzgh * stencilSize, MPI_DOUBLE, mpiData.back[0], 0,
                           static_cast<void*>(rbuf.data()), yzgh * stencilSize, MPI_DOUBLE, mpiData.front[0], 0,
                           mpiData.comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");

        // Send back planes to the front
        err = MPI_Sendrecv(static_cast<void*>(&(sbuf[base_yz_back])), yzgh * stencilSize, MPI_DOUBLE, mpiData.front[0], 0,
                           static_cast<void*>(&(rbuf[base_yz_back])), yzgh * stencilSize, MPI_DOUBLE, mpiData.back[0], 0,
                           mpiData.comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");

        // Asynchronously copy front and back to device again
        err = clEnqueueWriteBuffer(queue, dPlanesBuf.getBuf(), CL_FALSE, 0, sizeof(double) * numElementsFrontBack, sbuf_raw, 0, nullptr, nullptr);
        mgclCheckError(err, "clEnqueueWriteBuffer 1");

        // halt execution until second read is finished
        mgclCheckError(clWaitForEvents(1, &evRead2), "clWaitForEvents evRead2");

        // Write received edges of cuboid to send buffer
        // i0: i index of recv buffers for yz plane (always all i indices)
        // i1: i index of send buffers xz back ghosts
        // j0: j index of send buffers for xz plane (always all j indices)
        // j1: j index of recv buffer yz top edge
        // j2: j index of recv buffer yz bottom edge
        for (int st = 0; st < stencilSize; st++)
            for (int i0 = 0, i1 = m + ghosts_m;
                 i0 < ghosts_m;
                 i0++, i1++)
                for (int j0 = 0, j1 = ghosts_n, j2 = n;
                     j0 < ghosts_n;
                     j0++, j1++, j2++)
                    for (int k = 0; k < ogh; k++)
                    {
                        // Upper front edge - Write ghosts in the front (from back recv buffer) to xz top send buffer
                        sbuf[st * xzgh + base_xz_top + j0 * xz + i0 * ogh + k] = rbuf[st * yzgh + base_yz_back + i0 * yz + j1 * ogh + k];

                        // Lower front edge - Write ghosts in the front (from back recv buffer) to xz bottom send buffer
                        sbuf[st * xzgh + base_xz_bottom + j0 * xz + i0 * ogh + k] = rbuf[st * yzgh + base_yz_back + i0 * yz + j2 * ogh + k];

                        // Upper back edge - Write ghosts in the back (from front recv buffer, base 0) to xz top send buffer
                        sbuf[st * xzgh + base_xz_top + j0 * xz + i1 * ogh + k] = rbuf[st * yzgh + i0 * yz + j1 * ogh + k];

                        // Lower back edge - Write ghosts in the back (from front recv buffer, base 0) to xz bottom send buffer
                        sbuf[st * xzgh + base_xz_bottom + j0 * xz + i1 * ogh + k] = rbuf[st * yzgh + i0 * yz + j2 * ogh + k];
                    }

        // Send top planes to the bottom
        err = MPI_Sendrecv(static_cast<void*>(&(sbuf[base_xz_top])), xzgh * stencilSize, MPI_DOUBLE, mpiData.down[0], 0,
                           static_cast<void*>(&(rbuf[base_xz_top])), xzgh * stencilSize, MPI_DOUBLE, mpiData.up[0], 0,
                           mpiData.comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");

        // Send bottom planes to the top
        err = MPI_Sendrecv(static_cast<void*>(&(sbuf[base_xz_bottom])), xzgh * stencilSize, MPI_DOUBLE, mpiData.up[0], 0,
                           static_cast<void*>(&(rbuf[base_xz_bottom])), xzgh * stencilSize, MPI_DOUBLE, mpiData.down[0], 0,
                           mpiData.comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");

        // Asynchronously copy top and bottom to device again
        err = clEnqueueWriteBuffer(queue, dPlanesBuf.getBuf(), CL_FALSE, sizeof(double) * base_xz_top, sizeof(double) * numElementsTopBottom, &sbuf_raw[base_xz_top], 0, nullptr, &evRead2);
        mgclCheckError(err, "clEnqueueWriteBuffer 2");

        // halt execution until last read is finished
        mgclCheckError(clWaitForEvents(1, &evRead3), "clWaitForEvents evRead3");

        // Write received left torus of cuboid to send buffer
        // k0: k index of send buffers for xy planes (left and right)
        // k1: k index of recv buffers for copy into left send buffer
        // k2: k index of recv buffers for copy into right send buffer
        for (int st = 0; st < stencilSize; st++)
            for (int k0 = 0, k1 = ghosts_o, k2 = o;
                 k0 < ghosts_o;
                 k0++, k1++, k2++)
            {

                // Copying from yz planes (front back)
                // i0: i index of recv buffers for yz plane (always all i indices)
                // i1: i index of send buffers xz back ghosts
                for (int i0 = 0, i1 = m + ghosts_m;
                     i0 < ghosts_m;
                     i0++, i1++)
                    for (int j = ghosts_n; j < ghosts_n + n; j++)
                    {
                        // Left front face - Write ghosts in the send left buffer from recv back buffer
                        sbuf[st * xygh + base_xy_left + k0 * xy + i0 * ngh + j] = rbuf[st * yzgh + base_yz_back + i0 * yz + j * ogh + k1];

                        // Left back face - Write ghosts in the send left buffer from recv front buffer
                        sbuf[st * xygh + base_xy_left + k0 * xy + i1 * ngh + j] = rbuf[st * yzgh + i0 * yz + j * ogh + k1];

                        // Right front face - Write ghosts in the send right buffer from recv back buffer
                        sbuf[st * xygh + base_xy_right + k0 * xy + i0 * ngh + j] = rbuf[st * yzgh + base_yz_back + i0 * yz + j * ogh + k2];

                        // Right back face - Write ghosts in the send right buffer from recv front buffer
                        sbuf[st * xygh + base_xy_right + k0 * xy + i1 * ngh + j] = rbuf[st * yzgh + i0 * yz + j * ogh + k2];
                    }

                // Copying from xz planes (top bottom)
                // j0: j index of recv buffers yz bottom and send both left and right
                // j1: j index of send buffers xy bottom ghosts (recv top)
                for (int i = 0; i < mgh; i++)
                    for (int j0 = 0, j1 = n + ghosts_n;
                         j0 < ghosts_n;
                         j0++, j1++)
                    {
                        // Left top edge - Write ghosts in the send left buffer from recv bottom buffer
                        sbuf[st * xygh + base_xy_left + k0 * xy + i * ngh + j0] = rbuf[st * xzgh + base_xz_bottom + j0 * xz + i * ogh + k1];

                        // Left bottom edge - Write ghosts in the send left buffer from recv top buffer
                        sbuf[st * xygh + base_xy_left + k0 * xy + i * ngh + j1] = rbuf[st * xzgh + base_xz_top + j0 * xz + i * ogh + k1];

                        // Right top edge - Write ghosts in the send left buffer from recv bottom buffer
                        sbuf[st * xygh + base_xy_right + k0 * xy + i * ngh + j0] = rbuf[st * xzgh + base_xz_bottom + j0 * xz + i * ogh + k2];

                        // Right bottom face - Write ghosts in the send right buffer from recv top buffer
                        sbuf[st * xygh + base_xy_right + k0 * xy + i * ngh + j1] = rbuf[st * xzgh + base_xz_top + j0 * xz + i * ogh + k2];
                    }
            }

        // Send left planes to the right
        err = MPI_Sendrecv(static_cast<void*>(&(sbuf[base_xy_left])), xygh * stencilSize, MPI_DOUBLE, mpiData.right[0], 0,
                           static_cast<void*>(&(rbuf[base_xy_left])), xygh * stencilSize, MPI_DOUBLE, mpiData.left[0], 0,
                           mpiData.comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");

        // Send right planes to the left
        err = MPI_Sendrecv(static_cast<void*>(&(sbuf[base_xy_right])), xygh * stencilSize, MPI_DOUBLE, mpiData.left[0], 0,
                           static_cast<void*>(&(rbuf[base_xy_right])), xygh * stencilSize, MPI_DOUBLE, mpiData.right[0], 0,
                           mpiData.comm, MPI_STATUS_IGNORE);
        mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");

        err = clEnqueueWriteBuffer(queue, dPlanesBuf.getBuf(), CL_FALSE, sizeof(double) * base_xy_left, sizeof(double) * numElementsLeftRight, &sbuf_raw[base_xy_left], 0, nullptr, &evRead3);
        mgclCheckError(err, "clEnqueueWriteBuffer 3");

        clReleaseEvent(evRead1);
        clReleaseEvent(evRead2);
        clReleaseEvent(evRead3);
    }

    void extractBorderPlanesWithoutRead(CuboidGpu& c,
                                        cl_command_queue commands, cl_program program,
                                        BufferGpu& d_target,
                                        mgcl::conf::KernelConfig* conf, mgcl::ProfilingData* pd)

    {
        int m = c.getM();
        int n = c.getN();
        int o = c.getO();
        int mgh = c.getMgh();
        int ngh = c.getNgh();
        int ogh = c.getOgh();
        int ghosts_m = c.getGhostsM();
        int ghosts_n = c.getGhostsN();
        int ghosts_o = c.getGhostsO();
        cl_context context = c.getContext();
        auto buffer = c.getBuffer();

        // Plane sizes
        int yz = ngh * ogh;
        int xz = mgh * ogh;
        int xy = mgh * ngh;
        int ressize = 2 * yz * ghosts_m + 2 * xz * ghosts_n + 2 * xy * ghosts_o;

        if (ghosts_m > m || ghosts_n > n || ghosts_o > o)
            error("CuboidGpu::extractBorderPlanes: Only defined for ghosts <= m, n, o");

        int err;

        // Create the compute kernel from the program
        const char* kernelName = "extract_border_planes";
        cl_kernel kernel = clCreateKernel(program, kernelName, &err);
        mgcl::mgclCheckError(err, "clCreateKernel");

        // assign kernel arguments
        cl_mem d_target_buffer = d_target.getBuf();
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &buffer);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &d_target_buffer);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_o);
        mgcl::mgclCheckError(err, "Setting kernel arguments");

        // one work-item per ghost cell (excluding real cells). Pad global sizes to fit to local sizes
        size_t global = ressize;
        size_t local = 32;
        // Apply kernel config, if available
        if (conf)
        {
            const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(*conf, kernelName, global);
            local = c[0];
        }

        if (global % local != 0)
            global += local - (global % local);

        cl_event ev;

        // enqueue kernel
        err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, &ev);
        mgcl::mgclCheckError(err, "Enqueueing extract_border_planes kernel");

        if (pd != nullptr)
        {
            pd->addMeasurement(commands, ev, kernelName,
                               {global, 0, 0},
                               {local, 1, 1});
        }
        mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        // mgcl::mgclCheckError(clFinish(commands), "clFinish");

        err = clReleaseKernel(kernel);
        mgcl::mgclCheckError(err, "Releasing extract_border_planes kernel");

        // Read into h_target
        // d_target->read(commands, retraw->data(), true, ressize, pd);

        // return h_target;
    }

    // void updateGhostsOclMpi(Problem& p, CuboidGpu& d_buf, MPILevelData& mpiData,
    //                         bool periodic, bool forceLocal)
    void updateGhostsOclMpi(Args& args)
    {
        // Read back from GPU and update ghosts on host in order to update neighbouring nodes, too.
        // auto tmp = d_buf.read(commands, nullptr, true);
        // MultigridEngine::updateGhostsSeq(*tmp, &mpiData, periodic, forceLocal);
        // d_buf.write(commands, *tmp, true);

        auto& d_buf = args.d_buf;

        // Use temporary buffer for extracting and pasting planes. Check if it's large enough beforehand.
        // TODO maybe disable check in UNSAFE mode
        int yz = d_buf.getNgh() * d_buf.getOgh();
        int xz = d_buf.getMgh() * d_buf.getOgh();
        int xy = d_buf.getMgh() * d_buf.getNgh();
        int ressize = 2 * yz * d_buf.getGhostsM() + 2 * xz * d_buf.getGhostsN() + 2 * xy * d_buf.getGhostsO();

        auto& dPlanesBuf = args.dPlanesBuf;
        if (dPlanesBuf.getSize() < ressize)
            error("MultigridEngine::updateGhostsOclMpi: dPlanesBuf is too small. Need at least " + std::to_string(ressize) + ", but is " + std::to_string(dPlanesBuf.getSize()));

        auto& hPlanesBufSend = args.hPlanesBufSend;
        auto& hPlanesBufRecv = args.hPlanesBufRecv;
        if (hPlanesBufSend.size() < ressize || hPlanesBufRecv.size() < ressize)
            throw "MultigridEngine::updateGhostsOclMpi: hPlanesBufSend or hPlanesBufRecv is too small. Need at least " +
                std::to_string(ressize) + ", but is " + std::to_string(hPlanesBufSend.size()) +
                " (send) and " + std::to_string(hPlanesBufRecv.size()) + " (recv)";

        // Extract border planes from the buffer but don't read yet
        extractBorderPlanesWithoutRead(d_buf, args.queue, args.program,
                                       dPlanesBuf,
                                       args.kernelConfig, args.pd);

        // Read planes from device, send our planes to neighbours and receive their planes and write back to device
        sendBorderPlanesInterleaved(d_buf.getMgh(), d_buf.getNgh(), d_buf.getOgh(),
                                    d_buf.getGhostsM(), d_buf.getGhostsN(), d_buf.getGhostsO(), 1,
                                    dPlanesBuf, args.queue,
                                    args.hPlanesBufSend, args.hPlanesBufRecv, args.mpiData);

        // dPlanesBuf.write(args.queue, hPlanesBufRecv, false, ressize, args.pd);
        d_buf.pasteGhostsFromBorderPlanes(args.context, args.queue, args.program,
                                          &dPlanesBuf, nullptr,
                                          args.kernelConfig, args.pd);
    }

}

// Runs ghost update using MPI and OCL, as it happens between Jacobi iterations.
// Compares version a: reading entire buffer from device to host, send via MPI and write back to device vs.
// b: interleaving copies between device and host with MPI transfer
// Run with e.g.: mpiexec -n 4 benchmarks bench_ghost_update_mpi_ocl
TEST_CASE("benchGhostUpdateMpiOclWholeVsInterleaved")
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

    // Check if mpi is initialized
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REQUIRE(isInitialized);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    // Check number of processes
    int mpi_size = -1;
    MPI_Comm_size(mpi_comm, &mpi_size);
    REQUIRE(mpi_size > 1);

    int periodic = 1;

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

    if (mpi_rank == 0)
    {
        std::cout << "Testing the following grid sizes" << std::endl;
        for (auto gr : gridsTBT)
        {
            int m = gr[0];
            int n = gr[1];
            int o = gr[2];
            std::cout << "  local size: " << m << "," << n << "," << o << ", global size: "
                      << m * mpi_dims[0] << "," << n * mpi_dims[1] << "," << o * mpi_dims[2] << std::endl;
        }
    }
    MPI_Barrier(mpi_comm);

    int maxGhosts = 3;
    std::vector<bench_util::ResultGhostUpdateMpi> results;

    bool printedGpu = false;
    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];
        int mglob = m * mpi_dims[0];
        int nglob = n * mpi_dims[1];
        int oglob = o * mpi_dims[2];
        double hm = 1.0 / (double)mglob;
        double hn = 1.0 / (double)nglob;
        double ho = 1.0 / (double)oglob;

        mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
        mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_7POINT;

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ms, "ms")
            .epochs(CLI_ARGS::bench_epochs)
            .epochIterations(CLI_ARGS::bench_iterations)
            // .minEpochTime(100ms)
            .relative(CLI_ARGS::jacobiIters.size() > 1);

        if (mpi_rank > 0)
            bench.output(nullptr);

        for (int ghosts = 1; ghosts <= maxGhosts; ghosts++)
        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            auto r = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            v->fillRandom();
            f->fillRandom();

            auto p = std::make_shared<mgcl::Problem>(m, n, o, f, v, mglob, nglob, oglob);
            p->setMpiComm(mpi_comm);
            p->setGhosts(ghosts);
            p->setGhostsIn(ghosts);
            p->setUseOpencl(true);
            p->setDeviceType(CL_DEVICE_TYPE_GPU);
            p->setSilent(true);
            if (p->getStencilType() == mgcl::MGCL_VARYING)
                p->getStencilValues()->fillRandom();
            p->init();

            auto& buf = p->getLevelAt(0).getDVIn();
            auto h_buf = buf.read(p->getCommands(), nullptr, true);
            auto mpiLevelData = p->getLevelAt(0).getMpiDataPtr();

            if (!printedGpu)
            {
                for (int i = 0; i < mpi_size; i++)
                {
                    MPI_Barrier(mpi_comm);
                    if (i == mpi_rank)
                    {
                        std::cout << "on rank " << mpi_rank << ", GPU info: ";
                        p->getOpenCLHelper().outputDeviceInfo();
                    }
                }
                printedGpu = true;
            }

            {
                std::string name = std::string("ghupdate_ocl_mpi_fused_N")
                                       .append(std::to_string(m))
                                       .append("_")
                                       .append(std::to_string(n))
                                       .append("_")
                                       .append(std::to_string(o))
                                       .append("_gh")
                                       .append(std::to_string(ghosts));
                bench.run(std::string(name).c_str(), [&] { //
                    MPI_Barrier(mpi_comm);
                    mgcl::MultigridEngine::updateGhosts(*p, buf, mpiLevelData, false);
                    p->getOpenCLHelper().finish();
                    MPI_Barrier(mpi_comm);
                });

                // std::cout << "rank " << mpi_rank << " done" << std::endl;
                MPI_Barrier(mpi_comm);

                bench_util::ResultGhostUpdateMpi res;
                res.name = name;
                res.minTime = bench_util::getMinTime(bench, name);
                res.medianTime = bench_util::getMedianTime(bench, name);
                res.avgTime = bench_util::getAvgTime(bench, name);
                res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                res.mloc = m;
                res.nloc = n;
                res.oloc = o;
                res.mglob = mglob;
                res.nglob = nglob;
                res.oglob = oglob;
                res.ghosts = ghosts;
                res.gpus = mpi_size;
                results.push_back(res);
            }

            {
                // {
                //     CuboidGpu& d_buf;                    // buffer to update ghosts of
                //     BufferGpu& dPlanesBuf;               // GPU buffer that holds the border planes
                //     std::vector<double>& hPlanesBufSend; // host buffer for sending data via MPI
                //     std::vector<double>& hPlanesBufRecv; // host buffer for receiving data via MPI

                //     cl_program program;
                //     cl_command_queue queue;
                //     cl_context context;

                //     MPILevelData& mpiData;

                //     conf::KernelConfig* kernelConfig;
                //     ProfilingData* pd;
                // };
                mgclBenchGhostUpdateSplitFused::Args args{
                    buf,
                    p->getDPlanesBuf(),
                    p->getHPlanesBufSend(),
                    p->getHPlanesBufRecv(),
                    p->getProgram(),
                    p->getCommands(),
                    p->getContext(),
                    *mpiLevelData,
                    &p->getKernelConfig(),
                    p->getProfilingData()

                };
                std::string name = std::string("ghupdate_ocl_mpi_interleaved_N")
                                       .append(std::to_string(m))
                                       .append("_")
                                       .append(std::to_string(n))
                                       .append("_")
                                       .append(std::to_string(o))
                                       .append("_gh")
                                       .append(std::to_string(ghosts));
                bench.run(std::string(name).c_str(), [&] { //
                    MPI_Barrier(mpi_comm);
                    mgclBenchGhostUpdateSplitFused::updateGhostsOclMpi(args);
                    p->getOpenCLHelper().finish();
                    MPI_Barrier(mpi_comm);
                });

                // std::cout << "rank " << mpi_rank << " done" << std::endl;
                MPI_Barrier(mpi_comm);

                bench_util::ResultGhostUpdateMpi res;
                res.name = name;
                res.minTime = bench_util::getMinTime(bench, name);
                res.medianTime = bench_util::getMedianTime(bench, name);
                res.avgTime = bench_util::getAvgTime(bench, name);
                res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                res.mloc = m;
                res.nloc = n;
                res.oloc = o;
                res.mglob = mglob;
                res.nglob = nglob;
                res.oglob = oglob;
                res.ghosts = ghosts;
                res.gpus = mpi_size;
                results.push_back(res);
            }
        }
    }

    bench_util::printCsvFormat(results, mpi_comm, mpi_rank);

    MPI_Barrier(mpi_comm);
}

// Runs ghost update using MPI, as it happens between Jacobi iterations. I.e. when using MPI,
// Timings will be collected per node and printed by rank at the end.
// Run with e.g.: mpiexec -n 4 benchmarks bench_ghost_update_mpi_seq
TEST_CASE("bench_ghost_update_mpi_seq")
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

    // Check if mpi is initialized
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REQUIRE(isInitialized);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    // Check number of processes
    int mpi_size = -1;
    MPI_Comm_size(mpi_comm, &mpi_size);
    // REQUIRE(mpi_size == 1);

    int periodic = 1;

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

    if (mpi_rank == 0)
    {
        std::cout << "Testing the following grid sizes" << std::endl;
        for (auto gr : gridsTBT)
        {
            int m = gr[0];
            int n = gr[1];
            int o = gr[2];
            std::cout << "  local size: " << m << "," << n << "," << o << ", global size: "
                      << m * mpi_dims[0] << "," << n * mpi_dims[1] << "," << o * mpi_dims[2] << std::endl;
        }
    }
    MPI_Barrier(mpi_comm);

    int maxGhosts = 3;
    std::vector<bench_util::ResultGhostUpdateMpi> results;

    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];
        int mglob = m * mpi_dims[0];
        int nglob = n * mpi_dims[1];
        int oglob = o * mpi_dims[2];
        double hm = 1.0 / (double)mglob;
        double hn = 1.0 / (double)nglob;
        double ho = 1.0 / (double)oglob;

        mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
        mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_7POINT;

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ms, "ms")
            .epochs(CLI_ARGS::bench_epochs)
            .epochIterations(CLI_ARGS::bench_iterations)
            // .minEpochTime(100ms)
            .relative(CLI_ARGS::jacobiIters.size() > 1);

        for (int ghosts = 1; ghosts <= maxGhosts; ghosts++)
        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            auto r = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            v->fillRandom();
            f->fillRandom();

            auto p = std::make_shared<mgcl::Problem>(m, n, o, f, v, mglob, nglob, oglob);
            p->setMpiComm(mpi_comm);
            p->setGhosts(ghosts);
            p->setGhostsIn(ghosts);
            p->setUseOpencl(false);
            p->setDeviceType(CL_DEVICE_TYPE_GPU);
            p->setSilent(true);
            if (p->getStencilType() == mgcl::MGCL_VARYING)
                p->getStencilValues()->fillRandom();
            p->init();

            auto& buf = p->getLevelAt(0).getV();
            auto mpiLevelData = p->getLevelAt(0).getMpiDataPtr();

            std::string name = std::string("seq_mpi_N")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o))
                                   .append("_gh")
                                   .append(std::to_string(ghosts));
            bench.run(std::string(name).c_str(), [&] { //
                MPI_Barrier(mpi_comm);
                mgcl::MultigridEngine::updateGhostsSeq(buf, mpiLevelData, true, false);
                MPI_Barrier(mpi_comm);
            });

            // std::cout << "rank " << mpi_rank << " done" << std::endl;
            MPI_Barrier(mpi_comm);

            bench_util::ResultGhostUpdateMpi res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.mloc = m;
            res.nloc = n;
            res.oloc = o;
            res.mglob = mglob;
            res.nglob = nglob;
            res.oglob = oglob;
            res.ghosts = ghosts;
            res.gpus = mpi_size;
            results.push_back(res);
        }
    }

    bench_util::printCsvFormat(results, mpi_comm, mpi_rank);

    MPI_Barrier(mpi_comm);
}

// Tests the method mgcl::Cuboid::slice for all 3 directions, which is needed when updating ghosts.
// This test can be run without MPI.
// The argument --grids ... is required.
TEST_CASE("bench_cuboid_slice")
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

    // Check if mpi is initialized
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REQUIRE(isInitialized);

    /* MPI variables */
    int mpi_rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    if (mpi_rank == 0)
    {
        std::cout << "Testing the following grid sizes" << std::endl;
        for (auto gr : gridsTBT)
        {
            int m = gr[0];
            int n = gr[1];
            int o = gr[2];
            std::cout << "  " << m << "," << n << "," << o << std::endl;
        }
    }
    else
    {
        return;
    }

    std::vector<bench_util::Result> results;

    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];

        mgcl::Cuboid c(m, n, o);
        int gh = 1;

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ms, "ms")
            .epochs(CLI_ARGS::bench_epochs)
            .epochIterations(CLI_ARGS::bench_iterations);

        {
            std::string name = std::string("slice_xdir_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o));
            bench.run(std::string(name).c_str(), [&] { //
                ankerl::nanobench::doNotOptimizeAway(c.slice(gh, 2 * gh - 1, 0, n - 1, 0, o - 1));
            });

            bench_util::Result res;
            res.name = name;
            res.m = m;
            res.n = n;
            res.o = o;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            results.push_back(res);
        }

        {
            std::string name = std::string("slice_ydir_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o));
            bench.run(std::string(name).c_str(), [&] { //
                ankerl::nanobench::doNotOptimizeAway(c.slice(0, m - 1, gh, 2 * gh - 1, 0, o - 1));
            });

            bench_util::Result res;
            res.name = name;
            res.m = m;
            res.n = n;
            res.o = o;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            results.push_back(res);
        }

        {
            std::string name = std::string("slice_zdir_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o));
            bench.run(std::string(name).c_str(), [&] { //
                ankerl::nanobench::doNotOptimizeAway(c.slice(0, m - 1, 0, n - 1, gh, 2 * gh - 1));
            });

            bench_util::Result res;
            res.name = name;
            res.m = m;
            res.n = n;
            res.o = o;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            results.push_back(res);
        }
    }

    bench_util::printCsvFormat(results);
}

// Benchmarks extract ghosts kernel plus data transfer between host and device without MPI.
// If called with multiple processes, only root process runs the benchmarks.
// These benchmarks seem to be very unstable.
// Run with e.g.: benchmarks bench_data_transfer_host_device
TEST_CASE("bench_ghostupdate_mpi_ocl_steps")
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

    // Check if mpi is initialized
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REQUIRE(isInitialized);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    // Check number of processes
    int mpi_size = -1;
    MPI_Comm_size(mpi_comm, &mpi_size);
    // REQUIRE(mpi_size == 1);

    int periodic = 1;

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

    if (mpi_rank == 0)
    {
        std::cout << "Testing the following grid sizes" << std::endl;
        for (auto gr : gridsTBT)
        {
            int m = gr[0];
            int n = gr[1];
            int o = gr[2];
            std::cout << "  " << m << "," << n << "," << o << std::endl;
        }
    }

    std::vector<bench_util::ResultGhostUpdateMpi> results;

    bool printedGpu = false;
    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];
        int mglob = m * mpi_dims[0];
        int nglob = n * mpi_dims[1];
        int oglob = o * mpi_dims[2];
        int ghosts = 1;
        int mgh = m + 2 * ghosts;
        int ngh = n + 2 * ghosts;
        int ogh = o + 2 * ghosts;

        // Create a dummy problem
        auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
        auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
        mgcl::Problem p(m, n, o, f, v, mglob, nglob, oglob);
        p.setUseOpencl(true);
        p.setDeviceType(CL_DEVICE_TYPE_GPU);
        p.setGhosts(ghosts);
        p.setSilent(true);
        p.setMpiComm(mpi_comm);
        p.init();

        auto& mpiData = p.getLevelAt(0).getMpiData();

        if (!printedGpu)
        {
            for (int i = 0; i < mpi_size; i++)
            {
                MPI_Barrier(mpi_comm);
                if (i == mpi_rank)
                {
                    std::cout << "on rank " << mpi_rank << ", GPU info: ";
                    p.getOpenCLHelper().outputDeviceInfo();
                }
            }
            printedGpu = true;
        }

        int err;

        // actual test buffers
        mgcl::Cuboid& c_h = p.getLevelAt(0).getV();
        c_h.fillRandom();
        mgcl::CuboidGpu& c_d = p.getLevelAt(0).getDVIn();

        int yz = c_d.getNgh() * c_d.getOgh();
        int xz = c_d.getMgh() * c_d.getOgh();
        int xy = c_d.getMgh() * c_d.getNgh();
        int ressize = 2 * yz * c_d.getGhostsM() + 2 * xz * c_d.getGhostsN() + 2 * xy * c_d.getGhostsO();
        auto& d_planesbuf = p.getDPlanesBuf();

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ms, "ms")
            .epochs(CLI_ARGS::bench_epochs)
            .epochIterations(CLI_ARGS::bench_iterations);

        if (mpi_rank > 0)
            bench.output(nullptr);

        {
            std::string name = std::string("extractBorderPlanes_newret_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o))
                                   .append("_gh")
                                   .append(std::to_string(ghosts));

            bench.run(std::string(name).c_str(), [&] { //
                auto sbuf_ptr = c_d.extractBorderPlanes(p.getCommands(), p.getProgram(), &d_planesbuf, nullptr, nullptr, nullptr);
                p.getOpenCLHelper().finish();
            });

            bench_util::ResultGhostUpdateMpi res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.mloc = m;
            res.nloc = n;
            res.oloc = o;
            res.mglob = mglob;
            res.nglob = nglob;
            res.oglob = oglob;
            res.gpus = mpi_size;
            res.ghosts = ghosts;
            results.push_back(res);
        }

        {
            std::vector<double> h_planesbuf(ressize);
            std::string name = std::string("extractBorderPlanes_reuseret_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o))
                                   .append("_gh")
                                   .append(std::to_string(ghosts));

            bench.run(std::string(name).c_str(), [&] { //
                c_d.extractBorderPlanes(p.getCommands(), p.getProgram(), &d_planesbuf, &h_planesbuf, nullptr, nullptr);
                p.getOpenCLHelper().finish();
            });

            bench_util::ResultGhostUpdateMpi res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.mloc = m;
            res.nloc = n;
            res.oloc = o;
            res.mglob = mglob;
            res.nglob = nglob;
            res.oglob = oglob;
            res.gpus = mpi_size;
            res.ghosts = ghosts;
            results.push_back(res);
        }

        {
            std::vector<double> sbuf(ressize);
            std::vector<double> rbuf(ressize);
            std::string name = std::string("sendBorderPlanes")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o))
                                   .append("_gh")
                                   .append(std::to_string(ghosts));

            bench.run(std::string(name).c_str(), [&] { //
                mgcl::mpi_util::sendBorderPlanes(c_d.getMgh(), c_d.getNgh(), c_d.getOgh(),
                                                 c_d.getGhostsM(), c_d.getGhostsN(), c_d.getGhostsO(), 1,
                                                 sbuf, rbuf, mpiData);
            });

            bench_util::ResultGhostUpdateMpi res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.mloc = m;
            res.nloc = n;
            res.oloc = o;
            res.mglob = mglob;
            res.nglob = nglob;
            res.oglob = oglob;
            res.gpus = mpi_size;
            res.ghosts = ghosts;
            results.push_back(res);
        }

        {
            std::vector<double> sbuf(ressize);
            std::vector<double> rbuf(ressize);
            std::string name = std::string("sendBorderPlanes_copyedges")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o))
                                   .append("_gh")
                                   .append(std::to_string(ghosts));

            // int base_yz_front = 0;
            int base_yz_back = ghosts * yz;
            int base_xz_top = 2 * ghosts * yz;
            int base_xz_bottom = 2 * ghosts * yz + ghosts * xz;
            int base_xy_left = 2 * ghosts * yz + 2 * ghosts * xz;
            int base_xy_right = 2 * ghosts * yz + 2 * ghosts * xz + ghosts * xy;

            bench.run(std::string(name).c_str(), [&] { //
                // Write received edges of cuboid to send buffer
                // i0: i index of recv buffers for yz plane (always all i indices)
                // i1: i index of send buffers xz back ghosts
                // j0: j index of send buffers for xz plane (always all j indices)
                // j1: j index of recv buffer yz top edge
                // j2: j index of recv buffer yz bottom edge
                for (int i0 = 0, i1 = m + ghosts;
                     i0 < ghosts;
                     i0++, i1++)
                    for (int j0 = 0, j1 = ghosts, j2 = n;
                         j0 < ghosts;
                         j0++, j1++, j2++)
                        for (int k = 0; k < ogh; k++)
                        {
                            // Upper front edge - Write ghosts in the front (from back recv buffer) to xz top send buffer
                            sbuf[base_xz_top + j0 * xz + i0 * ogh + k] = rbuf[base_yz_back + i0 * yz + j1 * ogh + k];

                            // Lower front edge - Write ghosts in the front (from back recv buffer) to xz bottom send buffer
                            sbuf[base_xz_bottom + j0 * xz + i0 * ogh + k] = rbuf[base_yz_back + i0 * yz + j2 * ogh + k];

                            // Upper back edge - Write ghosts in the back (from front recv buffer, base 0) to xz top send buffer
                            sbuf[base_xz_top + j0 * xz + i1 * ogh + k] = rbuf[i0 * yz + j1 * ogh + k];

                            // Lower back edge - Write ghosts in the back (from front recv buffer, base 0) to xz bottom send buffer
                            sbuf[base_xz_bottom + j0 * xz + i1 * ogh + k] = rbuf[i0 * yz + j2 * ogh + k];
                        }

                // Write received left torus of cuboid to send buffer
                // k0: k index of send buffers for xy planes (left and right)
                // k1: k index of recv buffers for copy into left send buffer
                // k2: k index of recv buffers for copy into right send buffer
                for (int k0 = 0, k1 = ghosts, k2 = o;
                     k0 < ghosts;
                     k0++, k1++, k2++)
                {

                    // Copying from yz planes (front back)
                    // i0: i index of recv buffers for yz plane (always all i indices)
                    // i1: i index of send buffers xz back ghosts
                    for (int i0 = 0, i1 = m + ghosts;
                         i0 < ghosts;
                         i0++, i1++)
                        for (int j = ghosts; j < ghosts + n; j++)
                        {
                            // Left front face - Write ghosts in the send left buffer from recv back buffer
                            sbuf[base_xy_left + k0 * xy + i0 * ngh + j] = rbuf[base_yz_back + i0 * yz + j * ogh + k1];

                            // Left back face - Write ghosts in the send left buffer from recv front buffer
                            sbuf[base_xy_left + k0 * xy + i1 * ngh + j] = rbuf[i0 * yz + j * ogh + k1];

                            // TODO right
                            // Right front face - Write ghosts in the send right buffer from recv back buffer
                            sbuf[base_xy_right + k0 * xy + i0 * ngh + j] = rbuf[base_yz_back + i0 * yz + j * ogh + k2];

                            // Right back face - Write ghosts in the send right buffer from recv front buffer
                            sbuf[base_xy_right + k0 * xy + i1 * ngh + j] = rbuf[i0 * yz + j * ogh + k2];
                        }

                    // Copying from xz planes (top bottom)
                    // j0: j index of recv buffers yz bottom and send both left and right
                    // j1: j index of send buffers xy bottom ghosts (recv top)
                    for (int i = 0; i < mgh; i++)
                        for (int j0 = 0, j1 = n + ghosts;
                             j0 < ghosts;
                             j0++, j1++)
                        {
                            // Left top edge - Write ghosts in the send left buffer from recv bottom buffer
                            sbuf[base_xy_left + k0 * xy + i * ngh + j0] = rbuf[base_xz_bottom + j0 * xz + i * ogh + k1];

                            // Left bottom edge - Write ghosts in the send left buffer from recv top buffer
                            sbuf[base_xy_left + k0 * xy + i * ngh + j1] = rbuf[base_xz_top + j0 * xz + i * ogh + k1];

                            // Right top edge - Write ghosts in the send left buffer from recv bottom buffer
                            sbuf[base_xy_right + k0 * xy + i * ngh + j0] = rbuf[base_xz_bottom + j0 * xz + i * ogh + k2];

                            // Right bottom face - Write ghosts in the send right buffer from recv top buffer
                            sbuf[base_xy_right + k0 * xy + i * ngh + j1] = rbuf[base_xz_top + j0 * xz + i * ogh + k2];
                        }
                }
            });

            bench_util::ResultGhostUpdateMpi res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.mloc = m;
            res.nloc = n;
            res.oloc = o;
            res.mglob = mglob;
            res.nglob = nglob;
            res.oglob = oglob;
            res.gpus = mpi_size;
            res.ghosts = ghosts;
            results.push_back(res);
        }

        {
            std::vector<double> rbuf(ressize);
            std::string name = std::string("pasteGhostsFromBorderPlanes_newdevicebuf_N")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o))
                                   .append("_gh")
                                   .append(std::to_string(ghosts));

            bench.run(std::string(name).c_str(), [&] { //
                c_d.pasteGhostsFromBorderPlanes(p.getContext(), p.getCommands(), p.getProgram(), nullptr, &rbuf, nullptr, nullptr);
                p.getOpenCLHelper().finish();
            });

            bench_util::ResultGhostUpdateMpi res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.mloc = m;
            res.nloc = n;
            res.oloc = o;
            res.mglob = mglob;
            res.nglob = nglob;
            res.oglob = oglob;
            res.gpus = mpi_size;
            res.ghosts = ghosts;
            results.push_back(res);
        }

        {
            std::string name = std::string("pasteGhostsFromBorderPlanes_reusedevicebuf_N")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o))
                                   .append("_gh")
                                   .append(std::to_string(ghosts));

            bench.run(std::string(name).c_str(), [&] { //
                c_d.pasteGhostsFromBorderPlanes(p.getContext(), p.getCommands(), p.getProgram(), &d_planesbuf, nullptr, nullptr, nullptr);
                p.getOpenCLHelper().finish();
            });

            bench_util::ResultGhostUpdateMpi res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.mloc = m;
            res.nloc = n;
            res.oloc = o;
            res.mglob = mglob;
            res.nglob = nglob;
            res.oglob = oglob;
            res.gpus = mpi_size;
            res.ghosts = ghosts;
            results.push_back(res);
        }
    }

    bench_util::printCsvFormat(results, mpi_comm, mpi_rank);
}

namespace mgcl_bench_ghost_update_wgsizes
{
    enum class KernelVersion
    {
        THREE_D_MODULO,
        ONE_D,
        THREE_D_TYPECONV_DOUBLE,          // e.g. int ireal = i + floor(((double)(ghm - 1 - i)) / m + 1) * m;
        THREE_D_TYPECONV_FLOAT,           // e.g. int ireal = i + floor(((float)(ghm - 1 - i)) / m + 1) * m;
        THREE_D_TYPECONV_FLOAT_CONVERTFN, // e.g. int ireal = i + ((int)(((float)(ghm - 1 - i)) / m) + 1.0f) * m;
        THREE_D_TYPECONV_FLOAT_TRUNCATE,  // e.g. int ireal = i + ((int)(((float)(ghm - 1 - i)) / m) + 1.0f) * m;

        SPLIT // split into 3 kernels, one per dimension. Also is a 3d kernel.
    };

    using size_t3 = struct
    {
        int x, y, z;
    };

    // Regular jacobi method like in production code, but with mpi stuff removed. I.e. only single gpu
    int updateGhosts(mgcl::Problem& problem, mgcl::CuboidGpu& dBuffer, KernelVersion kernelVersion, std::vector<size_t> wgsizes)
    {
        // TODO actually request these as arguments
        int m = dBuffer.getM();
        int n = dBuffer.getN();
        int o = dBuffer.getO();
        int mgh = dBuffer.getMgh();
        int ngh = dBuffer.getNgh();
        int ogh = dBuffer.getOgh();
        int ghosts_m = dBuffer.getGhostsM();
        int ghosts_n = dBuffer.getGhostsN();
        int ghosts_o = dBuffer.getGhostsO();

        if (!problem.isPeriodic())
            return CL_SUCCESS;

        int err;

        bool is3d = kernelVersion == KernelVersion::THREE_D_MODULO ||
                    kernelVersion == KernelVersion::THREE_D_TYPECONV_DOUBLE ||
                    kernelVersion == KernelVersion::THREE_D_TYPECONV_FLOAT ||
                    kernelVersion == KernelVersion::THREE_D_TYPECONV_FLOAT_CONVERTFN ||
                    kernelVersion == KernelVersion::THREE_D_TYPECONV_FLOAT_TRUNCATE;

        // Create the compute kernel from the program
        std::string kernelName;
        if (kernelVersion == KernelVersion::THREE_D_MODULO)
            kernelName = "update_ghosts_periodic_3d_modulo";
        else if (kernelVersion == KernelVersion::THREE_D_TYPECONV_DOUBLE)
            kernelName = "update_ghosts_periodic_3d_typeconv_double";
        else if (kernelVersion == KernelVersion::THREE_D_TYPECONV_FLOAT)
            kernelName = "update_ghosts_periodic_3d_typeconv_float";
        else if (kernelVersion == KernelVersion::THREE_D_TYPECONV_FLOAT_CONVERTFN)
            kernelName = "update_ghosts_periodic_3d_typeconv_float_convertfn";
        else if (kernelVersion == KernelVersion::THREE_D_TYPECONV_FLOAT_TRUNCATE)
            kernelName = "update_ghosts_periodic_3d_typeconv_float_truncate";
        else if (kernelVersion == KernelVersion::ONE_D)
            kernelName = "update_ghosts_periodic_1d";
        cl_kernel kernel = clCreateKernel(problem.getOpenCLHelper().getProgram(), kernelName.c_str(), &err);
        mgcl::mgclCheckError(err, "clCreateKernel");

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dBuffer);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_m);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_n);
        err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts_o);
        mgcl::mgclCheckError(err, "Setting kernel arguments");

        // one work-item per ghost cell (excluding real cells). Pad global sizes to fit to local sizes
        // int mgh = m + 2 * gh;
        // int ngh = n + 2 * gh;
        // int ogh = o + 2 * gh;
        size_t global[3] = {static_cast<size_t>(ogh), static_cast<size_t>(ngh), static_cast<size_t>(mgh)};
        size_t const local[3] = {wgsizes[0], wgsizes[1], wgsizes[2]};

        if (!is3d)
        {
            global[0] = static_cast<size_t>(mgh * ngh * ogh);
            global[1] = 1;
            global[2] = 1;
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

    int updateGhostsSplit(mgcl::Problem& problem, mgcl::CuboidGpu& dBuffer, KernelVersion kernelVersion, std::vector<size_t> wgsizes)
    {
        // TODO actually request these as arguments
        int m = dBuffer.getM();
        int n = dBuffer.getN();
        int o = dBuffer.getO();
        int mgh = dBuffer.getMgh();
        int ngh = dBuffer.getNgh();
        int ogh = dBuffer.getOgh();
        int ghosts_m = dBuffer.getGhostsM();
        int ghosts_n = dBuffer.getGhostsN();
        int ghosts_o = dBuffer.getGhostsO();

        if (!problem.isPeriodic())
            return CL_SUCCESS;

        if (kernelVersion != KernelVersion::SPLIT)
            throw "Only KernelVersion::SPLIT is supported";

        int err;

        bool is3d = true;

        // Create the compute kernel from the program
        const char* kernelNamex = "update_ghosts_periodic_x";
        const char* kernelNamey = "update_ghosts_periodic_y";
        const char* kernelNamez = "update_ghosts_periodic_z";
        cl_kernel kernelx = clCreateKernel(problem.getOpenCLHelper().getProgram(), kernelNamex, &err);
        mgcl::mgclCheckError(err, "clCreateKernel");
        cl_kernel kernely = clCreateKernel(problem.getOpenCLHelper().getProgram(), kernelNamey, &err);
        mgcl::mgclCheckError(err, "clCreateKernel");
        cl_kernel kernelz = clCreateKernel(problem.getOpenCLHelper().getProgram(), kernelNamez, &err);
        mgcl::mgclCheckError(err, "clCreateKernel");

        // assign kernel arguments
        int pos = 0;
        err = clSetKernelArg(kernelx, pos, sizeof(cl_mem), &dBuffer);
        err |= clSetKernelArg(kernelx, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernelx, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernelx, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernelx, ++pos, sizeof(int), &ghosts_m);
        err |= clSetKernelArg(kernelx, ++pos, sizeof(int), &ghosts_n);
        err |= clSetKernelArg(kernelx, ++pos, sizeof(int), &ghosts_o);
        mgcl::mgclCheckError(err, "Setting kernelx arguments");
        pos = 0;
        err = clSetKernelArg(kernely, pos, sizeof(cl_mem), &dBuffer);
        err |= clSetKernelArg(kernely, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernely, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernely, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernely, ++pos, sizeof(int), &ghosts_m);
        err |= clSetKernelArg(kernely, ++pos, sizeof(int), &ghosts_n);
        err |= clSetKernelArg(kernely, ++pos, sizeof(int), &ghosts_o);
        mgcl::mgclCheckError(err, "Setting kernely arguments");
        pos = 0;
        err = clSetKernelArg(kernelz, pos, sizeof(cl_mem), &dBuffer);
        err |= clSetKernelArg(kernelz, ++pos, sizeof(int), &m);
        err |= clSetKernelArg(kernelz, ++pos, sizeof(int), &n);
        err |= clSetKernelArg(kernelz, ++pos, sizeof(int), &o);
        err |= clSetKernelArg(kernelz, ++pos, sizeof(int), &ghosts_m);
        err |= clSetKernelArg(kernelz, ++pos, sizeof(int), &ghosts_n);
        err |= clSetKernelArg(kernelz, ++pos, sizeof(int), &ghosts_o);
        mgcl::mgclCheckError(err, "Setting kernelz arguments");

        // one work-item per ghost cell (excluding real cells). Pad global sizes to fit to local sizes
        // int mgh = m + 2 * gh;
        // int ngh = n + 2 * gh;
        // int ogh = o + 2 * gh;
        size_t globalx[2] = {static_cast<size_t>(ngh), static_cast<size_t>(ogh)};
        size_t globaly[2] = {static_cast<size_t>(mgh), static_cast<size_t>(ogh)};
        size_t globalz[2] = {static_cast<size_t>(mgh), static_cast<size_t>(ngh)};
        size_t const local[3] = {wgsizes[0], wgsizes[1], wgsizes[2]};

        for (int i = 0; i < 2; i++)
        {
            if (globalx[i] % local[i] != 0)
                globalx[i] += local[i] - (globalx[i] % local[i]);
            if (globaly[i] % local[i] != 0)
                globaly[i] += local[i] - (globaly[i] % local[i]);
            if (globalz[i] % local[i] != 0)
                globalz[i] += local[i] - (globalz[i] % local[i]);
        }

        cl_event ev;

        // enqueue kernel
        err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernelx, 2, NULL, globalx, local, 0, NULL, &ev);
        mgcl::mgclCheckError(err, "Enqueueing update_ghosts_periodic kernel");
        err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernely, 2, NULL, globaly, local, 0, NULL, &ev);
        mgcl::mgclCheckError(err, "Enqueueing update_ghosts_periodic kernel");
        err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernelz, 2, NULL, globalz, local, 0, NULL, &ev);
        mgcl::mgclCheckError(err, "Enqueueing update_ghosts_periodic kernel");

        if (problem.isProfilingEnabled())
        {
            problem.getProfilingData()->addMeasurement(problem.getCommands(), ev, kernelNamex,
                                                       {globalx[0], globalx[1], 0},
                                                       {local[0], local[1], local[2]});
            problem.getProfilingData()->addMeasurement(problem.getCommands(), ev, kernelNamey,
                                                       {globaly[0], globaly[1], 0},
                                                       {local[0], local[1], local[2]});
            problem.getProfilingData()->addMeasurement(problem.getCommands(), ev, kernelNamez,
                                                       {globalz[0], globalz[1], 0},
                                                       {local[0], local[1], local[2]});
        }
        mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

        err = clReleaseKernel(kernelx);
        mgcl::mgclCheckError(err, "Releasing update_ghosts_periodic kernel");
        err = clReleaseKernel(kernely);
        mgcl::mgclCheckError(err, "Releasing update_ghosts_periodic kernel");
        err = clReleaseKernel(kernelz);
        mgcl::mgclCheckError(err, "Releasing update_ghosts_periodic kernel");

        return err;
    }

    // Benchs the ghost update of CuboidGpu for different workgroup sizes.
    TEST_CASE("benchGhostUpdateWgSizes")
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
            p.setKernelFile("kernels_ghost_update.cl");
            if (CLI_ARGS::useBinaryFile)
            {
                p.setBinaryFile("kernels_ghost_update.bin");
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

            std::vector<std::vector<size_t>> wg_sizes_1d = {{4, 1, 1}, {8, 1, 1}, {32, 1, 1}, {64, 1, 1}, {128, 1, 1}, {256, 1, 1}};
            for (auto wg : wg_sizes_1d)
            {
                lv0.getDVIn().fill(p.getProgram(), p.getCommands(), 0.0, false, nullptr, nullptr);
                lv0.getDVIn().fill1dIndex(p.getProgram(), p.getCommands(), true, true, nullptr, nullptr);

                std::string name = std::string("ghost_update_1d_")
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
                    updateGhosts(p, lv0.getDVIn(), KernelVersion::ONE_D, wg);
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

                // if (CLI_ARGS::checkResults)
                // {
                //     v_out_default = std::make_unique<mgcl::Cuboid>(ml, nl, ol, ghosts, ghosts, ghosts);
                //     lv0.getDVIn().read(p.getCommands(), v_out_default.get(), true);
                // }
            }

            std::vector<std::vector<size_t>> wg_sizes_3d = {{4, 4, 4}, {4, 4, 8}, {2, 2, 8}, {8, 8, 8}, {4, 4, 16}, {32, 1, 1}, {64, 1, 1}, {128, 1, 1}};
            for (auto wg : wg_sizes_3d)
            {
                {
                    lv0.getDVIn().fill(p.getProgram(), p.getCommands(), 0.0, false, nullptr, nullptr);
                    lv0.getDVIn().fill1dIndex(p.getProgram(), p.getCommands(), true, true, nullptr, nullptr);

                    std::string name = std::string("ghost_update_3d_modulo_")
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
                        updateGhosts(p, lv0.getDVIn(), KernelVersion::THREE_D_MODULO, wg);
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

                    // if (CLI_ARGS::checkResults)
                    // {
                    //     v_out_default = std::make_unique<mgcl::Cuboid>(ml, nl, ol, ghosts, ghosts, ghosts);
                    //     lv0.getDVIn().read(p.getCommands(), v_out_default.get(), true);
                    // } }
                }
            }

            for (auto wg : wg_sizes_3d)
            {
                lv0.getDVIn().fill(p.getProgram(), p.getCommands(), 0.0, false, nullptr, nullptr);
                lv0.getDVIn().fill1dIndex(p.getProgram(), p.getCommands(), true, true, nullptr, nullptr);

                std::string name = std::string("ghost_update_3d_typeconv_double_")
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
                    updateGhosts(p, lv0.getDVIn(), KernelVersion::THREE_D_TYPECONV_DOUBLE, wg);
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

                // if (CLI_ARGS::checkResults)
                // {
                //     v_out_default = std::make_unique<mgcl::Cuboid>(ml, nl, ol, ghosts, ghosts, ghosts);
                //     lv0.getDVIn().read(p.getCommands(), v_out_default.get(), true);
                // } }
            }

            for (auto wg : wg_sizes_3d)
            {
                lv0.getDVIn().fill(p.getProgram(), p.getCommands(), 0.0, false, nullptr, nullptr);
                lv0.getDVIn().fill1dIndex(p.getProgram(), p.getCommands(), true, true, nullptr, nullptr);

                std::string name = std::string("ghost_update_3d_typeconv_float_")
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
                    updateGhosts(p, lv0.getDVIn(), KernelVersion::THREE_D_TYPECONV_FLOAT, wg);
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

                // if (CLI_ARGS::checkResults)
                // {
                //     v_out_default = std::make_unique<mgcl::Cuboid>(ml, nl, ol, ghosts, ghosts, ghosts);
                //     lv0.getDVIn().read(p.getCommands(), v_out_default.get(), true);
                // } }
            }

            // if (false)
            for (auto wg : wg_sizes_3d)
            {
                lv0.getDVIn().fill(p.getProgram(), p.getCommands(), 0.0, false, nullptr, nullptr);
                lv0.getDVIn().fill1dIndex(p.getProgram(), p.getCommands(), true, true, nullptr, nullptr);

                std::string name = std::string("ghost_update_3d_typeconv_float_truncate_")
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
                    updateGhosts(p, lv0.getDVIn(), KernelVersion::THREE_D_TYPECONV_FLOAT_CONVERTFN, wg);
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

                // if (CLI_ARGS::checkResults)
                // {
                //     v_out_default = std::make_unique<mgcl::Cuboid>(ml, nl, ol, ghosts, ghosts, ghosts);
                //     lv0.getDVIn().read(p.getCommands(), v_out_default.get(), true);
                // } }
            }

            // if (false)
            for (auto wg : wg_sizes_3d)
            {
                lv0.getDVIn().fill(p.getProgram(), p.getCommands(), 0.0, false, nullptr, nullptr);
                lv0.getDVIn().fill1dIndex(p.getProgram(), p.getCommands(), true, true, nullptr, nullptr);

                std::string name = std::string("ghost_update_3d_typeconv_float_truncate_")
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
                    updateGhosts(p, lv0.getDVIn(), KernelVersion::THREE_D_TYPECONV_FLOAT_TRUNCATE, wg);
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

                // if (CLI_ARGS::checkResults)
                // {
                //     v_out_default = std::make_unique<mgcl::Cuboid>(ml, nl, ol, ghosts, ghosts, ghosts);
                //     lv0.getDVIn().read(p.getCommands(), v_out_default.get(), true);
                // } }
            }

            if (false)
                for (auto wg : wg_sizes_3d)
                {
                    lv0.getDVIn().fill(p.getProgram(), p.getCommands(), 0.0, false, nullptr, nullptr);
                    lv0.getDVIn().fill1dIndex(p.getProgram(), p.getCommands(), true, true, nullptr, nullptr);

                    std::string name = std::string("ghost_update_3d_split_")
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
                        updateGhostsSplit(p, lv0.getDVIn(), KernelVersion::SPLIT, wg);
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

                    // if (CLI_ARGS::checkResults)
                    // {
                    //     v_out_default = std::make_unique<mgcl::Cuboid>(ml, nl, ol, ghosts, ghosts, ghosts);
                    //     lv0.getDVIn().read(p.getCommands(), v_out_default.get(), true);
                    // } }
                }

            // call regular ghpst update that is in production code once for kernel timing comparison
            auto& conf = p.getKernelConfig();
            // Jacobi kernels
            conf["update_ghosts_periodic"] = mgcl::conf::KernelWorkgroupSizes{{1, {4, 4, 8}}};
            mgcl::MultigridEngine::updateGhosts(p, lv0.getDVIn(), nullptr, true);

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

    // Benchs the ghost update of CuboidGpu for different ghost layer sizes, i.e. ghost amounts.
    TEST_CASE("benchGhostUpdateLayerSize")
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

        std::vector<bench_util::ResultGhostUpdateMpi> results;

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
            p.setKernelFile("kernels_ghost_update.cl");
            if (CLI_ARGS::useBinaryFile)
            {
                p.setBinaryFile("kernels_ghost_update.bin");
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

            std::vector<size_t> gh_counts = {1, 2, 3, 4, 5};
            std::vector<std::vector<size_t>> wg_sizes_3d = {{4, 4, 4}, {4, 4, 8}, {4, 4, 16}};
            for (auto gh : gh_counts)
            {
                if (CLI_ARGS::enableKernelProfiling)
                    p.getProfilingData()->getMeasurements().clear();

                mgcl::CuboidGpu c(p.getContext(), CL_MEM_READ_WRITE, ml, nl, ol, gh, gh, gh);
                for (auto wg : wg_sizes_3d)
                {
                    c.fill(p.getProgram(), p.getCommands(), 0.0, false, nullptr, nullptr);
                    c.fill1dIndex(p.getProgram(), p.getCommands(), true, true, nullptr, nullptr);

                    std::string name = std::string("ghost_update_3d_modulo_")
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
                        updateGhosts(p, c, KernelVersion::THREE_D_MODULO, wg);
                        p.finish();
                    });

                    bench_util::ResultGhostUpdateMpi res;

                    res.name = name;
                    res.minTime = bench_util::getMinTime(bench, name);
                    res.medianTime = bench_util::getMedianTime(bench, name);
                    res.avgTime = bench_util::getAvgTime(bench, name);
                    res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                    res.mloc = ml;
                    res.nloc = nl;
                    res.oloc = ol;
                    res.mglob = mglob;
                    res.nglob = nglob;
                    res.oglob = oglob;
                    res.ghosts = gh;     // amount of ghost cells in one direction
                    res.gpus = mpi_size; // GPU-count, equals mpi proc count
                    results.push_back(res);

                    // if (CLI_ARGS::checkResults)
                    // {
                    //     v_out_default = std::make_unique<mgcl::Cuboid>(ml, nl, ol, ghosts, ghosts, ghosts);
                    //     lv0.getDVIn().read(p.getCommands(), v_out_default.get(), true);
                    // }
                }

                for (auto wg : wg_sizes_3d)
                {
                    c.fill(p.getProgram(), p.getCommands(), 0.0, false, nullptr, nullptr);
                    c.fill1dIndex(p.getProgram(), p.getCommands(), true, true, nullptr, nullptr);

                    std::string name = std::string("ghost_update_3d_float_")
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
                        updateGhosts(p, c, KernelVersion::THREE_D_TYPECONV_FLOAT, wg);
                        p.finish();
                    });

                    bench_util::ResultGhostUpdateMpi res;

                    res.name = name;
                    res.minTime = bench_util::getMinTime(bench, name);
                    res.medianTime = bench_util::getMedianTime(bench, name);
                    res.avgTime = bench_util::getAvgTime(bench, name);
                    res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
                    res.mloc = ml;
                    res.nloc = nl;
                    res.oloc = ol;
                    res.mglob = mglob;
                    res.nglob = nglob;
                    res.oglob = oglob;
                    res.ghosts = gh;     // amount of ghost cells in one direction
                    res.gpus = mpi_size; // GPU-count, equals mpi proc count
                    results.push_back(res);

                    // if (CLI_ARGS::checkResults)
                    // {
                    //     v_out_default = std::make_unique<mgcl::Cuboid>(ml, nl, ol, ghosts, ghosts, ghosts);
                    //     lv0.getDVIn().read(p.getCommands(), v_out_default.get(), true);
                    // }
                }
                if (CLI_ARGS::enableKernelProfiling)
                {
                    // p.getProfilingData()->printBestTimingsPerKernel(kernelProfilesStream);
                    p.getProfilingData()->printBestTimingsPerKernelAsCsv(kernelProfilesStream);
                    kernelProfilesStream << "----------" << std::endl;
                }
            }
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