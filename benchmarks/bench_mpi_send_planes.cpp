#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"

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

// forward declarations
void sendBorderPlanes(int mgh, int ngh, int ogh, int ghosts_m, int ghosts_n, int ghosts_o,
                      int stencilWidth,
                      std::vector<double>& sbuf, std::vector<double>& rbuf, mgcl::MPILevelData& mpiData);
void sendBorderPlanes(int mgh, int ngh, int ogh, int ghosts_m, int ghosts_n, int ghosts_o,
                      int stencilWidth,
                      std::vector<double>& sbuf, mgcl::MPILevelData& mpiData);

// Benchs mpiSendBorderPlanes using MPI_Sendrecv vs. MPI_Sendrecv_replace
// --grids argument must be given, e.g. --grids 4,8,16
TEST_CASE("bench_sendBorderPlanes")
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

    int periodic = 1;

    // check if mpi is initialized
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REQUIRE(isInitialized);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    // check number of processes, at least 2 needed (but 8 makes more sense so that the processes at the border don't send data to themselves)
    int mpi_size = -1;
    MPI_Comm_size(mpi_comm, &mpi_size);
    REQUIRE(mpi_size > 1);

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

    // Taken from mgcl::Level::initMpiData
    mgcl::MPILevelData mpiLevelData(mpi_comm);
    {
        /* Calculating neighbours */
        int ret;
        ret = MPI_Cart_shift(mpiLevelData.comm, 2, 1, &mpiLevelData.left, &mpiLevelData.right);
        mgcl::mpi_util::mgclCheckMpiError(mpiLevelData.comm, ret, "MPI_Cart_shift x-direction");
        ret = MPI_Cart_shift(mpiLevelData.comm, 1, 1, &mpiLevelData.down, &mpiLevelData.up);
        mgcl::mpi_util::mgclCheckMpiError(mpiLevelData.comm, ret, "MPI_Cart_shift y-direction");
        ret = MPI_Cart_shift(mpiLevelData.comm, 0, 1, &mpiLevelData.front, &mpiLevelData.back);
        mgcl::mpi_util::mgclCheckMpiError(mpiLevelData.comm, ret, "MPI_Cart_shift z-direction");
    }

    for (auto gr : gridsTBT)
    {
        // grid sizes to test the buffer size with
        int mtt = gr[0];
        int ntt = gr[1];
        int ott = gr[2];
        int ghosts = 1;

        int mgh = mtt + 2 * ghosts;
        int ngh = ntt + 2 * ghosts;
        int ogh = ott + 2 * ghosts;

        int yz = ngh * ogh;
        int xz = mgh * ogh;
        int xy = mgh * ngh;
        int ressize = (2 * yz * ghosts + 2 * xz * ghosts + 2 * xy * ghosts) * 1;

        std::vector<double> sbuf(ressize);
        auto rbuf(sbuf); // copy of sbuf

        for (size_t i = 0; i < sbuf.size(); i++)
        {
            sbuf[i] = i;
            rbuf[i] = rbuf.size() - i - 1;
        }

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ms, "ms")
            .epochs(CLI_ARGS::bench_epochs)
            .epochIterations(CLI_ARGS::bench_iterations);

        // output only on root process
        if (mpi_rank > 0)
            bench.output(nullptr);

        {
            std::string name = std::string("MPI_Sendrecv_")
                                   .append(std::to_string(mtt))
                                   .append("_")
                                   .append(std::to_string(ntt))
                                   .append("_")
                                   .append(std::to_string(ott));
            bench.run(std::string(name).c_str(), [&] { //
                sendBorderPlanes(mgh, ngh, ogh, 1, 1, 1, 1, sbuf, rbuf, mpiLevelData);
                MPI_Barrier(mpi_comm);
            });

            bench_util::ResultMpi res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.m = mtt;
            res.n = ntt;
            res.o = ott;
            res.mglob = mtt * mpi_dims[0];
            res.nglob = ntt * mpi_dims[1];
            res.oglob = ott * mpi_dims[2];
            res.gpus = mpi_size; // GPU-count, or process count here
            res.LT = -1;         // mpiLevelThreshold
            results.push_back(res);
        }

        {
            std::string name = std::string("MPI_Sendrecv_replace_")
                                   .append(std::to_string(mtt))
                                   .append("_")
                                   .append(std::to_string(ntt))
                                   .append("_")
                                   .append(std::to_string(ott));
            bench.run(std::string(name).c_str(), [&] { //
                sendBorderPlanes(mgh, ngh, ogh, 1, 1, 1, 1, sbuf, mpiLevelData);
                MPI_Barrier(mpi_comm);
            });

            bench_util::ResultMpi res;
            res.name = name;
            res.minTime = bench_util::getMinTime(bench, name);
            res.medianTime = bench_util::getMedianTime(bench, name);
            res.avgTime = bench_util::getAvgTime(bench, name);
            res.medianAbsolutePercentError = bench_util::getMedianAbsolutePercentError(bench, name);
            res.m = mtt;
            res.n = ntt;
            res.o = ott;
            res.mglob = mtt * mpi_dims[0];
            res.nglob = ntt * mpi_dims[1];
            res.oglob = ott * mpi_dims[2];
            res.gpus = mpi_size; // GPU-count, or process count here
            res.LT = -1;         // mpiLevelThreshold
            results.push_back(res);
        }
    }

    MPI_Barrier(mpi_comm);
    bench_util::printCsvFormat(results, mpi_comm, mpi_rank);
}

void sendBorderPlanes(int mgh, int ngh, int ogh, int ghosts_m, int ghosts_n, int ghosts_o,
                      int stencilWidth,
                      std::vector<double>& sbuf, std::vector<double>& rbuf, mgcl::MPILevelData& mpiData)
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

    // Send planes to neighbors
    int myid, err;
    MPI_Comm_rank(mpiData.comm, &myid);

    // Send front planes to the back
    err = MPI_Sendrecv(static_cast<void*>(sbuf.data()), yzgh * stencilSize, MPI_DOUBLE, mpiData.back, 0,
                       static_cast<void*>(rbuf.data()), yzgh * stencilSize, MPI_DOUBLE, mpiData.front, 0,
                       mpiData.comm, MPI_STATUS_IGNORE);
    mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");

    // Send back planes to the front
    err = MPI_Sendrecv(static_cast<void*>(&(sbuf[base_yz_back])), yzgh * stencilSize, MPI_DOUBLE, mpiData.front, 0,
                       static_cast<void*>(&(rbuf[base_yz_back])), yzgh * stencilSize, MPI_DOUBLE, mpiData.back, 0,
                       mpiData.comm, MPI_STATUS_IGNORE);
    mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");

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
    err = MPI_Sendrecv(static_cast<void*>(&(sbuf[base_xz_top])), xzgh * stencilSize, MPI_DOUBLE, mpiData.down, 0,
                       static_cast<void*>(&(rbuf[base_xz_top])), xzgh * stencilSize, MPI_DOUBLE, mpiData.up, 0,
                       mpiData.comm, MPI_STATUS_IGNORE);
    mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");

    // Send bottom planes to the top
    err = MPI_Sendrecv(static_cast<void*>(&(sbuf[base_xz_bottom])), xzgh * stencilSize, MPI_DOUBLE, mpiData.up, 0,
                       static_cast<void*>(&(rbuf[base_xz_bottom])), xzgh * stencilSize, MPI_DOUBLE, mpiData.down, 0,
                       mpiData.comm, MPI_STATUS_IGNORE);
    mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");

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
    err = MPI_Sendrecv(static_cast<void*>(&(sbuf[base_xy_left])), xygh * stencilSize, MPI_DOUBLE, mpiData.right, 0,
                       static_cast<void*>(&(rbuf[base_xy_left])), xygh * stencilSize, MPI_DOUBLE, mpiData.left, 0,
                       mpiData.comm, MPI_STATUS_IGNORE);
    mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");

    // Send right planes to the left
    err = MPI_Sendrecv(static_cast<void*>(&(sbuf[base_xy_right])), xygh * stencilSize, MPI_DOUBLE, mpiData.left, 0,
                       static_cast<void*>(&(rbuf[base_xy_right])), xygh * stencilSize, MPI_DOUBLE, mpiData.right, 0,
                       mpiData.comm, MPI_STATUS_IGNORE);
    mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");
}

void sendBorderPlanes(int mgh, int ngh, int ogh, int ghosts_m, int ghosts_n, int ghosts_o,
                      int stencilWidth,
                      std::vector<double>& sbuf, mgcl::MPILevelData& mpiData)
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

    // Send planes to neighbors
    int myid, err;
    MPI_Comm_rank(mpiData.comm, &myid);

    // Send front planes to the back
    err = MPI_Sendrecv_replace(static_cast<void*>(sbuf.data()), yzgh * stencilSize, MPI_DOUBLE, mpiData.back, 0,
                               mpiData.front, 0,
                               mpiData.comm, MPI_STATUS_IGNORE);
    mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");

    // Send back planes to the front
    err = MPI_Sendrecv_replace(static_cast<void*>(&(sbuf[base_yz_back])), yzgh * stencilSize, MPI_DOUBLE, mpiData.front, 0,
                               mpiData.back, 0,
                               mpiData.comm, MPI_STATUS_IGNORE);
    mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");

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
                    sbuf[st * xzgh + base_xz_top + j0 * xz + i0 * ogh + k] = sbuf[st * yzgh + base_yz_back + i0 * yz + j1 * ogh + k];

                    // Lower front edge - Write ghosts in the front (from back recv buffer) to xz bottom send buffer
                    sbuf[st * xzgh + base_xz_bottom + j0 * xz + i0 * ogh + k] = sbuf[st * yzgh + base_yz_back + i0 * yz + j2 * ogh + k];

                    // Upper back edge - Write ghosts in the back (from front recv buffer, base 0) to xz top send buffer
                    sbuf[st * xzgh + base_xz_top + j0 * xz + i1 * ogh + k] = sbuf[st * yzgh + i0 * yz + j1 * ogh + k];

                    // Lower back edge - Write ghosts in the back (from front recv buffer, base 0) to xz bottom send buffer
                    sbuf[st * xzgh + base_xz_bottom + j0 * xz + i1 * ogh + k] = sbuf[st * yzgh + i0 * yz + j2 * ogh + k];
                }

    // Send top planes to the bottom
    err = MPI_Sendrecv_replace(static_cast<void*>(&(sbuf[base_xz_top])), xzgh * stencilSize, MPI_DOUBLE, mpiData.down, 0,
                               mpiData.up, 0,
                               mpiData.comm, MPI_STATUS_IGNORE);
    mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");

    // Send bottom planes to the top
    err = MPI_Sendrecv_replace(static_cast<void*>(&(sbuf[base_xz_bottom])), xzgh * stencilSize, MPI_DOUBLE, mpiData.up, 0,
                               mpiData.down, 0,
                               mpiData.comm, MPI_STATUS_IGNORE);
    mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");

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
                    sbuf[st * xygh + base_xy_left + k0 * xy + i0 * ngh + j] = sbuf[st * yzgh + base_yz_back + i0 * yz + j * ogh + k1];

                    // Left back face - Write ghosts in the send left buffer from recv front buffer
                    sbuf[st * xygh + base_xy_left + k0 * xy + i1 * ngh + j] = sbuf[st * yzgh + i0 * yz + j * ogh + k1];

                    // Right front face - Write ghosts in the send right buffer from recv back buffer
                    sbuf[st * xygh + base_xy_right + k0 * xy + i0 * ngh + j] = sbuf[st * yzgh + base_yz_back + i0 * yz + j * ogh + k2];

                    // Right back face - Write ghosts in the send right buffer from recv front buffer
                    sbuf[st * xygh + base_xy_right + k0 * xy + i1 * ngh + j] = sbuf[st * yzgh + i0 * yz + j * ogh + k2];
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
                    sbuf[st * xygh + base_xy_left + k0 * xy + i * ngh + j0] = sbuf[st * xzgh + base_xz_bottom + j0 * xz + i * ogh + k1];

                    // Left bottom edge - Write ghosts in the send left buffer from recv top buffer
                    sbuf[st * xygh + base_xy_left + k0 * xy + i * ngh + j1] = sbuf[st * xzgh + base_xz_top + j0 * xz + i * ogh + k1];

                    // Right top edge - Write ghosts in the send left buffer from recv bottom buffer
                    sbuf[st * xygh + base_xy_right + k0 * xy + i * ngh + j0] = sbuf[st * xzgh + base_xz_bottom + j0 * xz + i * ogh + k2];

                    // Right bottom face - Write ghosts in the send right buffer from recv top buffer
                    sbuf[st * xygh + base_xy_right + k0 * xy + i * ngh + j1] = sbuf[st * xzgh + base_xz_top + j0 * xz + i * ogh + k2];
                }
        }

    // Send left planes to the right
    err = MPI_Sendrecv_replace(static_cast<void*>(&(sbuf[base_xy_left])), xygh * stencilSize, MPI_DOUBLE, mpiData.right, 0,
                               mpiData.left, 0,
                               mpiData.comm, MPI_STATUS_IGNORE);
    mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");

    // Send right planes to the left
    err = MPI_Sendrecv_replace(static_cast<void*>(&(sbuf[base_xy_right])), xygh * stencilSize, MPI_DOUBLE, mpiData.left, 0,
                               mpiData.right, 0,
                               mpiData.comm, MPI_STATUS_IGNORE);
    mgcl::mpi_util::mgclCheckMpiError(mpiData.comm, err, "MPI_Sendrecv");
}