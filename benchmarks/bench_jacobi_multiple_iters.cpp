#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
using namespace std::chrono_literals;

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/multigrid_engine.hpp"
#include "../src/mgcl/opencl_helper.hpp"
#include "../src/mgcl/problem.hpp"
#include "../test/test_utility.hpp"
#include "bench_util.hpp"
#include "cli_args.hpp"

namespace MGCL_GLOBS
{
    bool PRINT_PARAMS = true;
}

/**
 * Benchmarks Jacobi multiple iterations without ghost-update in-between vs. standard Jacobi.
 *
 */
TEST_CASE("benchmark_jacobi_seq_multiple_iters")
{
    int N = GENERATE(8, 16, 32, 64);
    // int N = 16;
    int m = N;
    int n = N;
    int o = N;
    double h = 1.0 / (double)N;

    int maxStepsPerIter = 3;
    int iters = 3;

    double omega = 0.8;
    int maxiter = 3;
    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
    mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_27POINT;
    double stencilFactor = 1.0 / (26.0 * h * h);

    ankerl::nanobench::Bench bench;
    bench.timeUnit(1ms, "ms")
        // .epochs(1)
        // .epochIterations(1)
        .minEpochTime(100ms)
        .relative(true);

    for (int spi = 1; spi <= maxStepsPerIter; spi++)
    {
        mgcl::Cuboid v(m, n, o, spi, spi, spi);
        mgcl::Cuboid r(m, n, o, spi, spi, spi);
        mgcl::Cuboid f(m, n, o, spi, spi, spi);
        v.fillRandom();
        f.fillRandom();

        std::string name = std::string("seq, N = ")
                               .append(std::to_string(N))
                               .append(", spi = ")
                               .append(std::to_string(spi));
        bench.run(std::string(name).c_str(), [&]
                  { mgcl::MultigridEngine::jacobiSeq(v, f, r, omega, h * h, iters, resnorm, stencilType, stencilFactor,
                                                     nullptr, false, true, spi); });
    }
}

// Same as above but with OCL.
TEST_CASE("benchmark_jacobi_ocl_multiple_iters")
{
    int maxStepsPerIter = 3;
    std::stringstream ss;
    ss << "N;iters;spi;ns" << std::endl;

    // Vector to collect all minimum times per spi, in order to get avg results later.
    std::vector<std::vector<int>> mintimesPerSpi(maxStepsPerIter);

    std::vector<int> Ns = {8, 16, 32, 64};
    std::vector<int> itersAll = {3, 5, 10};
    for (auto N : Ns)
    {
        int m = N;
        int n = N;
        int o = N;
        double h = 1.0 / (double)N;

        double omega = 0.8;
        int maxiter = 3;
        mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
        mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_27POINT;
        double stencilFactor = 1.0 / (26.0 * h * h);

        for (auto iters : itersAll)
        {
            ankerl::nanobench::Bench bench;
            bench.timeUnit(1ns, "ns")
                // .epochs(1)
                // .epochIterations(1)
                .minEpochTime(100ms)
                .relative(true);

            for (int spi = 1; spi <= maxStepsPerIter; spi++)
            {
                auto v = std::make_shared<mgcl::Cuboid>(m, n, o, spi, spi, spi);
                auto r = std::make_shared<mgcl::Cuboid>(m, n, o, spi, spi, spi);
                auto f = std::make_shared<mgcl::Cuboid>(m, n, o, spi, spi, spi);
                v->fillRandom();
                f->fillRandom();

                auto p = std::make_shared<mgcl::Problem>(m, n, o, v, f);
                p->setGhosts(spi);
                p->setGhostsIn(spi);
                p->setJacobiIterationsPerKernel(spi);
                p->setUseOpencl(true);
                p->setSilent(true);
                p->init();
                auto& level = p->getLevelAt(0);
                mgcl_test::TestUtility tu(p);

                std::string name = std::string("ocl, N = ")
                                       .append(std::to_string(N))
                                       .append(", spi = ")
                                       .append(std::to_string(spi))
                                       .append(", iters = ")
                                       .append(std::to_string(iters));
                bench.run(std::string(name).c_str(), [&] { //
                    mgcl::MultigridEngine::jacobi(*p, level, iters, false, spi);
                    tu.finish(); //
                });

                // Get minimum of all epochs in ns
                double min = 1000000;
                for (auto r : bench.results())
                    if (r.minimum(ankerl::nanobench::Result::Measure::elapsed) < min)
                        min = r.minimum(ankerl::nanobench::Result::Measure::elapsed) * 1000.0 * 1000.0 * 1000.0;

                ss << N << ";" << iters << ";" << spi << ";" << min << std::endl;
                mintimesPerSpi[spi - 1].push_back(min);

                // std::cout << "bench.results()[0].size(): " << bench.results()[0].size() << std::endl;
            }

            // std::ofstream renderOutCsv(std::string("jacobiMultiIterOcl_")
            //                                .append(std::to_string(N))
            //                                .append("_")
            //                                .append(".csv"));

            // std::streambuf *old = std::cout.rdbuf(ss.rdbuf());
            // bench.render(ankerl::nanobench::templates::csv(), std::cout);
            // std::cout.rdbuf(old);
        }
    }

    std::cout << ss.str() << std::endl;

    std::vector<double> avgs = {0, 0, 0};
    for (int spi = 0; spi < maxStepsPerIter; spi++)
    {
        for (int val : mintimesPerSpi[spi])
            avgs[spi] += val;

        avgs[spi] /= (double)mintimesPerSpi[spi].size();
    }

    std::cout << "avgs:" << std::endl
              << "  spi: 1: " << avgs[0] << " ns" << std::endl
              << "  spi: 2: " << avgs[1] << " ns" << std::endl
              << "  spi: 3: " << avgs[2] << " ns" << std::endl;
}

// Same as above but with OCL and MPI.
// Timings will be collected per node and printed by rank at the end.
// Run with e.g.: mpiexec -n 4 benchmarks "benchmark Jacobi MPI OCL multiple iters"
TEST_CASE("bench_jacobi_mpi_ocl_multiple_iters")
{

    using std::min;

    if (CLI_ARGS::grids.size() == 0 && (CLI_ARGS::gridsMin.size() == 0 || CLI_ARGS::gridsMax.size() == 0))
        throw "Need to specify at least one local grid size, e.g. using --grids 4,8,16 or --gridsMin 4,4,4 AND --gridsMax 32,32,32";

    if (CLI_ARGS::jacobiIters.size() == 0)
        throw "Need to specify at least one jacobi iteration to test, e.g. using --jacobiIters 1,3,10";

    if (CLI_ARGS::jacobiStepsPerIter.size() == 0)
        throw "Need to specify at least one jacobi steps per iteration to test, e.g. using --jacobiStepsPerIter 1,2,3";

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

    if (mpi_rank == 0 && MGCL_GLOBS::PRINT_PARAMS)
    {
        MGCL_GLOBS::PRINT_PARAMS = false;

        std::cout << "Testing the following grid sizes" << std::endl;
        for (auto gr : gridsTBT)
        {
            int m = gr[0];
            int n = gr[1];
            int o = gr[2];
            std::cout << "  local size: " << m << "," << n << "," << o << ", global size: "
                      << m * mpi_dims[0] << "," << n * mpi_dims[1] << "," << o * mpi_dims[2] << std::endl;
        }

        std::cout << "And the following jacobi iterations" << std::endl
                  << "  ";
        for (auto it : CLI_ARGS::jacobiIters)
            std::cout << it << ", ";
        std::cout << std::endl;

        std::cout << "And the following steps per iteration" << std::endl
                  << "  ";
        for (auto it : CLI_ARGS::jacobiStepsPerIter)
            std::cout << it << ", ";
        std::cout << std::endl;

        std::cout << "Tests with iters < stepsPerIter will be skipped!" << std::endl;
    }
    MPI_Barrier(mpi_comm);

    std::vector<bench_util::ResultJacobiMpi> results;
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

        double omega = 0.8;
        int maxiter = 3;
        mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
        mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_VARYING;

        for (auto iters : CLI_ARGS::jacobiIters)
        {
            ankerl::nanobench::Bench bench;
            bench.timeUnit(1ms, "ms")
                .epochs(CLI_ARGS::bench_epochs)
                .epochIterations(CLI_ARGS::bench_iterations)
                // .minEpochTime(100ms)
                .relative(CLI_ARGS::jacobiStepsPerIter.size() > 1);

            for (int spi : CLI_ARGS::jacobiStepsPerIter)
            {
                if (iters < spi)
                {
                    if (mpi_rank == 0)
                        std::cout << "skipping iter = " << iters << ", spi = " << spi << std::endl;
                    continue;
                }

                auto v = std::make_shared<mgcl::Cuboid>(m, n, o, spi, spi, spi);
                auto r = std::make_shared<mgcl::Cuboid>(m, n, o, spi, spi, spi);
                auto f = std::make_shared<mgcl::Cuboid>(m, n, o, spi, spi, spi);
                v->fillRandom();
                f->fillRandom();

                auto p = std::make_shared<mgcl::Problem>(m, n, o, f, v, mglob, nglob, oglob);
                p->setMpiComm(mpi_comm);
                p->setGhosts(spi);
                p->setGhostsIn(spi);
                p->setOmega(omega);
                p->setJacobiIterationsPerKernel(spi);
                p->setUseOpencl(true);
                p->setSilent(true);
                if (p->getStencilType() == mgcl::MGCL_VARYING)
                    p->getStencilValues()->fillRandom();
                p->init();
                auto& level = p->getLevelAt(0);

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

                std::string name = std::string("ocl_mpi_N")
                                       .append(std::to_string(m))
                                       .append("_")
                                       .append(std::to_string(n))
                                       .append("_")
                                       .append(std::to_string(o))
                                       .append("_spi")
                                       .append(std::to_string(spi))
                                       .append("_iters")
                                       .append(std::to_string(iters));
                bench.run(std::string(name).c_str(), [&] { //
                    MPI_Barrier(mpi_comm);
                    mgcl::MultigridEngine::jacobi(*p, level, iters, false, spi);
                    p->getOpenCLHelper().finish();
                    MPI_Barrier(mpi_comm);
                });

                // std::cout << "rank " << mpi_rank << " done" << std::endl;
                MPI_Barrier(mpi_comm);

                bench_util::ResultJacobiMpi res;
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
                res.spi = spi;
                res.iters = iters;
                results.push_back(res);
            }
        }
    }

    bench_util::printCsvFormat(results, mpi_comm, mpi_rank);

    MPI_Barrier(mpi_comm);
}
