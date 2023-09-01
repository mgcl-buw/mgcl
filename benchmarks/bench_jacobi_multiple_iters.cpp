#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
using namespace std::chrono_literals;

#include "../src/cuboid.hpp"
#include "../src/multigrid_engine.hpp"
#include "../src/opencl_helper.hpp"
#include "../src/problem.hpp"
#include "../test/ocl_wrapper.hpp"
#include "../test/test_utility.hpp"
#include "bench_render_templates.hpp"

/**
 * Benchmarks Jacobi multiple iterations without ghost-update in-between vs. standard Jacobi.
 *
 */
TEST_CASE("benchmark Jacobi seq multiple iters", "[console][jacobiMulti][seq]")
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
    double stencilFactor = 1.0 / (30.0 * h * h);

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
                  { mgcl::MultigridEngine::jacobiSeq(v, f, r, omega, iters, resnorm, stencilType, stencilFactor,
                                                     nullptr, false, true, spi); });
    }
}

// Same as above but with OCL.
TEST_CASE("benchmark Jacobi OCL multiple iters", "[console][jacobiMulti][ocl]")
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
        double stencilFactor = 1.0 / (30.0 * h * h);

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
TEST_CASE("benchmark Jacobi MPI OCL multiple iters", "[console][jacobiMulti][mpi][ocl]")
{
    using std::min;

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

    int maxStepsPerIter = 3;
    std::stringstream ss;
    ss << "N;iters;spi;ns" << std::endl;

    // Vector to collect all minimum times per spi, in order to get avg results later.
    std::vector<std::vector<int>> mintimesPerSpi(maxStepsPerIter);

    std::vector<int> Ns = {8, 16, 32, 64};
    std::vector<int> itersAll = {3, 5, 10};
    for (auto N : Ns)
    {
        int mglob = N;
        int nglob = N;
        int oglob = N;
        double h = 1.0 / (double)N;

        double omega = 0.8;
        int maxiter = 3;
        mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
        mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_27POINT;
        double stencilFactor = 1.0 / (30.0 * h * h);

        /* Initialize start and end for local grid */
        int m_start = (mglob / mpi_dims[0]) * mpi_coords[0] + min(mpi_coords[0], (mglob % mpi_dims[0]));
        int m_end = (mglob / mpi_dims[0]) * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (mglob % mpi_dims[0])) - 1;
        int n_start = (nglob / mpi_dims[1]) * mpi_coords[1] + min(mpi_coords[1], (nglob % mpi_dims[1]));
        int n_end = (nglob / mpi_dims[1]) * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (nglob % mpi_dims[1])) - 1;
        int o_start = (oglob / mpi_dims[2]) * mpi_coords[2] + min(mpi_coords[2], (oglob % mpi_dims[2]));
        int o_end = (oglob / mpi_dims[2]) * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (oglob % mpi_dims[2])) - 1;

        int m = (m_end - m_start) + 1;
        int n = (n_end - n_start) + 1;
        int o = (o_end - o_start) + 1;

        for (auto iters : itersAll)
        {
            ankerl::nanobench::Bench bench;
            bench.timeUnit(1ns, "ns")
                .epochs(11)
                .epochIterations(1)
                // .minEpochTime(100ms)
                .relative(true);

            for (int spi = 1; spi <= maxStepsPerIter; spi++)
            {
                auto v = std::make_shared<mgcl::Cuboid>(m, n, o, spi, spi, spi);
                auto r = std::make_shared<mgcl::Cuboid>(m, n, o, spi, spi, spi);
                auto f = std::make_shared<mgcl::Cuboid>(m, n, o, spi, spi, spi);
                v->fillRandom();
                f->fillRandom();

                auto p = std::make_shared<mgcl::Problem>(m, n, o, v, f, mglob, nglob, oglob);
                p->setMpiComm(mpi_comm);
                p->setGhosts(spi);
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
                    MPI_Barrier(mpi_comm);
                });

                // std::cout << "rank " << mpi_rank << " done" << std::endl;
                MPI_Barrier(mpi_comm);

                // Get minimum of all epochs in ns
                double minTime = 1000000;
                for (auto r : bench.results())
                    if (r.minimum(ankerl::nanobench::Result::Measure::elapsed) < minTime)
                        minTime = r.minimum(ankerl::nanobench::Result::Measure::elapsed) * 1000.0 * 1000.0 * 1000.0;

                ss << N << ";" << iters << ";" << spi << ";" << minTime << std::endl;
                mintimesPerSpi[spi - 1].push_back(minTime);

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

    for (int i = 0; i < mpi_size; i++)
    {
        MPI_Barrier(mpi_comm);
        if (i == mpi_rank)
            std::cout << "rank " << i << " avgs:" << std::endl
                      << "  spi: 1: " << avgs[0] << " ns" << std::endl
                      << "  spi: 2: " << avgs[1] << " ns" << std::endl
                      << "  spi: 3: " << avgs[2] << " ns" << std::endl;
    }
    MPI_Barrier(mpi_comm);
}
