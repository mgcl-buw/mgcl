#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>

#include "../../src/mgcl/cuboid.hpp"
#include "../../src/mgcl/mpi_level_data.hpp"
#include "../../src/mgcl/mpi_util.hpp"
#include "../../src/mgcl/problem.hpp"
#include "../test_utility.hpp"

#include "mpi.h"

std::shared_ptr<mgcl::Cuboid> calculateError(mgcl::Cuboid& solution, mgcl::Cuboid& approximation);
double calculateMaxError(mgcl::Cuboid& error);
double calculateErrorNorm(double h, mgcl::Cuboid& error);

// Tests if vcycle is correct for multiple processes but everything is actually done on one process, i.e.
// gathering and scattering happens on level 0.
// With 1 process mgcl should detect that MPI is actually not used, thus this test should behave like the solve tests.
// Run with e.g. mpiexec -n 2 tests_mpi MPI_vcycle_immediate_gather_scatter_Laplace7p
TEST_CASE("MPI_vcycle_immediate_gather_scatter_Laplace7p")
{
    using std::min;

    // global grid sizes
    int m = 8;
    int n = 8;
    int o = 8;
    int periodic = 1;
    int gh = 1;

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

    /* Initialize start and end for local grid */
    int m_start = (m / mpi_dims[0]) * mpi_coords[0] + min(mpi_coords[0], (m % mpi_dims[0]));
    int m_end = (m / mpi_dims[0]) * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (m % mpi_dims[0])) - 1;
    int n_start = (n / mpi_dims[1]) * mpi_coords[1] + min(mpi_coords[1], (n % mpi_dims[1]));
    int n_end = (n / mpi_dims[1]) * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (n % mpi_dims[1])) - 1;
    int o_start = (o / mpi_dims[2]) * mpi_coords[2] + min(mpi_coords[2], (o % mpi_dims[2]));
    int o_end = (o / mpi_dims[2]) * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (o % mpi_dims[2])) - 1;

    int ml = (m_end - m_start) + 1;
    int nl = (n_end - n_start) + 1;
    int ol = (o_end - o_start) + 1;

    CAPTURE(ml, nl, ol);

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
    REQUIRE(ml <= m);
    REQUIRE(nl > 0);
    REQUIRE(nl <= n);
    REQUIRE(ol > 0);
    REQUIRE(ol <= o);

    // Set up 4th order periodic problem
    int ghin = 0; // TODO check with ghin > 0
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    auto f = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    auto solution = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    mgcl_test::create4hOrderPeriodicProblem(*v, *f, *solution);

    // Create local slices
    std::shared_ptr<mgcl::Cuboid> vloc(v->slice(m_start, m_end, n_start, n_end, o_start, o_end));
    std::shared_ptr<mgcl::Cuboid> floc(f->slice(m_start, m_end, n_start, n_end, o_start, o_end));
    std::shared_ptr<mgcl::Cuboid> solutionloc(solution->slice(m_start, m_end, n_start, n_end, o_start, o_end));

    // Create local problem
    mgcl::Problem p(ml, nl, ol, floc, vloc, m, n, o);
    // p.setMaxiterVcycles(5);
    p.setGhosts(gh);
    p.setGhostsIn(ghin);
    p.setMpiLevelThreshold(0);
    p.setMpiComm(mpi_comm);

    p.solveSeq();

    // Gather local approximations on rank 0 for checking.
    if (mpi_rank > 0)
        mgcl::mpi_util::gather(p.getMpiComm(), *vloc);
    else
    {
        // copy from vloc to v first
        for (int i = vloc->getGhostsM(); i < vloc->getM() + vloc->getGhostsM(); i++)
            for (int j = vloc->getGhostsN(); j < vloc->getN() + vloc->getGhostsN(); j++)
                for (int k = vloc->getGhostsO(); k < vloc->getO() + vloc->getGhostsO(); k++)
                {
                    (*v)[i][j][k] = (*vloc)[i][j][k];
                }

        // Gather into v from other processes
        if (mpi_size > 1)
            mgcl::mpi_util::gather(p.getMpiComm(), *v); // TODO check with different ghost amounts

        // check if solution is good
        auto err = calculateError(*solution, *v);
        auto errNorm = calculateErrorNorm(1.0 / (double)m, *err);
        auto errMax = calculateMaxError(*err);

        std::cout << "seq MPI Laplace" << std::endl;
        std::cout << "rank 0: " << std::endl;
        std::cout
            << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
            << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

        // Running this with 1 proc yields
        // ||e||_2 = 4.62293179000930129e-03
        // e_max = 9.00816189282011015e-03
        // which should be equal to the global result when run with multiple processors.

        REQUIRE(errNorm < 1e-2);
        REQUIRE(errMax < 1e-2);
        REQUIRE_THAT(errNorm, Catch::Matchers::WithinRel(4.62293179000930129e-03));
        REQUIRE_THAT(errMax, Catch::Matchers::WithinRel(9.00816189282011015e-03));
    }
}

// Tests if vcycle is correct for multiple processes but everything is actually done on one process, i.e.
// gathering and scattering happens on level 0.
// With 1 process mgcl should detect that MPI is actually not used, thus this test should behave like the solve tests.
// Run with e.g. mpiexec -n 2 tests_mpi MPI_vcycle_GPU_immediate_gather_scatter_Laplace7p
TEST_CASE("MPI_vcycle_GPU_immediate_gather_scatter_Laplace7p")
{
    if (mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_GPU))
    {
        using std::min;

        // global grid sizes
        int m = 8;
        int n = 8;
        int o = 8;
        int periodic = 1;
        int gh = 1;

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

        /* Initialize start and end for local grid */
        int m_start = (m / mpi_dims[0]) * mpi_coords[0] + min(mpi_coords[0], (m % mpi_dims[0]));
        int m_end = (m / mpi_dims[0]) * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (m % mpi_dims[0])) - 1;
        int n_start = (n / mpi_dims[1]) * mpi_coords[1] + min(mpi_coords[1], (n % mpi_dims[1]));
        int n_end = (n / mpi_dims[1]) * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (n % mpi_dims[1])) - 1;
        int o_start = (o / mpi_dims[2]) * mpi_coords[2] + min(mpi_coords[2], (o % mpi_dims[2]));
        int o_end = (o / mpi_dims[2]) * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (o % mpi_dims[2])) - 1;

        int ml = (m_end - m_start) + 1;
        int nl = (n_end - n_start) + 1;
        int ol = (o_end - o_start) + 1;

        CAPTURE(ml, nl, ol);

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
        REQUIRE(ml <= m);
        REQUIRE(nl > 0);
        REQUIRE(nl <= n);
        REQUIRE(ol > 0);
        REQUIRE(ol <= o);

        // Set up 4th order periodic problem
        int ghin = 0; // TODO check with ghin > 0
        auto v = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
        auto f = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
        auto solution = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
        mgcl_test::create4hOrderPeriodicProblem(*v, *f, *solution);

        // Create local slices
        std::shared_ptr<mgcl::Cuboid> vloc(v->slice(m_start, m_end, n_start, n_end, o_start, o_end));
        std::shared_ptr<mgcl::Cuboid> floc(f->slice(m_start, m_end, n_start, n_end, o_start, o_end));
        std::shared_ptr<mgcl::Cuboid> solutionloc(solution->slice(m_start, m_end, n_start, n_end, o_start, o_end));

        // Create local problem
        mgcl::Problem p(ml, nl, ol, floc, vloc, m, n, o);
        // p.setMaxiterVcycles(5);
        p.setUseOpencl(true);
        p.setReadResults(true);
        p.setGhosts(gh);
        p.setGhostsIn(ghin);
        p.setMpiLevelThreshold(0);
        p.setMpiComm(mpi_comm);

        p.solve();

        // Gather local approximations on rank 0 for checking.
        if (mpi_rank > 0)
            mgcl::mpi_util::gather(p.getMpiComm(), *vloc);
        else
        {
            // copy from vloc to v first
            for (int i = vloc->getGhostsM(); i < vloc->getM() + vloc->getGhostsM(); i++)
                for (int j = vloc->getGhostsN(); j < vloc->getN() + vloc->getGhostsN(); j++)
                    for (int k = vloc->getGhostsO(); k < vloc->getO() + vloc->getGhostsO(); k++)
                    {
                        (*v)[i][j][k] = (*vloc)[i][j][k];
                    }

            // Gather into v from other processes
            if (mpi_size > 1)
                mgcl::mpi_util::gather(p.getMpiComm(), *v); // TODO check with different ghost amounts

            // check if solution is good
            auto err = calculateError(*solution, *v);
            auto errNorm = calculateErrorNorm(1.0 / (double)m, *err);
            auto errMax = calculateMaxError(*err);

            std::cout << "seq MPI Laplace" << std::endl;
            std::cout << "rank 0: " << std::endl;
            std::cout
                << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
                << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

            // Running this with 1 proc yields
            // ||e||_2 = 4.62293179000930129e-03
            // e_max = 9.00816189282011015e-03
            // which should be equal to the global result when run with multiple processors.

            REQUIRE(errNorm < 1e-2);
            REQUIRE(errMax < 1e-2);
            REQUIRE_THAT(errNorm, Catch::Matchers::WithinRel(4.62293179000930129e-03));
            REQUIRE_THAT(errMax, Catch::Matchers::WithinRel(9.00816189282011015e-03));
        }
    }
}

// Tests if vcycle is correct for multiple processes but everything is actually done on one process, i.e.
// gathering and scattering happens on level 0.
// With 1 process mgcl should detect that MPI is actually not used, thus this test should behave like the solve tests.
// Run with e.g. mpiexec -n 2 tests_mpi MPI_vcycle_immediate_gather_scatter
TEST_CASE("MPI_vcycle_immediate_gather_scatter_Varying27p")
{
    using std::min;

    // global grid sizes
    int m = 16;
    int n = 16;
    int o = 16;
    int periodic = 1;
    int gh = 1;

    // Problem parameters
    double tol = 1e-7;
    int nu1 = 2;
    int nu2 = 2;
    double omega = 0.8;
    int maxIterVCycles = 5;

    // check if mpi is initialized
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REQUIRE(isInitialized);

    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    // check number of processes
    int mpi_size = -1;
    MPI_Comm_size(mpi_comm, &mpi_size);
    // REQUIRE(mpi_size == 8);

    // MPI_Comm_set_errhandler(mpi_comm, MPI_ERRORS_RETURN);

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

    /* Initialize start and end for local grid */
    int m_start = (m / mpi_dims[0]) * mpi_coords[0] + min(mpi_coords[0], (m % mpi_dims[0]));
    int m_end = (m / mpi_dims[0]) * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (m % mpi_dims[0])) - 1;
    int n_start = (n / mpi_dims[1]) * mpi_coords[1] + min(mpi_coords[1], (n % mpi_dims[1]));
    int n_end = (n / mpi_dims[1]) * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (n % mpi_dims[1])) - 1;
    int o_start = (o / mpi_dims[2]) * mpi_coords[2] + min(mpi_coords[2], (o % mpi_dims[2]));
    int o_end = (o / mpi_dims[2]) * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (o % mpi_dims[2])) - 1;

    int ml = (m_end - m_start) + 1;
    int nl = (n_end - n_start) + 1;
    int ol = (o_end - o_start) + 1;

    CAPTURE(ml, nl, ol);

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
    REQUIRE(ml <= m);
    REQUIRE(nl > 0);
    REQUIRE(nl <= n);
    REQUIRE(ol > 0);
    REQUIRE(ol <= o);

    // Set up 4th order periodic problem
    int ghin = 0; // TODO check with ghin > 0
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    auto f = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    auto solution = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    mgcl_test::create4hOrderPeriodicProblem(*v, *f, *solution);

    // Create local slices
    std::shared_ptr<mgcl::Cuboid> vloc(v->slice(m_start, m_end, n_start, n_end, o_start, o_end));
    std::shared_ptr<mgcl::Cuboid> floc(f->slice(m_start, m_end, n_start, n_end, o_start, o_end));
    std::shared_ptr<mgcl::Cuboid> solutionloc(solution->slice(m_start, m_end, n_start, n_end, o_start, o_end));

    // Create local problem
    mgcl::Problem p(ml, nl, ol, floc, vloc, m, n, o);
    p.setMaxiterVcycles(maxIterVCycles);
    p.setOmega(omega);
    p.setNu1(nu1);
    p.setNu2(nu2);
    p.setTol(tol);
    p.setGhosts(gh);
    p.setGhostsIn(ghin);
    p.setMpiLevelThreshold(0);
    p.setMpiComm(mpi_comm);

    p.setStencilType(mgcl::MGCL_VARYING);
    auto& sv = *p.getStencilValues();

    if (mpi_rank == 0)
    {
        REQUIRE(sv.getM() == m);
        REQUIRE(sv.getM() == n);
        REQUIRE(sv.getM() == o);
    }

    mgcl_test::fill7pLaplace(sv, false);

    p.solveSeq();

    // Gather local approximations on rank 0 for checking.
    if (mpi_rank > 0)
        mgcl::mpi_util::gather(p.getMpiComm(), *vloc);
    else
    {
        // copy from vloc to v first
        for (int i = vloc->getGhostsM(); i < vloc->getM() + vloc->getGhostsM(); i++)
            for (int j = vloc->getGhostsN(); j < vloc->getN() + vloc->getGhostsN(); j++)
                for (int k = vloc->getGhostsO(); k < vloc->getO() + vloc->getGhostsO(); k++)
                {
                    (*v)[i][j][k] = (*vloc)[i][j][k];
                }

        // Gather into v from other processes
        if (mpi_size > 1)
            mgcl::mpi_util::gather(p.getMpiComm(), *v); // TODO check with different ghost amounts

        // check if solution is good
        auto err = calculateError(*solution, *v);
        auto errNorm = calculateErrorNorm(1.0 / (double)m, *err);
        auto errMax = calculateMaxError(*err);

        std::cout << "seq MPI Varying27p" << std::endl;
        std::cout << "rank 0: " << std::endl;
        std::cout
            << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
            << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

        // Running this with 1 proc yields
        // ||e||_2 = 2.79451354429798363e-03
        // e_max = 2.89860791352128345e-03
        // which should be equal to the global result when run with multiple processors.

        REQUIRE(errNorm < 1e-2);
        REQUIRE(errMax < 1e-2);
        REQUIRE_THAT(errNorm, Catch::Matchers::WithinRel(2.79451354429798363e-03));
        REQUIRE_THAT(errMax, Catch::Matchers::WithinRel(2.89860791352128345e-03));
    }
}

// Tests if vcycle is correct for multiple processes with mpiLevelThreshold > 0, i.e. some fine levels are
// calculated distributed and only the coarse levels are calculated on rank 0.
// Run with e.g. mpiexec -n 2 tests_mpi MPI_vcycle_threshold_gt_0_Laplace7p
TEST_CASE("MPI_vcycle_threshold_gt_0_Laplace7p")
{
    using std::min;

    // global grid sizes
    int m = 16;
    int n = 16;
    int o = 16;
    int periodic = 1;
    int gh = 1;

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

    /* Initialize start and end for local grid */
    int m_start = (m / mpi_dims[0]) * mpi_coords[0] + min(mpi_coords[0], (m % mpi_dims[0]));
    int m_end = (m / mpi_dims[0]) * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (m % mpi_dims[0])) - 1;
    int n_start = (n / mpi_dims[1]) * mpi_coords[1] + min(mpi_coords[1], (n % mpi_dims[1]));
    int n_end = (n / mpi_dims[1]) * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (n % mpi_dims[1])) - 1;
    int o_start = (o / mpi_dims[2]) * mpi_coords[2] + min(mpi_coords[2], (o % mpi_dims[2]));
    int o_end = (o / mpi_dims[2]) * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (o % mpi_dims[2])) - 1;

    int ml = (m_end - m_start) + 1;
    int nl = (n_end - n_start) + 1;
    int ol = (o_end - o_start) + 1;

    CAPTURE(ml, nl, ol);

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
    REQUIRE(ml <= m);
    REQUIRE(nl > 0);
    REQUIRE(nl <= n);
    REQUIRE(ol > 0);
    REQUIRE(ol <= o);

    int threshold = GENERATE(1, 2, 3);
    CAPTURE(threshold);

    if (mpi_rank == 0)
        std::cerr << std::endl
                  << "Testing with threshold: " << threshold << std::endl;

    // Set up 4th order periodic problem
    int ghin = 0; // TODO check with ghin > 0
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    auto f = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    auto solution = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    mgcl_test::create4hOrderPeriodicProblem(*v, *f, *solution);

    // Create local slices
    std::shared_ptr<mgcl::Cuboid> vloc(v->slice(m_start, m_end, n_start, n_end, o_start, o_end));
    std::shared_ptr<mgcl::Cuboid> floc(f->slice(m_start, m_end, n_start, n_end, o_start, o_end));
    std::shared_ptr<mgcl::Cuboid> solutionloc(solution->slice(m_start, m_end, n_start, n_end, o_start, o_end));

    // Create local problem
    mgcl::Problem p(ml, nl, ol, floc, vloc, m, n, o);
    // p.setMaxiterVcycles(5);
    p.setGhosts(gh);
    p.setGhostsIn(ghin);
    p.setMpiLevelThreshold(threshold);
    p.setMpiComm(mpi_comm);

    p.solveSeq();

    // Gather local approximations on rank 0 for checking.
    if (mpi_rank > 0)
        mgcl::mpi_util::gather(p.getMpiComm(), *vloc);
    else
    {
        // copy from vloc to v first
        for (int i = vloc->getGhostsM(); i < vloc->getM() + vloc->getGhostsM(); i++)
            for (int j = vloc->getGhostsN(); j < vloc->getN() + vloc->getGhostsN(); j++)
                for (int k = vloc->getGhostsO(); k < vloc->getO() + vloc->getGhostsO(); k++)
                {
                    (*v)[i][j][k] = (*vloc)[i][j][k];
                }

        // Gather into v from other processes
        if (mpi_size > 1)
            mgcl::mpi_util::gather(p.getMpiComm(), *v); // TODO check with different ghost amounts

        // check if solution is good
        auto err = calculateError(*solution, *v);
        auto errNorm = calculateErrorNorm(1.0 / (double)m, *err);
        auto errMax = calculateMaxError(*err);

        std::cout << "seq MPI Laplace" << std::endl;
        std::cout << "rank 0: " << std::endl;
        std::cout
            << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
            << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

        // Running this with 1 proc yields
        // ||e||_2 = 3.45809323000492415e-03
        // e_max = 3.56864935637552037e-03
        // which should be equal to the global result when run with multiple processors.

        REQUIRE(errNorm < 1e-2);
        REQUIRE(errMax < 1e-2);
        REQUIRE_THAT(errNorm, Catch::Matchers::WithinRel(3.45809323000492415e-03));
        REQUIRE_THAT(errMax, Catch::Matchers::WithinRel(3.56864935637552037e-03));
    }
}

// Tests if vcycle is correct for multiple processes with mpiLevelThreshold > 0, i.e. some fine levels are
// calculated distributed and only the coarse levels are calculated on rank 0.
// Run with e.g. mpiexec -n 2 tests_mpi MPI_vcycle_GPU_threshold_gt_0_Laplace7p
TEST_CASE("MPI_vcycle_GPU_threshold_gt_0_Laplace7p")
{
    using std::min;

    // global grid sizes
    int m = 16;
    int n = 16;
    int o = 16;
    int periodic = 1;
    int gh = 1;

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

    /* Initialize start and end for local grid */
    int m_start = (m / mpi_dims[0]) * mpi_coords[0] + min(mpi_coords[0], (m % mpi_dims[0]));
    int m_end = (m / mpi_dims[0]) * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (m % mpi_dims[0])) - 1;
    int n_start = (n / mpi_dims[1]) * mpi_coords[1] + min(mpi_coords[1], (n % mpi_dims[1]));
    int n_end = (n / mpi_dims[1]) * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (n % mpi_dims[1])) - 1;
    int o_start = (o / mpi_dims[2]) * mpi_coords[2] + min(mpi_coords[2], (o % mpi_dims[2]));
    int o_end = (o / mpi_dims[2]) * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (o % mpi_dims[2])) - 1;

    int ml = (m_end - m_start) + 1;
    int nl = (n_end - n_start) + 1;
    int ol = (o_end - o_start) + 1;

    CAPTURE(ml, nl, ol);

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
    REQUIRE(ml <= m);
    REQUIRE(nl > 0);
    REQUIRE(nl <= n);
    REQUIRE(ol > 0);
    REQUIRE(ol <= o);

    int threshold = GENERATE(1, 2, 3);
    CAPTURE(threshold);

    if (mpi_rank == 0)
        std::cerr << std::endl
                  << "Testing with threshold: " << threshold << std::endl;

    // Set up 4th order periodic problem
    int ghin = 0; // TODO check with ghin > 0
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    auto f = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    auto solution = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    mgcl_test::create4hOrderPeriodicProblem(*v, *f, *solution);

    // Create local slices
    std::shared_ptr<mgcl::Cuboid> vloc(v->slice(m_start, m_end, n_start, n_end, o_start, o_end));
    std::shared_ptr<mgcl::Cuboid> floc(f->slice(m_start, m_end, n_start, n_end, o_start, o_end));
    std::shared_ptr<mgcl::Cuboid> solutionloc(solution->slice(m_start, m_end, n_start, n_end, o_start, o_end));

    // Create local problem
    mgcl::Problem p(ml, nl, ol, floc, vloc, m, n, o);
    // p.setMaxiterVcycles(1);
    p.setUseOpencl(true);
    p.setReadResults(true);
    p.setGhosts(gh);
    p.setGhostsIn(ghin);
    p.setMpiLevelThreshold(threshold);
    p.setMpiComm(mpi_comm);

    p.solve();

    // Gather local approximations on rank 0 for checking.
    if (mpi_rank > 0)
        mgcl::mpi_util::gather(p.getMpiComm(), *vloc);
    else
    {
        // copy from vloc to v first
        for (int i = vloc->getGhostsM(); i < vloc->getM() + vloc->getGhostsM(); i++)
            for (int j = vloc->getGhostsN(); j < vloc->getN() + vloc->getGhostsN(); j++)
                for (int k = vloc->getGhostsO(); k < vloc->getO() + vloc->getGhostsO(); k++)
                {
                    (*v)[i][j][k] = (*vloc)[i][j][k];
                }

        // Gather into v from other processes
        if (mpi_size > 1)
            mgcl::mpi_util::gather(p.getMpiComm(), *v); // TODO check with different ghost amounts

        // check if solution is good
        auto err = calculateError(*solution, *v);
        auto errNorm = calculateErrorNorm(1.0 / (double)m, *err);
        auto errMax = calculateMaxError(*err);

        std::cout << "seq MPI Laplace" << std::endl;
        std::cout << "rank 0: " << std::endl;
        std::cout
            << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
            << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

        // Running this with 1 proc yields
        // ||e||_2 = 3.45809323000492415e-03
        // e_max = 3.56864935637552037e-03
        // which should be equal to the global result when run with multiple processors.

        REQUIRE(errNorm < 1e-2);
        REQUIRE(errMax < 1e-2);
        REQUIRE_THAT(errNorm, Catch::Matchers::WithinRel(3.45809323000492415e-03));
        REQUIRE_THAT(errMax, Catch::Matchers::WithinRel(3.56864935637552037e-03));
    }
}

// Tests if vcycle is correct for multiple processes with mpiLevelThreshold = 1, so stencil values are gathered
// for level 1 from level 0, which is a special case regarding the setup.
// Run with e.g. mpiexec -n 2 tests_mpi MPI_vcycle_threshold_eq_1_Varying27p
TEST_CASE("MPI_vcycle_threshold_eq_1_Varying27p")
{
    using std::min;

    // global grid sizes
    int m = 16;
    int n = 16;
    int o = 16;
    int periodic = 1;
    int gh = 1;

    // Problem parameters
    double tol = 1e-7;
    int nu1 = 2;
    int nu2 = 2;
    double omega = 0.8;
    int maxIterVCycles = 5;

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

    /* Initialize start and end for local grid */
    int m_start = (m / mpi_dims[0]) * mpi_coords[0] + min(mpi_coords[0], (m % mpi_dims[0]));
    int m_end = (m / mpi_dims[0]) * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (m % mpi_dims[0])) - 1;
    int n_start = (n / mpi_dims[1]) * mpi_coords[1] + min(mpi_coords[1], (n % mpi_dims[1]));
    int n_end = (n / mpi_dims[1]) * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (n % mpi_dims[1])) - 1;
    int o_start = (o / mpi_dims[2]) * mpi_coords[2] + min(mpi_coords[2], (o % mpi_dims[2]));
    int o_end = (o / mpi_dims[2]) * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (o % mpi_dims[2])) - 1;

    int ml = (m_end - m_start) + 1;
    int nl = (n_end - n_start) + 1;
    int ol = (o_end - o_start) + 1;

    CAPTURE(ml, nl, ol);

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
    REQUIRE(ml <= m);
    REQUIRE(nl > 0);
    REQUIRE(nl <= n);
    REQUIRE(ol > 0);
    REQUIRE(ol <= o);

    // Set up 4th order periodic problem
    int ghin = 0; // TODO check with ghin > 0
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    auto f = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    auto solution = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    mgcl_test::create4hOrderPeriodicProblem(*v, *f, *solution);

    // Create local slices
    std::shared_ptr<mgcl::Cuboid> vloc(v->slice(m_start, m_end, n_start, n_end, o_start, o_end));
    std::shared_ptr<mgcl::Cuboid> floc(f->slice(m_start, m_end, n_start, n_end, o_start, o_end));
    std::shared_ptr<mgcl::Cuboid> solutionloc(solution->slice(m_start, m_end, n_start, n_end, o_start, o_end));

    // Create local problem
    mgcl::Problem p(ml, nl, ol, floc, vloc, m, n, o);
    p.setMaxiterVcycles(maxIterVCycles);
    p.setOmega(omega);
    p.setNu1(nu1);
    p.setNu2(nu2);
    p.setTol(tol);
    p.setGhosts(gh);
    p.setGhostsIn(ghin);
    p.setMpiLevelThreshold(1);
    p.setMpiComm(mpi_comm);

    p.setStencilType(mgcl::MGCL_VARYING);
    auto& sv = *p.getStencilValues();

    if (mpi_rank == 0)
    {
        // For threshold level 0 or 1, stencilValues must have global sizes.
        REQUIRE(p.getMpiLevelThreshold() == 1);
        REQUIRE(sv.getM() == m);
        REQUIRE(sv.getN() == n);
        REQUIRE(sv.getO() == o);
    }

    mgcl_test::fill7pLaplace(sv, false);

    p.solveSeq();

    // Gather local approximations on rank 0 for checking.
    if (mpi_rank > 0)
        mgcl::mpi_util::gather(p.getMpiComm(), *vloc);
    else
    {
        // copy from vloc to v first
        for (int i = vloc->getGhostsM(); i < vloc->getM() + vloc->getGhostsM(); i++)
            for (int j = vloc->getGhostsN(); j < vloc->getN() + vloc->getGhostsN(); j++)
                for (int k = vloc->getGhostsO(); k < vloc->getO() + vloc->getGhostsO(); k++)
                {
                    (*v)[i][j][k] = (*vloc)[i][j][k];
                }

        // Gather into v from other processes
        if (mpi_size > 1)
            mgcl::mpi_util::gather(p.getMpiComm(), *v); // TODO check with different ghost amounts

        // check if solution is good
        auto err = calculateError(*solution, *v);
        auto errNorm = calculateErrorNorm(1.0 / (double)m, *err);
        auto errMax = calculateMaxError(*err);

        std::cout << "seq MPI Varying27p" << std::endl;
        std::cout << "rank 0: " << std::endl;
        std::cout
            << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
            << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

        // Running this with 1 proc yields
        // ||e||_2 = 2.79451354429798363e-03
        // e_max = 2.89860791352128345e-03
        // which should be equal to the global result when run with multiple processors.

        REQUIRE(errNorm < 1e-2);
        REQUIRE(errMax < 1e-2);
        REQUIRE_THAT(errNorm, Catch::Matchers::WithinRel(2.79451354429798363e-03));
        REQUIRE_THAT(errMax, Catch::Matchers::WithinRel(2.89860791352128345e-03));
    }
}

// Tests if vcycle is correct for multiple processes with mpiLevelThreshold = 2, so stencil values are gathered
// for level 2 from level 1. StencilValues on level 1 on rank 0 must have halve of global sizes.
// Run with e.g. mpiexec -n 2 tests_mpi MPI_vcycle_threshold_eq_2_Varying27p
TEST_CASE("MPI_vcycle_threshold_eq_2_Varying27p")
{
    using std::min;

    // global grid sizes
    int m = 16;
    int n = 16;
    int o = 16;
    int periodic = 1;
    int gh = 1;

    // Problem parameters
    double tol = 1e-7;
    int nu1 = 2;
    int nu2 = 2;
    double omega = 0.8;
    int maxIterVCycles = 5;

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

    /* Initialize start and end for local grid */
    int m_start = (m / mpi_dims[0]) * mpi_coords[0] + min(mpi_coords[0], (m % mpi_dims[0]));
    int m_end = (m / mpi_dims[0]) * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (m % mpi_dims[0])) - 1;
    int n_start = (n / mpi_dims[1]) * mpi_coords[1] + min(mpi_coords[1], (n % mpi_dims[1]));
    int n_end = (n / mpi_dims[1]) * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (n % mpi_dims[1])) - 1;
    int o_start = (o / mpi_dims[2]) * mpi_coords[2] + min(mpi_coords[2], (o % mpi_dims[2]));
    int o_end = (o / mpi_dims[2]) * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (o % mpi_dims[2])) - 1;

    int ml = (m_end - m_start) + 1;
    int nl = (n_end - n_start) + 1;
    int ol = (o_end - o_start) + 1;

    CAPTURE(ml, nl, ol);

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
    REQUIRE(ml <= m);
    REQUIRE(nl > 0);
    REQUIRE(nl <= n);
    REQUIRE(ol > 0);
    REQUIRE(ol <= o);

    // Set up 4th order periodic problem
    int ghin = 0; // TODO check with ghin > 0
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    auto f = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    auto solution = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    mgcl_test::create4hOrderPeriodicProblem(*v, *f, *solution);

    // Create local slices
    std::shared_ptr<mgcl::Cuboid> vloc(v->slice(m_start, m_end, n_start, n_end, o_start, o_end));
    std::shared_ptr<mgcl::Cuboid> floc(f->slice(m_start, m_end, n_start, n_end, o_start, o_end));
    std::shared_ptr<mgcl::Cuboid> solutionloc(solution->slice(m_start, m_end, n_start, n_end, o_start, o_end));

    // Create local problem
    mgcl::Problem p(ml, nl, ol, floc, vloc, m, n, o);
    p.setMaxiterVcycles(maxIterVCycles);
    p.setOmega(omega);
    p.setNu1(nu1);
    p.setNu2(nu2);
    p.setTol(tol);
    p.setGhosts(gh);
    p.setGhostsIn(ghin);
    p.setMpiLevelThreshold(2);
    p.setMpiComm(mpi_comm);

    p.setStencilType(mgcl::MGCL_VARYING);
    auto& sv = *p.getStencilValues();

    if (mpi_rank == 0)
    {
        // For threshold level 0 or 1, stencilValues must have global sizes, else local sizes.
        REQUIRE(p.getMpiLevelThreshold() == 2);
        REQUIRE(sv.getM() == ml);
        REQUIRE(sv.getN() == nl);
        REQUIRE(sv.getO() == ol);
    }

    mgcl_test::fill7pLaplace(sv, false);

    p.solveSeq();

    // Gather local approximations on rank 0 for checking.
    if (mpi_rank > 0)
        mgcl::mpi_util::gather(p.getMpiComm(), *vloc);
    else
    {
        // copy from vloc to v first
        for (int i = vloc->getGhostsM(); i < vloc->getM() + vloc->getGhostsM(); i++)
            for (int j = vloc->getGhostsN(); j < vloc->getN() + vloc->getGhostsN(); j++)
                for (int k = vloc->getGhostsO(); k < vloc->getO() + vloc->getGhostsO(); k++)
                {
                    (*v)[i][j][k] = (*vloc)[i][j][k];
                }

        // Gather into v from other processes
        if (mpi_size > 1)
            mgcl::mpi_util::gather(p.getMpiComm(), *v); // TODO check with different ghost amounts

        // check if solution is good
        auto err = calculateError(*solution, *v);
        auto errNorm = calculateErrorNorm(1.0 / (double)m, *err);
        auto errMax = calculateMaxError(*err);

        std::cout << "seq MPI Varying27p" << std::endl;
        std::cout << "rank 0: " << std::endl;
        std::cout
            << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
            << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

        // Running this with 1 proc yields
        // ||e||_2 = 2.79451354429798363e-03
        // e_max = 2.89860791352128345e-03
        // which should be equal to the global result when run with multiple processors.

        REQUIRE(errNorm < 1e-2);
        REQUIRE(errMax < 1e-2);
        REQUIRE_THAT(errNorm, Catch::Matchers::WithinRel(2.79451354429798363e-03));
        REQUIRE_THAT(errMax, Catch::Matchers::WithinRel(2.89860791352128345e-03));
    }
}

// Tests if vcycle is correct for multiple processes using OCL with mpiLevelThreshold = 1, so
//  stencil values are gathered for level 1 from level 0, which is a special case regarding the setup.
// Run with e.g. mpiexec -n 2 tests_mpi MPI_vcycle_GPU_threshold_eq_1_Varying27p
TEST_CASE("MPI_vcycle_GPU_threshold_eq_1_Varying27p")
{
    using std::min;

    // global grid sizes
    int m = 16;
    int n = 16;
    int o = 16;
    int periodic = 1;
    int gh = 1;

    // Problem parameters
    double tol = 1e-7;
    int nu1 = 2;
    int nu2 = 2;
    double omega = 0.8;
    int maxIterVCycles = 5;

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

    /* Initialize start and end for local grid */
    int m_start = (m / mpi_dims[0]) * mpi_coords[0] + min(mpi_coords[0], (m % mpi_dims[0]));
    int m_end = (m / mpi_dims[0]) * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (m % mpi_dims[0])) - 1;
    int n_start = (n / mpi_dims[1]) * mpi_coords[1] + min(mpi_coords[1], (n % mpi_dims[1]));
    int n_end = (n / mpi_dims[1]) * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (n % mpi_dims[1])) - 1;
    int o_start = (o / mpi_dims[2]) * mpi_coords[2] + min(mpi_coords[2], (o % mpi_dims[2]));
    int o_end = (o / mpi_dims[2]) * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (o % mpi_dims[2])) - 1;

    int ml = (m_end - m_start) + 1;
    int nl = (n_end - n_start) + 1;
    int ol = (o_end - o_start) + 1;

    CAPTURE(ml, nl, ol);

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
    REQUIRE(ml <= m);
    REQUIRE(nl > 0);
    REQUIRE(nl <= n);
    REQUIRE(ol > 0);
    REQUIRE(ol <= o);

    // Set up 4th order periodic problem
    int ghin = 0; // TODO check with ghin > 0
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    auto f = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    auto solution = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    mgcl_test::create4hOrderPeriodicProblem(*v, *f, *solution);

    // Create local slices
    std::shared_ptr<mgcl::Cuboid> vloc(v->slice(m_start, m_end, n_start, n_end, o_start, o_end));
    std::shared_ptr<mgcl::Cuboid> floc(f->slice(m_start, m_end, n_start, n_end, o_start, o_end));
    std::shared_ptr<mgcl::Cuboid> solutionloc(solution->slice(m_start, m_end, n_start, n_end, o_start, o_end));

    // Create local problem
    mgcl::Problem p(ml, nl, ol, floc, vloc, m, n, o);
    p.setMaxiterVcycles(maxIterVCycles);
    p.setUseOpencl(true);
    p.setReadResults(true);
    p.setOmega(omega);
    p.setNu1(nu1);
    p.setNu2(nu2);
    p.setTol(tol);
    p.setGhosts(gh);
    p.setGhostsIn(ghin);
    p.setMpiLevelThreshold(1);
    p.setMpiComm(mpi_comm);

    p.setStencilType(mgcl::MGCL_VARYING);
    auto& sv = *p.getStencilValues();

    if (mpi_rank == 0)
    {
        // For threshold level 0 or 1, stencilValues must have global sizes.
        REQUIRE(p.getMpiLevelThreshold() == 1);
        REQUIRE(sv.getM() == m);
        REQUIRE(sv.getN() == n);
        REQUIRE(sv.getO() == o);
    }

    mgcl_test::fill7pLaplace(sv, false);

    p.solve();

    // Gather local approximations on rank 0 for checking.
    if (mpi_rank > 0)
        mgcl::mpi_util::gather(p.getMpiComm(), *vloc);
    else
    {
        // copy from vloc to v first
        for (int i = vloc->getGhostsM(); i < vloc->getM() + vloc->getGhostsM(); i++)
            for (int j = vloc->getGhostsN(); j < vloc->getN() + vloc->getGhostsN(); j++)
                for (int k = vloc->getGhostsO(); k < vloc->getO() + vloc->getGhostsO(); k++)
                {
                    (*v)[i][j][k] = (*vloc)[i][j][k];
                }

        // Gather into v from other processes
        if (mpi_size > 1)
            mgcl::mpi_util::gather(p.getMpiComm(), *v); // TODO check with different ghost amounts

        // check if solution is good
        auto err = calculateError(*solution, *v);
        auto errNorm = calculateErrorNorm(1.0 / (double)m, *err);
        auto errMax = calculateMaxError(*err);

        std::cout << "seq MPI Varying27p" << std::endl;
        std::cout << "rank 0: " << std::endl;
        std::cout
            << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
            << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

        // Running this with 1 proc yields
        // ||e||_2 = 2.79451354429819309e-03
        // e_max = 2.89860791352150159e-03
        // which should be equal to the global result when run with multiple processors.

        REQUIRE(errNorm < 1e-2);
        REQUIRE(errMax < 1e-2);
        REQUIRE_THAT(errNorm, Catch::Matchers::WithinRel(2.79451354429819309e-03));
        REQUIRE_THAT(errMax, Catch::Matchers::WithinRel(2.89860791352150159e-03));
    }
}

// Tests if vcycle is correct for multiple processes using OCL with mpiLevelThreshold = 2,
// so stencil values are gathered for level 2 from level 1.
// StencilValues on level 1 on rank 0 must have halve of global sizes.
// Run with e.g. mpiexec -n 2 tests_mpi MPI_vcycle_GPU_threshold_eq_2_Varying27p
TEST_CASE("MPI_vcycle_GPU_threshold_eq_2_Varying27p")
{
    using std::min;

    // global grid sizes
    int m = 16;
    int n = 16;
    int o = 16;
    int periodic = 1;
    int gh = 1;

    // Problem parameters
    double tol = 1e-7;
    int nu1 = 2;
    int nu2 = 2;
    double omega = 0.8;
    int maxIterVCycles = 5;

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

    /* Initialize start and end for local grid */
    int m_start = (m / mpi_dims[0]) * mpi_coords[0] + min(mpi_coords[0], (m % mpi_dims[0]));
    int m_end = (m / mpi_dims[0]) * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (m % mpi_dims[0])) - 1;
    int n_start = (n / mpi_dims[1]) * mpi_coords[1] + min(mpi_coords[1], (n % mpi_dims[1]));
    int n_end = (n / mpi_dims[1]) * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (n % mpi_dims[1])) - 1;
    int o_start = (o / mpi_dims[2]) * mpi_coords[2] + min(mpi_coords[2], (o % mpi_dims[2]));
    int o_end = (o / mpi_dims[2]) * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (o % mpi_dims[2])) - 1;

    int ml = (m_end - m_start) + 1;
    int nl = (n_end - n_start) + 1;
    int ol = (o_end - o_start) + 1;

    CAPTURE(ml, nl, ol);

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
    REQUIRE(ml <= m);
    REQUIRE(nl > 0);
    REQUIRE(nl <= n);
    REQUIRE(ol > 0);
    REQUIRE(ol <= o);

    // Set up 4th order periodic problem
    int ghin = 0; // TODO check with ghin > 0
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    auto f = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    auto solution = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    mgcl_test::create4hOrderPeriodicProblem(*v, *f, *solution);

    // Create local slices
    std::shared_ptr<mgcl::Cuboid> vloc(v->slice(m_start, m_end, n_start, n_end, o_start, o_end));
    std::shared_ptr<mgcl::Cuboid> floc(f->slice(m_start, m_end, n_start, n_end, o_start, o_end));
    std::shared_ptr<mgcl::Cuboid> solutionloc(solution->slice(m_start, m_end, n_start, n_end, o_start, o_end));

    // Create local problem
    mgcl::Problem p(ml, nl, ol, floc, vloc, m, n, o);
    p.setMaxiterVcycles(maxIterVCycles);
    p.setUseOpencl(true);
    p.setReadResults(true);
    p.setOmega(omega);
    p.setNu1(nu1);
    p.setNu2(nu2);
    p.setTol(tol);
    p.setGhosts(gh);
    p.setGhostsIn(ghin);
    p.setMpiLevelThreshold(2);
    p.setMpiComm(mpi_comm);

    p.setStencilType(mgcl::MGCL_VARYING);
    auto& sv = *p.getStencilValues();

    if (mpi_rank == 0)
    {
        // For threshold level 0 or 1, stencilValues must have global sizes, else local sizes.
        REQUIRE(p.getMpiLevelThreshold() == 2);
        REQUIRE(sv.getM() == ml);
        REQUIRE(sv.getN() == nl);
        REQUIRE(sv.getO() == ol);
    }

    mgcl_test::fill7pLaplace(sv, false);

    p.solve();

    // Gather local approximations on rank 0 for checking.
    if (mpi_rank > 0)
        mgcl::mpi_util::gather(p.getMpiComm(), *vloc);
    else
    {
        // copy from vloc to v first
        for (int i = vloc->getGhostsM(); i < vloc->getM() + vloc->getGhostsM(); i++)
            for (int j = vloc->getGhostsN(); j < vloc->getN() + vloc->getGhostsN(); j++)
                for (int k = vloc->getGhostsO(); k < vloc->getO() + vloc->getGhostsO(); k++)
                {
                    (*v)[i][j][k] = (*vloc)[i][j][k];
                }

        // Gather into v from other processes
        if (mpi_size > 1)
            mgcl::mpi_util::gather(p.getMpiComm(), *v); // TODO check with different ghost amounts

        // check if solution is good
        auto err = calculateError(*solution, *v);
        auto errNorm = calculateErrorNorm(1.0 / (double)m, *err);
        auto errMax = calculateMaxError(*err);

        std::cout << "seq MPI Varying27p" << std::endl;
        std::cout << "rank 0: " << std::endl;
        std::cout
            << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
            << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

        // Running this with 1 proc yields
        // ||e||_2 = 2.79451354429819309e-03
        // e_max = 2.89860791352150159e-03
        // which should be equal to the global result when run with multiple processors.

        REQUIRE(errNorm < 1e-2);
        REQUIRE(errMax < 1e-2);
        REQUIRE_THAT(errNorm, Catch::Matchers::WithinRel(2.79451354429819309e-03));
        REQUIRE_THAT(errMax, Catch::Matchers::WithinRel(2.89860791352150159e-03));
    }
}

// Tests if vcycle is correct for multiple processes using OCL with mpiLevelThreshold = 2,
// so stencil values are gathered for level 2 from level 1.
// Also uses different amounts of jacobi iterations without ghost update in-between.
// StencilValues on level 1 on rank 0 must have halve of global sizes.
// Run with e.g. mpiexec -n 2 tests_mpi MPI_vcycle_GPU_threshold_eq_2_Varying27p_multiple_jacobi_iters
TEST_CASE("MPI_vcycle_GPU_threshold_eq_2_Varying27p_multiple_jacobi_iters")
{
    using std::min;

    // global grid sizes
    int m = 16;
    int n = 16;
    int o = 16;
    int periodic = 1;

    // Problem parameters
    double tol = 1e-7;
    int nu1 = 2;
    int nu2 = 2;
    double omega = 0.8;
    int maxIterVCycles = 5;
    int jacobiItersPerKernel = 3;
    int gh = jacobiItersPerKernel;

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

    /* Initialize start and end for local grid */
    int m_start = (m / mpi_dims[0]) * mpi_coords[0] + min(mpi_coords[0], (m % mpi_dims[0]));
    int m_end = (m / mpi_dims[0]) * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (m % mpi_dims[0])) - 1;
    int n_start = (n / mpi_dims[1]) * mpi_coords[1] + min(mpi_coords[1], (n % mpi_dims[1]));
    int n_end = (n / mpi_dims[1]) * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (n % mpi_dims[1])) - 1;
    int o_start = (o / mpi_dims[2]) * mpi_coords[2] + min(mpi_coords[2], (o % mpi_dims[2]));
    int o_end = (o / mpi_dims[2]) * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (o % mpi_dims[2])) - 1;

    int ml = (m_end - m_start) + 1;
    int nl = (n_end - n_start) + 1;
    int ol = (o_end - o_start) + 1;

    CAPTURE(ml, nl, ol);

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
    REQUIRE(ml <= m);
    REQUIRE(nl > 0);
    REQUIRE(nl <= n);
    REQUIRE(ol > 0);
    REQUIRE(ol <= o);

    // Set up 4th order periodic problem
    int ghin = 0; // TODO check with ghin > 0
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    auto f = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    auto solution = std::make_shared<mgcl::Cuboid>(m, n, o, ghin, ghin, ghin);
    mgcl_test::create4hOrderPeriodicProblem(*v, *f, *solution);

    // Create local slices
    std::shared_ptr<mgcl::Cuboid> vloc(v->slice(m_start, m_end, n_start, n_end, o_start, o_end));
    std::shared_ptr<mgcl::Cuboid> floc(f->slice(m_start, m_end, n_start, n_end, o_start, o_end));
    std::shared_ptr<mgcl::Cuboid> solutionloc(solution->slice(m_start, m_end, n_start, n_end, o_start, o_end));

    // Create local problem
    mgcl::Problem p(ml, nl, ol, floc, vloc, m, n, o);
    p.setMaxiterVcycles(maxIterVCycles);
    p.setUseOpencl(true);
    p.setReadResults(true);
    p.setOmega(omega);
    p.setNu1(nu1);
    p.setNu2(nu2);
    p.setTol(tol);
    p.setGhosts(gh);
    p.setGhostsIn(ghin);
    p.setMpiLevelThreshold(2);
    p.setMpiComm(mpi_comm);
    p.setJacobiIterationsPerKernel(jacobiItersPerKernel);

    p.setStencilType(mgcl::MGCL_VARYING);
    auto& sv = *p.getStencilValues();

    if (mpi_rank == 0)
    {
        // For threshold level 0 or 1, stencilValues must have global sizes, else local sizes.
        REQUIRE(p.getMpiLevelThreshold() == 2);
        REQUIRE(sv.getM() == ml);
        REQUIRE(sv.getN() == nl);
        REQUIRE(sv.getO() == ol);
    }

    mgcl_test::fill7pLaplace(sv, false);

    p.solve();

    // Gather local approximations on rank 0 for checking.
    if (mpi_rank > 0)
        mgcl::mpi_util::gather(p.getMpiComm(), *vloc);
    else
    {
        // copy from vloc to v first
        for (int i = vloc->getGhostsM(); i < vloc->getM() + vloc->getGhostsM(); i++)
            for (int j = vloc->getGhostsN(); j < vloc->getN() + vloc->getGhostsN(); j++)
                for (int k = vloc->getGhostsO(); k < vloc->getO() + vloc->getGhostsO(); k++)
                {
                    (*v)[i][j][k] = (*vloc)[i][j][k];
                }

        // Gather into v from other processes
        if (mpi_size > 1)
            mgcl::mpi_util::gather(p.getMpiComm(), *v); // TODO check with different ghost amounts

        // check if solution is good
        auto err = calculateError(*solution, *v);
        auto errNorm = calculateErrorNorm(1.0 / (double)m, *err);
        auto errMax = calculateMaxError(*err);

        std::cout << "seq MPI Varying27p" << std::endl;
        std::cout << "rank 0: " << std::endl;
        std::cout
            << std::scientific << std::setprecision(17) << "  ||e||_2 = " << errNorm << std::endl
            << std::scientific << std::setprecision(17) << "  e_max = " << errMax << std::endl;

        // Running this with 1 proc yields
        // ||e||_2 = 2.79451354429819309e-03
        // e_max = 2.89860791352150159e-03
        // which should be equal to the global result when run with multiple processors.

        REQUIRE(errNorm < 1e-2);
        REQUIRE(errMax < 1e-2);
        REQUIRE_THAT(errNorm, Catch::Matchers::WithinRel(2.79451354429819309e-03));
        REQUIRE_THAT(errMax, Catch::Matchers::WithinRel(2.89860791352150159e-03));
    }
}

/**
 * @brief Calculates error for each cell, e.g. difference between solution and approximation. Dimensions must match.
 *
 * @param solution
 * @param approximation
 * @return std::shared_ptr<mgcl::Cuboid>
 */
std::shared_ptr<mgcl::Cuboid> calculateError(mgcl::Cuboid& solution, mgcl::Cuboid& approximation)
{
    if (solution.getM() != approximation.getM() ||
        solution.getN() != approximation.getN() ||
        solution.getO() != approximation.getO())
        throw std::invalid_argument("Dimensions do not match.");

    auto ret = std::make_shared<mgcl::Cuboid>(solution.getM(), solution.getN(), solution.getO());
    for (int i = 0, is = solution.getGhostsM(), ia = approximation.getGhostsM(); is < solution.getMgh() - solution.getGhostsM(); i++, is++, ia++)
        for (int j = 0, js = solution.getGhostsN(), ja = approximation.getGhostsN(); js < solution.getNgh() - solution.getGhostsN(); j++, js++, ja++)
            for (int k = 0, ks = solution.getGhostsO(), ka = approximation.getGhostsO(); ks < solution.getOgh() - solution.getGhostsO(); k++, ks++, ka++)
            {
                (*ret)[i][j][k] = fabs(solution[is][js][ks] - approximation[ia][ja][ka]);
            }

    return ret;
}

/**
 * @brief Returns the maximum absolute error. calculateError should have been called first.
 *
 * @param error
 * @return double
 */
double calculateMaxError(mgcl::Cuboid& error)
{
    double max = 0;
    for (int i = 0; i < error.getM(); i++)
        for (int j = 0; j < error.getN(); j++)
            for (int k = 0; k < error.getO(); k++)
            {
                if (max < error[i][j][k])
                    max = error[i][j][k];
            }
    return max;
}

/**
 * @brief Returns the 2-norm of the given error. calculateError should have been called first.
 *
 * @param h width of one cell
 * @param error precalculated error per cell
 * @return double Error norm of form ||e||_2 * h^3
 */
double calculateErrorNorm(double h, mgcl::Cuboid& error)
{
    double sum = 0;

    for (int i = 0; i < error.getM(); i++)
        for (int j = 0; j < error.getN(); j++)
            for (int k = 0; k < error.getO(); k++)
            {
                sum += error[i][j][k] * error[i][j][k];
            }

    return sqrt(sum * h * h * h);
}