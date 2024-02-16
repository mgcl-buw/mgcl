#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <iostream>
#include <memory>

#include "../../src/mgcl/cuboid.hpp"
#include "../../src/mgcl/multigrid_engine.hpp"
#include "../../src/mgcl/problem.hpp"
#include "../test_utility.hpp"

#include "mpi.h"

// Checks ghost update for any number of processes that is allowed by mgcl, e.g. 1, 2, 4, 8, 24.
// Run with: mpiexec -n 8 tests_mpi "MPI jacobiSeq (n processes)"
// TODO non-periodic
TEST_CASE("MPI jacobiSeq (n processes)", "[mpiN]")
{
    using std::min;

    // global grid sizes
    int m = 16;
    int n = 16;
    int o = 16;
    int periodic = 1;

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

    double omega = 0.8;
    double h = 1.0 / ((double)m);
    mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_27POINT;
    double stencilFactor = 1.0 / (30.0 * h * h);
    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
    int maxiter = 10;

    int gh = 1;

    // global test data
    mgcl::Cuboid v_glob(m, n, o, gh, gh, gh);
    mgcl::Cuboid r_glob(m, n, o, gh, gh, gh);
    mgcl::Cuboid f_glob(m, n, o, gh, gh, gh);

    // Fill with 4th order periodic Problem
    mgcl::Cuboid solution(m, n, o);
    mgcl_test::create4hOrderPeriodicProblem(v_glob, f_glob, solution);

    // Create local slices of global data
    std::shared_ptr<mgcl::Cuboid> vlptr = v_glob.slice(m_start, m_end, n_start, n_end, o_start, o_end);
    auto& v_loc = *vlptr;
    std::shared_ptr<mgcl::Cuboid> flptr = f_glob.slice(m_start, m_end, n_start, n_end, o_start, o_end);
    auto& f_loc = *flptr;
    std::shared_ptr<mgcl::Cuboid> rlptr = r_glob.slice(m_start, m_end, n_start, n_end, o_start, o_end);
    auto& r_loc = *rlptr;

    // Update ghosts of test data
    // mgcl::MultigridEngine::updateGhostsSeq(vl, mpiData, true);

    // Init Problem to create all needed structures
    mgcl::Problem p(ml, nl, ol, vlptr, flptr, m, n, o);
    p.setGhosts(gh);
    p.setGhostsIn(gh);
    p.setMpiComm(mpi_comm);
    p.setStencilType(stencilType);
    p.init();

    // Check on level 0
    auto& lv = p.getLevelAt(0);
    auto mpiData = lv.getMpiDataPtr();

    // print neighbours per rank
    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //         std::cout << mpi_rank << ": " << mpiData->left << "," << mpiData->right << ","
    //                   << mpiData->up << "," << mpiData->down << ","
    //                   << mpiData->back << "," << mpiData->front << std::endl;
    // }

    // Update ghosts of expected result locally, i.e. not using MPI routines.
    mgcl::MultigridEngine::jacobiSeq(v_glob, f_glob, r_glob, omega, h * h, maxiter, resnorm, stencilType,
                                     stencilFactor, nullptr, true, true, false, 1, nullptr);

    mgcl::MultigridEngine::jacobiSeq(v_loc, f_loc, r_loc, omega, h * h, maxiter, resnorm, stencilType,
                                     stencilFactor, nullptr, true, true, false, 1, mpiData);

    for (int i = gh; i < ml + gh; i++)
        for (int j = gh; j < nl + gh; j++)
            for (int k = gh; k < ol + gh; k++)
            {
                v_loc[i][j][k] = v_glob[i + m_start][j + n_start][k + o_start];
            }

    // cl.dumpToFile("cl" + std::to_string(mpi_rank) + ".txt");

    // if (mpi_rank == 0)
    // {
    //     cg.dumpToFile("cg.txt");
}

// Checks ghost update for any number of processes that is allowed by mgcl, e.g. 1, 2, 4, 8, 24.
// Run with: mpiexec -n 8 tests_mpi "MPI jacobi ocl Laplace (n processes)"
// TODO non-periodic
TEST_CASE("MPI jacobi ocl Laplace (n processes)", "[mpiN]")
{
    using std::min;

    // global grid sizes
    int m = 16;
    int n = 16;
    int o = 16;
    int periodic = 1;

    int stepsPerIter = GENERATE(1, 2);
    CAPTURE(stepsPerIter);

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

    double omega = 0.8;
    double h = 1.0 / ((double)m);
    mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_LAPLACE_27POINT;
    double stencilFactor = 1.0 / (30.0 * h * h);
    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
    int maxiter = 10;

    int gh = stepsPerIter;

    // global test data
    mgcl::Cuboid v_glob(m, n, o, gh, gh, gh);
    mgcl::Cuboid r_glob(m, n, o, gh, gh, gh);
    mgcl::Cuboid f_glob(m, n, o, gh, gh, gh);

    // Fill with 4th order periodic Problem
    mgcl::Cuboid solution(m, n, o);
    mgcl_test::create4hOrderPeriodicProblem(v_glob, f_glob, solution);

    // Create local slices of global data
    std::shared_ptr<mgcl::Cuboid> vlptr = v_glob.slice(m_start, m_end, n_start, n_end, o_start, o_end);
    auto& v_loc = *vlptr;
    std::shared_ptr<mgcl::Cuboid> flptr = f_glob.slice(m_start, m_end, n_start, n_end, o_start, o_end);
    auto& f_loc = *flptr;

    // Update ghosts of test data
    // mgcl::MultigridEngine::updateGhostsSeq(vl, mpiData, true);

    // Init Problem to create all needed structures
    auto pptr = std::make_shared<mgcl::Problem>(ml, nl, ol, vlptr, flptr, m, n, o);
    auto& p = *pptr;
    p.setUseOpencl(true);
    p.setGhosts(gh);
    p.setGhostsIn(gh);
    p.setMpiComm(mpi_comm);
    p.setStencilType(stencilType);
    p.setResidualNorm(resnorm);
    p.init();

    mgcl_test::TestUtility tu(pptr);

    // Check on level 0
    auto& lv = p.getLevelAt(0);
    auto mpiData = lv.getMpiDataPtr();

    // print neighbours per rank
    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //         std::cout << mpi_rank << ": " << mpiData->left << "," << mpiData->right << ","
    //                   << mpiData->up << "," << mpiData->down << ","
    //                   << mpiData->back << "," << mpiData->front << std::endl;
    // }

    // Run Jacobi on global dataset for the expected result.
    mgcl::MultigridEngine::jacobiSeq(v_glob, f_glob, r_glob, omega, h * h, maxiter, resnorm, stencilType,
                                     stencilFactor, nullptr, true, true, false, stepsPerIter, nullptr);

    mgcl::MultigridEngine::jacobi(p, lv, maxiter, true, stepsPerIter);
    tu.finish();

    auto v_loc_ret_ptr = lv.getDVIn().read(tu.getCommands(), nullptr, true);
    auto& v_loc_ret = *v_loc_ret_ptr;

    for (int i = gh; i < ml + gh; i++)
        for (int j = gh; j < nl + gh; j++)
            for (int k = gh; k < ol + gh; k++)
            {
                v_loc_ret[i][j][k] = v_glob[i + m_start][j + n_start][k + o_start];
            }

    // cl.dumpToFile("cl" + std::to_string(mpi_rank) + ".txt");

    // if (mpi_rank == 0)
    // {
    //     cg.dumpToFile("cg.txt");
}

// Checks Jacobi using a VaryingStencil for any number of processes that is allowed by mgcl, e.g. 1, 2, 4, 8, 24.
// Use an arbitrary stencil with no duplicate values (result is only checked against global jacobi, not
// against the actual solution).
// Run with: mpiexec -n 8 tests_mpi "MPI_jacobi_ocl_VaryingStencil_(n_processes)"
// TODO non-periodic
TEST_CASE("MPI_jacobi_ocl_VaryingStencil_(n_processes)", "[mpiN]")
{
    using std::min;

    // global grid sizes
    int m = 16;
    int n = 16;
    int o = 16;
    int periodic = 1;

    int stepsPerIter = GENERATE(1, 2);
    CAPTURE(stepsPerIter);

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

    double omega = 0.8;
    double h = 1.0 / ((double)m);
    mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_VARYING;
    double stencilFactor = 1.0 / (30.0 * h * h);
    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
    int maxiter = 10;

    int gh = stepsPerIter;

    // global test data
    mgcl::Cuboid v_glob(m, n, o, gh, gh, gh);
    mgcl::Cuboid r_glob(m, n, o, gh, gh, gh);
    mgcl::Cuboid f_glob(m, n, o, gh, gh, gh);

    // Fill with 4th order periodic Problem
    mgcl::Cuboid solution(m, n, o);
    mgcl_test::create4hOrderPeriodicProblem(v_glob, f_glob, solution);

    // Create global stencilValues seperately since its bigger
    int svgh = std::max(gh, 2);
    mgcl::VaryingStencil sv_glob(m, n, o, 3, svgh, svgh, svgh);
    // mgcl_test::fill7pLaplace(sv_glob, h, false);
    for (int i = 0; i < sv_glob.field1d().size(); i++)
        sv_glob.field1d()[i] = i;

    // Create local slices of global data
    std::shared_ptr<mgcl::Cuboid> vlptr = v_glob.slice(m_start, m_end, n_start, n_end, o_start, o_end);
    auto& v_loc = *vlptr;
    std::shared_ptr<mgcl::Cuboid> flptr = f_glob.slice(m_start, m_end, n_start, n_end, o_start, o_end);
    auto& f_loc = *flptr;

    // Update ghosts of test data
    // mgcl::MultigridEngine::updateGhostsSeq(vl, mpiData, true);

    // Init Problem to create all needed structures
    auto pptr = std::make_shared<mgcl::Problem>(ml, nl, ol, vlptr, flptr, m, n, o);
    auto& p = *pptr;
    p.setUseOpencl(true);
    p.setGhosts(gh);
    p.setGhostsIn(gh);
    p.setMpiComm(mpi_comm);
    p.setStencilType(stencilType);
    p.setResidualNorm(resnorm);

    auto svptr = p.getStencilValues();
    auto& sv = *svptr;
    // mgcl_test::fill7pLaplace(sv, h, false);
    for (int i = 0; i < sv.field1d().size(); i++)
        sv.field1d()[i] = i;

    p.init();

    mgcl_test::TestUtility tu(pptr);

    // Check on level 0
    auto& lv = p.getLevelAt(0);
    auto mpiData = lv.getMpiDataPtr();

    // print neighbours per rank
    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //         std::cout << mpi_rank << ": " << mpiData->left << "," << mpiData->right << ","
    //                   << mpiData->up << "," << mpiData->down << ","
    //                   << mpiData->back << "," << mpiData->front << std::endl;
    // }

    // Run Jacobi on global dataset for the expected result.
    mgcl::MultigridEngine::jacobiSeq(v_glob, f_glob, r_glob, omega, h * h, maxiter, resnorm, stencilType,
                                     stencilFactor, &sv_glob, true, true, false, stepsPerIter, nullptr);

    mgcl::MultigridEngine::jacobi(p, lv, maxiter, true, stepsPerIter);
    tu.finish();

    auto v_loc_ret_ptr = lv.getDVIn().read(tu.getCommands(), nullptr, true);
    auto& v_loc_ret = *v_loc_ret_ptr;

    for (int i = gh; i < ml + gh; i++)
        for (int j = gh; j < nl + gh; j++)
            for (int k = gh; k < ol + gh; k++)
            {
                v_loc_ret[i][j][k] = v_glob[i + m_start][j + n_start][k + o_start];
            }

    // cl.dumpToFile("cl" + std::to_string(mpi_rank) + ".txt");

    // if (mpi_rank == 0)
    // {
    //     cg.dumpToFile("cg.txt");
}