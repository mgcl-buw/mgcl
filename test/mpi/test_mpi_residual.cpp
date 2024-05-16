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

// Checks Residual for any number of processes that is allowed by mgcl, e.g. 1, 2, 4, 8, 24.
// Run with: mpiexec -n 8 tests_mpi MPI_residual_seq_VaryingStencil_n_processes
// TODO non-periodic
TEST_CASE("MPI_residual_seq_VaryingStencil_n_processes", "[mpiN]")
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
    mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_VARYING;
    double stencilFactor = 1.0 / (30.0 * h * h);
    mgcl::MGCL_RESIDUAL_NORM resnorm = mgcl::MGCL_L2;
    int maxiter = 10;

    int gh = 1;

    // global test data
    mgcl::Cuboid v_glob(m, n, o, gh, gh, gh);
    mgcl::Cuboid r_glob(m, n, o, gh, gh, gh);
    mgcl::Cuboid f_glob(m, n, o, gh, gh, gh);
    mgcl::VaryingStencil sv_glob(m, n, o, 3, 2, 2, 2);
    for (int i = 0; i < sv_glob.field1d().size(); i++)
        sv_glob.field1d()[i] = i;

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
    std::shared_ptr<mgcl::VaryingStencil> svptr_loc_slice = sv_glob.slice(m_start, m_end, n_start, n_end, o_start, o_end);
    auto& sv_loc_slice = *svptr_loc_slice;

    // Update ghosts of test data
    // mgcl::MultigridEngine::updateGhostsSeq(vl, mpiData, true);

    // Init Problem to create all needed structures
    mgcl::Problem p(ml, nl, ol, flptr, vlptr, m, n, o);
    p.setGhosts(gh);
    p.setGhostsIn(gh);
    p.setMpiComm(mpi_comm);
    p.setStencilType(stencilType);

    auto svptr = p.getStencilValues();
    auto& sv = *svptr;

    // check that dimensions of sliced stencilValues matches with the one from the problem
    REQUIRE(sv.getM() == sv_loc_slice.getM());
    REQUIRE(sv.getN() == sv_loc_slice.getN());
    REQUIRE(sv.getO() == sv_loc_slice.getO());
    REQUIRE(sv.getGhostsM() == sv_loc_slice.getGhostsM());
    REQUIRE(sv.getGhostsN() == sv_loc_slice.getGhostsN());
    REQUIRE(sv.getGhostsO() == sv_loc_slice.getGhostsO());

    for (int i = 0; i < sv.field1d().size(); i++)
        sv.field1d()[i] = sv_loc_slice.field1d()[i];

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
    mgcl::MultigridEngine::residualSeq(f_glob, v_glob, r_glob, resnorm, stencilType,
                                       stencilFactor, &sv_glob, true, true, true, 0, 0, 0, nullptr);

    mgcl::MultigridEngine::residualSeq(f_loc, v_loc, r_loc, resnorm, stencilType,
                                       stencilFactor, svptr.get(), true, true, false, 0, 0, 0, mpiData);

    // Compare local results with respective chunk of global results
    for (int il = r_glob.getGhostsM(), ig = mpi_coords[0] * ml + r_loc.getGhostsM(); il < r_loc.getM() + r_glob.getGhostsM(); il++, ig++)
        for (int jl = r_glob.getGhostsN(), jg = mpi_coords[1] * nl + r_loc.getGhostsN(); jl < r_loc.getN() + r_glob.getGhostsN(); jl++, jg++)
            for (int kl = r_glob.getGhostsO(), kg = mpi_coords[2] * ol + r_loc.getGhostsO(); kl < r_loc.getO() + r_glob.getGhostsO(); kl++, kg++)
            {
                CAPTURE(il, jl, kl, ig, jg, kg);
                REQUIRE(r_loc[il][jl][kl] == r_glob[ig][jg][kg]);
                REQUIRE(v_loc[il][jl][kl] == v_glob[ig][jg][kg]);
                REQUIRE(f_loc[il][jl][kl] == f_glob[ig][jg][kg]);
            }
}

// Checks Residual using a VaryingStencil for any number of processes that is allowed by mgcl, e.g. 1, 2, 4, 8, 24.
// Use an arbitrary stencil with no duplicate values (result is only checked against global jacobi, not
// against the actual solution).
// Run with: mpiexec -n 8 tests_mpi MPI_residual_ocl_VaryingStencil_n_processes
// TODO non-periodic
TEST_CASE("MPI_residual_ocl_VaryingStencil_n_processes", "[mpiN]")
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

    CAPTURE(mpi_rank);

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

    int gh = 1;

    // global test data
    auto v_glob_ptr = std::make_shared<mgcl::Cuboid>(m, n, o, gh, gh, gh);
    auto f_glob_ptr = std::make_shared<mgcl::Cuboid>(m, n, o, gh, gh, gh);
    auto& v_glob = *v_glob_ptr;
    auto& f_glob = *f_glob_ptr;

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

    // Init global problem to create all needed structures
    auto pptr_glob = std::make_shared<mgcl::Problem>(m, n, o, f_glob_ptr, v_glob_ptr);
    auto& p_glob = *pptr_glob;
    p_glob.setDeviceType(CL_DEVICE_TYPE_GPU);
    p_glob.setUseOpencl(true);
    p_glob.setGhosts(gh);
    p_glob.setGhostsIn(gh);
    // p_glob.setMpiComm(mpi_comm);
    p_glob.setStencilType(stencilType);
    p_glob.setResidualNorm(resnorm);
    p_glob.setIgnoreMpi(true);

    auto svptr_glob = p_glob.getStencilValues();
    auto& sv_glob = *svptr_glob;
    // mgcl_test::fill7pLaplace(sv, h, false);
    for (int i = 0; i < sv_glob.field1d().size(); i++)
        sv_glob.field1d()[i] = i;

    p_glob.init();
    return;

    // Init Problem to create all needed structures
    auto pptr = std::make_shared<mgcl::Problem>(ml, nl, ol, flptr, vlptr, m, n, o);
    auto& p = *pptr;
    p.setDeviceType(CL_DEVICE_TYPE_GPU);
    p.setUseOpencl(true);
    p.setDeviceType(CL_DEVICE_TYPE_GPU);
    p.setGhosts(gh);
    p.setGhostsIn(gh);
    p.setMpiComm(mpi_comm);
    p.setStencilType(stencilType);
    p.setResidualNorm(resnorm);

    // create slice of global stencil values and copy into local ones
    std::shared_ptr<mgcl::VaryingStencil> svptr_loc_slice = sv_glob.slice(m_start, m_end, n_start, n_end, o_start, o_end);
    auto& sv_loc_slice = *svptr_loc_slice;

    auto svptr = p.getStencilValues();
    auto& sv = *svptr;

    // check that dimensions of sliced stencilValues matches with the one from the problem
    REQUIRE(sv.getM() == sv_loc_slice.getM());
    REQUIRE(sv.getN() == sv_loc_slice.getN());
    REQUIRE(sv.getO() == sv_loc_slice.getO());
    REQUIRE(sv.getGhostsM() == sv_loc_slice.getGhostsM());
    REQUIRE(sv.getGhostsN() == sv_loc_slice.getGhostsN());
    REQUIRE(sv.getGhostsO() == sv_loc_slice.getGhostsO());

    for (int i = 0; i < sv.field1d().size(); i++)
        sv.field1d()[i] = sv_loc_slice.field1d()[i];

    p.init();

    // Check on level 0
    auto& lv = p.getLevelAt(0);
    auto mpiData = lv.getMpiDataPtr();
    auto& lv_glob = p_glob.getLevelAt(0);

    // print neighbours per rank
    // for (int i = 0; i < mpi_size; i++)
    // {
    //     MPI_Barrier(mpi_comm);
    //     if (i == mpi_rank)
    //         std::cout << mpi_rank << ": " << mpiData->left << "," << mpiData->right << ","
    //                   << mpiData->up << "," << mpiData->down << ","
    //                   << mpiData->back << "," << mpiData->front << std::endl;
    // }

    // auto f_loc_ret_ptr = lv.getDF().read(p.getCommands(), nullptr, true);
    // auto& f_loc_ret = *f_loc_ret_ptr;

    // CAPTURE(v_glob.getMgh(), v_glob.getNgh(), v_glob.getOgh());
    // CAPTURE(v_loc.getMgh(), v_loc.getNgh(), v_loc.getOgh());

    // // Make sure input is equal
    // for (int il = v_glob.getGhostsM(), ig = mpi_coords[0] * ml + v_loc.getGhostsM(); il < v_loc.getM() + v_glob.getGhostsM(); il++, ig++)
    //     for (int jl = v_glob.getGhostsN(), jg = mpi_coords[1] * nl + v_loc.getGhostsN(); jl < v_loc.getN() + v_glob.getGhostsN(); jl++, jg++)
    //         for (int kl = v_glob.getGhostsO(), kg = mpi_coords[2] * ol + v_loc.getGhostsO(); kl < v_loc.getO() + v_glob.getGhostsO(); kl++, kg++)
    //         {
    //             CAPTURE(il, jl, kl, ig, jg, kg);
    //             REQUIRE(v_loc[il][jl][kl] == v_glob[ig][jg][kg]);
    //             REQUIRE(f_loc[il][jl][kl] == f_glob[ig][jg][kg]);
    //             REQUIRE(f_loc_ret[il][jl][kl] == f_loc[il][jl][kl]);
    //             REQUIRE(f_loc_ret[il][jl][kl] == f_glob[ig][jg][kg]);
    //         }

    mgcl::MultigridEngine::residual(p_glob, lv_glob, true, 0, 0, 0);
    mgcl::MultigridEngine::residual(p, lv, true, 0, 0, 0);
    p.getOpenCLHelper().finish();
    p_glob.getOpenCLHelper().finish();

    auto v_loc_ret_ptr = lv.getDVIn().read(p.getCommands(), nullptr, true);
    auto& v_loc_ret = *v_loc_ret_ptr;
    auto v_glob_ret_ptr = lv_glob.getDVIn().read(p_glob.getCommands(), nullptr, true);
    auto& v_glob_ret = *v_glob_ret_ptr;
    auto r_loc_ret_ptr = lv.getDR().read(p.getCommands(), nullptr, true);
    auto& r_loc_ret = *r_loc_ret_ptr;
    auto r_glob_ret_ptr = lv_glob.getDR().read(p_glob.getCommands(), nullptr, true);
    auto& r_glob_ret = *r_glob_ret_ptr;
    auto f_loc_ret_ptr = lv.getDF().read(p.getCommands(), nullptr, true);
    auto& f_loc_ret = *f_loc_ret_ptr;
    auto f_glob_ret_ptr = lv_glob.getDF().read(p_glob.getCommands(), nullptr, true);
    auto& f_glob_ret = *f_glob_ret_ptr;

    // Compare local results with respective chunk of global results
    for (int il = v_glob.getGhostsM(), ig = mpi_coords[0] * ml + v_loc.getGhostsM(); il < v_loc.getM() + v_glob.getGhostsM(); il++, ig++)
        for (int jl = v_glob.getGhostsN(), jg = mpi_coords[1] * nl + v_loc.getGhostsN(); jl < v_loc.getN() + v_glob.getGhostsN(); jl++, jg++)
            for (int kl = v_glob.getGhostsO(), kg = mpi_coords[2] * ol + v_loc.getGhostsO(); kl < v_loc.getO() + v_glob.getGhostsO(); kl++, kg++)
            {
                CAPTURE(il, jl, kl, ig, jg, kg);
                REQUIRE(v_loc_ret[il][jl][kl] == v_glob_ret[ig][jg][kg]);
                REQUIRE(f_loc_ret[il][jl][kl] == f_glob_ret[ig][jg][kg]);
                REQUIRE(r_loc_ret[il][jl][kl] == r_glob_ret[ig][jg][kg]);
            }
}