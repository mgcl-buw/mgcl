#include "bench_util.hpp"
#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <CL/cl.h>
#include <chrono>
#include <fstream>
#include <functional> // for function
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
using namespace std::chrono_literals;

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/problem.hpp"
#include "../test/test_utility.hpp"
#include "bench_render_templates.hpp"
#include "cli_args.hpp"

// Simplified version of old stencil multiplication (removed ghost update, kernel config and profiling features) as of 08.08.2024.
mgcl::VaryingStencilGpu galerkinStencilMult(mgcl::VaryingStencilGpu& a_h, int gh_a2h,
                                            cl_program program, cl_command_queue queue, cl_context context,
                                            int resm, int resn, int reso)
{
    // Make sure a_h has two ghosts at each border for periodic bc.
    if (a_h.getGh() < 2)
        throw "galerkin: a_h needs to have 2 ghosts at each border for periodic bc!";

    if (gh_a2h < 2)
        throw "galerkin: gh_a2h must be at least 2.";

    // Get the full-weight restriction stencil S as 3x3x3 stencil with two additional ghosts at each border.
    // The ghosts are needed in order to respect periodic boundary conditions. One ghost per stencil multiplication.
    auto sr = mgcl::create3dFullWeightRestrictionStencilGpu(context, queue, program);
    auto sp = mgcl::create3dBilinearProlongationStencilGpu(context, queue, program);

    // A_2h = R * A_h * P = K * S * A_h * S * K^T, where K is the cutting matrix. We first calculate
    // S * A_h * S and cut out later manually.
    auto sas = sr.multiply(a_h, 2, nullptr, nullptr, nullptr, program, queue, context, nullptr, true, nullptr, nullptr)
                   .multiply(sp, 0, nullptr, nullptr, nullptr, program, queue, context, nullptr, true, nullptr, nullptr);

    // Cut stencil from 7x7x7 down to 3x3x3, i.e. copy only selected values to new stencil, skipping ghosts.
    auto a_2h = sas.cutFromW7ToW3(program, queue, context, gh_a2h, nullptr, nullptr, resm, resn, reso);

    return a_2h;
}

enum class KernelVersion
{
    DEFAULT,
    POINTER,
    PRIVATE_R_P,
    PARALLEL_IIJJKK,
    CACHED_RA,
    CACHED_RA_LOCALMEM
};

// Simplified version of optimized galerkin (without kernel config and profiling) as of 08.08.2024.
std::unique_ptr<mgcl::VaryingStencilGpu> galerkinOptimized(mgcl::VaryingStencilGpu& a_h, int gh_a2h,
                                                           int resm, int resn, int reso,
                                                           cl_program program, cl_command_queue queue, cl_context context,
                                                           KernelVersion kernelVersion)
{
    // Make sure a_h has two ghosts at each border for periodic bc.
    if (a_h.getGh() < 1 || a_h.getGh() < 1 || a_h.getGh() < 1)
        error("galerkin: a_h needs to have at least 1 ghosts at each border for periodic bc!");

    if (gh_a2h < 1)
        error("galerkin: gh_a2h must be at least 1.");

    // TODO sanity checks on resm, resn, reso?

    // Get the full-weight restriction stencil S as 3x3x3 stencil with two additional ghosts at each border.
    // The ghosts are needed in order to respect periodic boundary conditions. One ghost per stencil multiplication.
    auto r = mgcl::create3dFullWeightRestrictionStencilGpu(context, queue, program);
    auto p = mgcl::create3dBilinearProlongationStencilGpu(context, queue, program);

    auto a_2h = std::make_unique<mgcl::VaryingStencilGpu>(resm, resn, reso, 3, gh_a2h, context, queue, program);

    int err;

    // Create the compute kernel from the program
    std::string kernelName;
    switch (kernelVersion)
    {
    case KernelVersion::DEFAULT:
        kernelName = "galerkin";
        break;
    case KernelVersion::POINTER:
        kernelName = "galerkin_ptr";
        break;
    case KernelVersion::PRIVATE_R_P:
        kernelName = "galerkin_private_r_p";
        break;
    case KernelVersion::PARALLEL_IIJJKK:
        kernelName = "galerkin_parallel_iijjkk";
        break;
    case KernelVersion::CACHED_RA:
        kernelName = "galerkin_cached_RA";
        break;
    case KernelVersion::CACHED_RA_LOCALMEM:
        kernelName = "galerkin_cached_RA_localmem";
        break;
    }
    cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &err);
    mgcl::mgclCheckError(err, "Creating kernel");

    cl_mem a_h_raw = a_h.getBuf();
    cl_mem a_2h_raw = a_2h->getBuf();
    cl_mem r_raw = r.getBuf();
    cl_mem p_raw = p.getBuf();

    int mgh_f = a_h.getMgh();
    int ngh_f = a_h.getNgh();
    int ogh_f = a_h.getOgh();
    int m_c_loc = a_h.getM() >> 1;
    int n_c_loc = a_h.getN() >> 1;
    int o_c_loc = a_h.getO() >> 1;
    int gh_f = a_h.getGh();
    int gh_c = a_2h->getGh();

    // one work-item per local real coarse grid point.
    size_t global = (a_h.getM() >> 1) * (a_h.getN() >> 1) * (a_h.getO() >> 1);
    if (kernelVersion == KernelVersion::PARALLEL_IIJJKK)
        global *= 27;
    size_t local = 128;

    if (kernelVersion == KernelVersion::CACHED_RA_LOCALMEM)
    {
        local = 32;
    }

    // // Apply kernel config, if available
    // if (kernelConfig)
    // {
    //     const auto& c = conf::getWorkGroupSizeForKernelAndWiCount(*kernelConfig, kernelName, global);
    //     local = static_cast<size_t>(global > c[0] ? c[0] : global);
    // }

    // pad global size to fit multiple of local size
    if (global % local != 0)
        global += local - (global % local);

    // assign kernel arguments
    int pos = 0;
    err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &a_h_raw);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &a_2h_raw);
    mgcl::mgclCheckError(err, "Setting kernel arguments");
    if (kernelVersion != KernelVersion::PRIVATE_R_P)
    {
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &r_raw);
        err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &p_raw);
    }
    if (kernelVersion == KernelVersion::CACHED_RA_LOCALMEM)
    {
        size_t localMemSize = local * 125 * sizeof(double); // one wi needs 125 doubles of local memory for caching RA
        // size_t localMemSize = m_c_loc * n_c_loc * o_c_loc * 125 * sizeof(double); // one wi needs 125 doubles of local memory for caching RA
        err |= clSetKernelArg(kernel, ++pos, localMemSize, NULL);
    }
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh_f);
    mgcl::mgclCheckError(err, "Setting kernel arguments");
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh_f);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh_f);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m_c_loc);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n_c_loc);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o_c_loc);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &resm);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &resn);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &reso);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &gh_f);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &gh_c);
    mgcl::mgclCheckError(err, "Setting kernel arguments");

    cl_event ev;

    // enqueue kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, &ev);
    mgcl::mgclCheckError(err, "Enqueueing galerkin kernel");

    // if (pd != nullptr)
    // {
    //     pd->addMeasurement(queue, ev, kernelName,
    //                        {global, 0, 0},
    //                        {local, 1, 1});
    // }
    mgcl::mgclCheckError(clReleaseEvent(ev), "clReleaseEvent");

    err = clReleaseKernel(kernel);
    mgcl::mgclCheckError(err, "Releasing galerkin kernel");

    return a_2h;
}

std::unique_ptr<mgcl::VaryingStencilGpu> galerkinHandcrafted(mgcl::VaryingStencilGpu& a_h, int gh_a2h,
                                                             int resm, int resn, int reso,
                                                             cl_program program, cl_command_queue queue, cl_context context)
{
    // Make sure a_h has two ghosts at each border for periodic bc.
    if (a_h.getGh() < 1 || a_h.getGh() < 1 || a_h.getGh() < 1)
        error("galerkin: a_h needs to have at least 1 ghosts at each border for periodic bc!");

    if (gh_a2h < 1)
        error("galerkin: gh_a2h must be at least 1.");

    // TODO sanity checks on resm, resn, reso?

    // Get the full-weight restriction stencil S as 3x3x3 stencil with two additional ghosts at each border.
    // The ghosts are needed in order to respect periodic boundary conditions. One ghost per stencil multiplication.
    auto r = mgcl::create3dFullWeightRestrictionStencilGpu(context, queue, program);
    auto p = mgcl::create3dBilinearProlongationStencilGpu(context, queue, program);

    auto a_2h = std::make_unique<mgcl::VaryingStencilGpu>(resm, resn, reso, 3, gh_a2h, context, queue, program);

    int err;

    // Create the compute kernel from the program
    const char* kernelName = "galerkin_handcrafted";
    cl_kernel kernel = clCreateKernel(program, kernelName, &err);
    mgcl::mgclCheckError(err, "Creating kernel");

    cl_mem a_h_raw = a_h.getBuf();
    cl_mem a_2h_raw = a_2h->getBuf();
    cl_mem r_raw = r.getBuf();
    cl_mem p_raw = p.getBuf();

    int mgh_f = a_h.getMgh();
    int ngh_f = a_h.getNgh();
    int ogh_f = a_h.getOgh();
    int m_c_loc = a_h.getM() >> 1;
    int n_c_loc = a_h.getN() >> 1;
    int o_c_loc = a_h.getO() >> 1;
    int gh_f = a_h.getGh();
    int gh_c = a_2h->getGh();

    // assign kernel arguments
    int pos = 0;
    err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &a_h_raw);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &a_2h_raw);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &r_raw);
    err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &p_raw);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh_f);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh_f);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh_f);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &m_c_loc);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &n_c_loc);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &o_c_loc);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &resm);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &resn);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &reso);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &gh_f);
    err |= clSetKernelArg(kernel, ++pos, sizeof(int), &gh_c);
    mgcl::mgclCheckError(err, "Setting kernel arguments");

    // one work-item per local real coarse grid point.
    size_t global = (a_h.getM() >> 1) * (a_h.getN() >> 1) * (a_h.getO() >> 1);
    size_t local = 128;

    // pad global size to fit multiple of local size
    if (global % local != 0)
        global += local - (global % local);

    cl_event ev;

    // enqueue kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, &ev);
    mgcl::mgclCheckError(err, "Enqueueing update ghosts of varying stencil kernel");

    err = clReleaseKernel(kernel);
    mgcl::mgclCheckError(err, "Releasing update ghosts of varying stencil kernel");

    return a_2h;
}

// Benchs the old galerkin version (using stencil multiplication) vs. the new optimized one, that writes
// directly to the result stencil
// Creation date: 08.08.2024
TEST_CASE("benchGalerkinOldVsOptimized")
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
    auto v = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    auto f = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    mgcl::Problem p(1, 1, 1, f, v);
    p.setKernelFile("kernel_optimizations.cl");
    p.setUseOpencl(true);
    p.setDeviceType(CL_DEVICE_TYPE_GPU);
    p.init();

    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];

        mgcl::VaryingStencilGpu a_h(m, n, o, 3, 2, p.getContext(), p.getCommands(), p.getProgram());

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ms, "ms")
            .epochs(CLI_ARGS::bench_epochs)
            .epochIterations(CLI_ARGS::bench_iterations)
            .relative(false);

        {
            std::string name = std::string("galerkin_stencil_arithmetic_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o));

            bench.run(std::string(name).c_str(), [&] { //
                galerkinStencilMult(a_h, 2, p.getProgram(), p.getCommands(), p.getContext(), m >> 1, n >> 1, o >> 1);
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
        }

        {
            std::string name = std::string("galerkin_optimized_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o));

            bench.run(std::string(name).c_str(), [&] { //
                galerkinOptimized(a_h, 2, m >> 1, n >> 1, o >> 1, p.getProgram(), p.getCommands(), p.getContext(), KernelVersion::DEFAULT);
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
        }
    }

    bench_util::printCsvFormat(results);
}

// Checks results for correctness
TEST_CASE("benchGalerkinOptimizedKernelVersions_checkResults")
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

    // Create dummy problem to initialize OpenCL
    auto v = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    auto f = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    mgcl::Problem p(1, 1, 1, f, v);
    p.setKernelFile("kernel_optimizations.cl");
    p.getOpenCLHelper().setReadKernelFromFile(true);
    p.setUseOpencl(true);
    p.setDeviceType(CL_DEVICE_TYPE_GPU);
    p.init();

    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];

        mgcl::VaryingStencilGpu a_h(m, n, o, 3, 2, p.getContext(), p.getCommands(), p.getProgram());
        mgcl::VaryingStencil tmp(m, n, o, 3, 2, 2, 2);
        tmp.fillRandomInt(-10, 10, false);
        a_h.fill(tmp, p.getCommands(), true);

        std::unique_ptr<mgcl::VaryingStencilGpu> a_2h_check = galerkinOptimized(
            a_h, 2, m >> 1, n >> 1, o >> 1,
            p.getProgram(), p.getCommands(), p.getContext(),
            KernelVersion::DEFAULT);
        auto a_2h_check_h = a_2h_check->read(p.getCommands(), true);

        auto a_2h_private_rp = galerkinOptimized(
            a_h, 2, m >> 1, n >> 1, o >> 1,
            p.getProgram(), p.getCommands(), p.getContext(),
            KernelVersion::PRIVATE_R_P);
        auto a_2h_private_rp_h = a_2h_private_rp->read(p.getCommands(), true);
        REQUIRE(a_2h_check_h.isEqual(a_2h_private_rp_h));
        a_2h_private_rp.reset();

        auto a_2h_parallel_iijjkk = galerkinOptimized(
            a_h, 2, m >> 1, n >> 1, o >> 1, p.getProgram(), p.getCommands(), p.getContext(),
            KernelVersion::PARALLEL_IIJJKK);
        auto a_2h_parallel_iijjkk_h = a_2h_parallel_iijjkk->read(p.getCommands(), true);
        REQUIRE(a_2h_check_h.isEqual(a_2h_parallel_iijjkk_h));
        a_2h_parallel_iijjkk.reset();

        auto a_2h_pointer = galerkinOptimized(
            a_h, 2, m >> 1, n >> 1, o >> 1, p.getProgram(), p.getCommands(), p.getContext(),
            KernelVersion::POINTER);
        auto a_2h_pointer_h = a_2h_pointer->read(p.getCommands(), true);
        REQUIRE(a_2h_check_h.isEqual(a_2h_pointer_h));
        a_2h_pointer.reset();

        auto a_2h_cached_ra = galerkinOptimized(
            a_h, 2, m >> 1, n >> 1, o >> 1, p.getProgram(), p.getCommands(), p.getContext(),
            KernelVersion::CACHED_RA);
        auto a_2h_cached_ra_h = a_2h_cached_ra->read(p.getCommands(), true);
        REQUIRE(a_2h_check_h.isEqual(a_2h_cached_ra_h));

        auto a_2h_cached_ra_localmem = galerkinOptimized(
            a_h, 2, m >> 1, n >> 1, o >> 1, p.getProgram(), p.getCommands(), p.getContext(),
            KernelVersion::CACHED_RA_LOCALMEM);
        auto a_2h_cached_ra_localmem_h = a_2h_cached_ra_localmem->read(p.getCommands(), true);
        REQUIRE(a_2h_check_h.isEqual(a_2h_cached_ra_localmem_h));
    }
}

// Benchs the optimized version of Galerkin using structs vs. pointer to structs, in order to answer the question
// "Is it faster to use pointers to structs in the utility functions because the structs won't be copied?".
// Creation date: 08.08.2024
// 18.09.2024: Refactored to test various kernel versions of optimized galerkin
TEST_CASE("benchGalerkinOptimizedKernelVersions")
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
    auto v = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    auto f = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    mgcl::Problem p(1, 1, 1, f, v);
    p.setKernelFile("kernel_optimizations.cl");
    p.getOpenCLHelper().setReadKernelFromFile(true);
    p.setUseOpencl(true);
    p.setDeviceType(CL_DEVICE_TYPE_GPU);
    p.init();

    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];

        mgcl::VaryingStencilGpu a_h(m, n, o, 3, 2, p.getContext(), p.getCommands(), p.getProgram());

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ms, "ms")
            .epochs(CLI_ARGS::bench_epochs)
            .epochIterations(CLI_ARGS::bench_iterations)
            .relative(false);

        {
            std::string name = std::string("galerkin_optimized_noptr_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o));

            bench.run(std::string(name).c_str(), [&] { //
                galerkinOptimized(a_h, 2, m >> 1, n >> 1, o >> 1, p.getProgram(), p.getCommands(), p.getContext(), KernelVersion::DEFAULT);
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
        }

        {
            std::string name = std::string("galerkin_optimized_cached_RA_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o));

            bench.run(std::string(name).c_str(), [&] { //
                galerkinOptimized(a_h, 2, m >> 1, n >> 1, o >> 1, p.getProgram(), p.getCommands(), p.getContext(), KernelVersion::CACHED_RA);
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
        }
        {
            std::string name = std::string("galerkin_optimized_cached_RA_localmem_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o));

            bench.run(std::string(name).c_str(), [&] { //
                galerkinOptimized(a_h, 2, m >> 1, n >> 1, o >> 1, p.getProgram(), p.getCommands(), p.getContext(), KernelVersion::CACHED_RA_LOCALMEM);
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
        }

        {
            std::string name = std::string("galerkin_optimized_private_r_p_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o));

            bench.run(std::string(name).c_str(), [&] { //
                galerkinOptimized(a_h, 2, m >> 1, n >> 1, o >> 1, p.getProgram(), p.getCommands(), p.getContext(), KernelVersion::PRIVATE_R_P);
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
        }
        {
            std::string name = std::string("galerkin_parallel_iijjkk")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o));

            bench.run(std::string(name).c_str(), [&] { //
                galerkinOptimized(a_h, 2, m >> 1, n >> 1, o >> 1, p.getProgram(), p.getCommands(), p.getContext(), KernelVersion::PARALLEL_IIJJKK);
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
        }
        {
            std::string name = std::string("galerkin_optimized_ptr_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o));

            bench.run(std::string(name).c_str(), [&] { //
                galerkinOptimized(a_h, 2, m >> 1, n >> 1, o >> 1, p.getProgram(), p.getCommands(), p.getContext(), KernelVersion::POINTER);
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
        }

        {
            std::string name = std::string("galerkin_handcrafted_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o));

            bench.run(std::string(name).c_str(), [&] { //
                galerkinHandcrafted(a_h, 2, m >> 1, n >> 1, o >> 1, p.getProgram(), p.getCommands(), p.getContext());
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
        }
    }

    bench_util::printCsvFormat(results);
}

// Benchs the optimized version of Galerkin vs. handcrafted one.
// Creation date: 06.11.2024
TEST_CASE("benchGalerkinOptimizedVsHandcrafted")
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
    auto v = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    auto f = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    mgcl::Problem p(1, 1, 1, f, v);
    p.setKernelFile("kernel_optimizations.cl");
    p.getOpenCLHelper().setReadKernelFromFile(true);
    p.setUseOpencl(true);
    p.setDeviceType(CL_DEVICE_TYPE_GPU);
    p.init();

    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];

        mgcl::VaryingStencilGpu a_h(m, n, o, 3, 2, p.getContext(), p.getCommands(), p.getProgram());

        ankerl::nanobench::Bench bench;
        bench.timeUnit(1ms, "ms")
            .epochs(CLI_ARGS::bench_epochs)
            .epochIterations(CLI_ARGS::bench_iterations)
            .relative(false);

        {
            std::string name = std::string("galerkin_optimized_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o));

            bench.run(std::string(name).c_str(), [&] { //
                galerkinOptimized(a_h, 2, m >> 1, n >> 1, o >> 1, p.getProgram(), p.getCommands(), p.getContext(), KernelVersion::DEFAULT);
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
        }
        {
            std::string name = std::string("galerkin_handcrafted_")
                                   .append(std::to_string(m))
                                   .append("_")
                                   .append(std::to_string(n))
                                   .append("_")
                                   .append(std::to_string(o));

            bench.run(std::string(name).c_str(), [&] { //
                galerkinHandcrafted(a_h, 2, m >> 1, n >> 1, o >> 1, p.getProgram(), p.getCommands(), p.getContext());
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
        }
    }

    bench_util::printCsvFormat(results);
}