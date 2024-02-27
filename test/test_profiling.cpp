#include "catch2/catch_test_macros.hpp"

#include "../src/mgcl/multigrid_engine.hpp"
#include "../src/mgcl/problem.hpp"
#include "test_utility.hpp"
#include <memory>

/**
 * @brief Checks results for a single kernel.
 *
 * @param p Problem
 * @param kernelName Name of the kernel as in mgcl.cl
 * @param global work-item count that the kernel gets called with
 * @param measurementCount number of measurements
 */
void checkResult(mgcl::Problem& p, std::string kernelName, std::array<int, 3> global, int measurementCount = 1)
{
    auto& conf = p.getKernelConfig();

    // auto& lv0 = p.getLevelAt(0);

    // int mgh = lv0.getMgh();
    // int ngh = lv0.getNgh();
    // int ogh = lv0.getOgh();

    // get workgroup size from config for smallest problem size
    auto& wg = conf[kernelName][0].second;

    // calculate padded work-item count
    // size_t global[3] = {static_cast<size_t>(mgh), static_cast<size_t>(ngh), static_cast<size_t>(ogh)};
    for (int i = 0; i < 3; i++)
        if (global[i] % wg[i] != 0)
            global[i] += wg[i] - (global[i] % wg[i]);

    // extract profiling data for this kernel
    auto d = p.getProfilingData()->getMeasurements();
    auto& measurements = d[kernelName];
    REQUIRE(measurements.size() == measurementCount);
    for (auto el : measurements)
    {
        REQUIRE(el.elapsed > 0);
        REQUIRE(el.work_group[0] == wg[0]);
        REQUIRE(el.work_group[1] == wg[1]);
        REQUIRE(el.work_group[2] == wg[2]);
        REQUIRE(el.work_items[0] == global[0]);
        REQUIRE(el.work_items[1] == global[1]);
        REQUIRE(el.work_items[2] == global[2]);
    }
}

TEST_CASE("profiling_setup")
{
    int m, n, o;
    m = n = o = 4;
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
    auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
    mgcl::Problem p(m, n, o, f, v);

    p.setUseOpencl(true);
    p.setProfilingEnabled(true);

    REQUIRE(p.getProfilingData() != nullptr);

    p.setProfilingEnabled(false);
    REQUIRE(p.getProfilingData() == nullptr);
}

TEST_CASE("profiling_kernels")
{
    int m, n, o;
    m = n = o = 8;
    double h = 1.0 / static_cast<double>(m);
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
    auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
    mgcl::Problem p(m, n, o, f, v);

    p.setUseOpencl(true);
    p.setProfilingEnabled(true);
    p.setStencilType(mgcl::MGCL_VARYING);
    auto sv = p.getStencilValues();
    mgcl_test::fill7pLaplace(*sv, h, false);
    p.init();
    auto& lv0 = p.getLevelAt(0);

    int mgh = lv0.getMgh();
    int ngh = lv0.getNgh();
    int ogh = lv0.getOgh();

    auto& conf = p.getKernelConfig();

    // Clear measurements from Problem::init call
    p.getProfilingData()->getMeasurements().clear();

    SECTION("jacobi")
    {
        int maxiter = 3;
        mgcl::MultigridEngine::jacobi(p, lv0, maxiter, true);
        checkResult(p, "jacobi_iter_27point_varying_stencil_1d", {mgh * ngh * ogh, 0, 0}, maxiter);
    }

    SECTION("residual")
    {
        mgcl::MultigridEngine::residual(p, lv0, true);

        // check residual itself
        checkResult(p, "residual_27point_varying_stencil", {mgh * ngh * ogh, 0, 0});

        // check residual_squared
        checkResult(p, "residual_squared", {mgh * ngh * ogh, 0, 0});
    }

    SECTION("update_ghosts_cuboid")
    {
        auto& dbuf = lv0.getDVIn();
        mgcl::MultigridEngine::updateGhosts(p, dbuf, nullptr, true);

        checkResult(p, "update_ghosts_periodic", {mgh, ngh, ogh});
    }

    SECTION("copy_input_buffers")
    {
        // create buffers manually, so they can be copied
        std::shared_ptr<mgcl::CuboidGpu> dv = lv0.getDVIn().copyShallow();
        std::shared_ptr<mgcl::CuboidGpu> df = lv0.getDVIn().copyShallow();
        p.setDV(dv);
        p.setDF(df);

        p.getOpenCLHelper().copyInputBuffers();
        checkResult(p, "copy_input_data", {mgh, ngh, ogh});
    }

    SECTION("copy_output_buffers")
    {
        // create buffers manually, so they can be copied
        std::shared_ptr<mgcl::CuboidGpu> dv = lv0.getDVIn().copyShallow();
        p.setDV(dv);

        p.getOpenCLHelper().copyOutputBuffers();

        checkResult(p, "copy_output_data", {m, n, o});
    }

    SECTION("correct_error")
    {
        mgcl::MultigridEngine::correctError(lv0);
        checkResult(p, "correct_error", {m, n, o});
    }

    SECTION("restrict")
    {
        auto& lv1 = p.getLevelAt(1);
        mgcl::MultigridEngine::restrict(lv0, lv1, lv0.getDVIn(), lv1.getDR());
        checkResult(p, "restrict_to_coarse", {lv1.getM(), lv1.getN(), lv1.getO()});
    }

    SECTION("prolongate")
    {
        auto& lv1 = p.getLevelAt(1);
        mgcl::MultigridEngine::prolongate(lv0, lv1, lv0.getDVIn(), lv1.getDR());
        checkResult(p, "prolongate_to_fine", {lv1.getMgh(), lv1.getNgh(), lv1.getOgh()});
    }

    SECTION("stencil_update_ghosts")
    {
        // update ghosts is already called in Problem::init, so clear measurements first
        p.getProfilingData()->getMeasurements().clear();

        auto svgpu = lv0.getStencilValuesGpu();
        svgpu->updateGhosts(p.getProgram(), p.getCommands(), &conf, p.getProfilingData());
        checkResult(p, "update_ghosts_varying_stencil", {mgh, ngh, ogh});
    }

    SECTION("mult_stencils_var_var")
    {
        auto svgpu = lv0.getStencilValuesGpu();
        svgpu->multiply(*svgpu, 0, p.getProgram(), p.getCommands(), p.getContext(), nullptr,
                        p.isPeriodic(), false, &conf, p.getProfilingData());

        checkResult(p, "mult_stencils_var_var", {m, n, o});
    }

    SECTION("mult_stencils_var_fix")
    {
        auto svgpu = lv0.getStencilValuesGpu();
        mgcl::FixedStencilGpu f(3, p.getContext(), p.getCommands());
        svgpu->multiply(f, 0, p.getProgram(), p.getCommands(), p.getContext(), nullptr,
                        p.isPeriodic(), false, &conf, p.getProfilingData());

        checkResult(p, "mult_stencils_var_fix", {m, n, o * 5 * 5 * 5});
    }

    SECTION("mult_stencils_fix_var")
    {
        auto svgpu = lv0.getStencilValuesGpu();
        mgcl::FixedStencilGpu f(3, p.getContext(), p.getCommands());
        f.multiply(*svgpu, 0, p.getProgram(), p.getCommands(), p.getContext(), nullptr,
                   p.isPeriodic(), false, &conf, p.getProfilingData());

        checkResult(p, "mult_stencils_fix_var", {m, n, o * 5 * 5 * 5});
    }

    SECTION("cut_stencils_w7_to_w3")
    {
        mgcl::VaryingStencilGpu svgpu(4, 4, 4, 7, 0, p.getContext(), p.getCommands());
        svgpu.cutFromW7ToW3(p.getProgram(), p.getCommands(), p.getContext(), 0, &conf, p.getProfilingData());

        checkResult(p, "cut_stencils_w7_to_w3", {2, 2, 2});
    }
}