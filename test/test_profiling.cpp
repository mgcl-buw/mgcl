#include "catch2/catch_test_macros.hpp"

#include "../src/mgcl/multigrid_engine.hpp"
#include "../src/mgcl/problem.hpp"
#include "test_utility.hpp"

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
    m = n = o = 4;
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

    SECTION("jacobi")
    {
        int maxiter = 3;
        std::string kernelName = "jacobi_iter_27point_varying_stencil_1d";

        mgcl::MultigridEngine::jacobi(p, lv0, maxiter, true);

        // get workgroup size from config for smallest problem size
        auto& wg = conf[kernelName][0].second;

        // calculate padded work-item count
        size_t global = static_cast<size_t>(mgh * ngh * ogh);
        if (global % wg[0] != 0)
            global += wg[0] - (global % wg[0]);

        // extract profiling data for this kernel
        auto d = p.getProfilingData()->getMeasurements();
        auto& measurements = d[kernelName];
        REQUIRE(measurements.size() == maxiter);
        for (auto el : measurements)
        {
            REQUIRE(el.elapsed > 0);
            REQUIRE(el.work_group[0] == wg[0]);
            REQUIRE(el.work_group[1] == wg[1]);
            REQUIRE(el.work_group[2] == wg[2]);
            REQUIRE(el.work_items[0] == global);
            REQUIRE(el.work_items[1] == 0);
            REQUIRE(el.work_items[2] == 0);
        }
    }

    SECTION("residual")
    {
        mgcl::MultigridEngine::residual(p, lv0, true);

        // check residual itself
        {
            std::string kernelName = "residual_27point_varying_stencil";

            // get workgroup size from config for smallest problem size
            auto& wg = conf[kernelName][0].second;

            // calculate padded work-item count
            size_t global = static_cast<size_t>(mgh * ngh * ogh);
            if (global % wg[0] != 0)
                global += wg[0] - (global % wg[0]);

            // extract profiling data for this kernel
            auto d = p.getProfilingData()->getMeasurements();
            auto& measurements = d[kernelName];
            REQUIRE(measurements.size() == 1);
            for (auto el : measurements)
            {
                REQUIRE(el.elapsed > 0);
                REQUIRE(el.work_group[0] == wg[0]);
                REQUIRE(el.work_group[1] == wg[1]);
                REQUIRE(el.work_group[2] == wg[2]);
                REQUIRE(el.work_items[0] == global);
                REQUIRE(el.work_items[1] == 0);
                REQUIRE(el.work_items[2] == 0);
            }
        }

        // check residual_squared
        {
            std::string kernelName = "residual_squared";

            // get workgroup size from config for smallest problem size
            auto& wg = conf[kernelName][0].second;

            // calculate padded work-item count
            size_t global = static_cast<size_t>(mgh * ngh * ogh);
            if (global % wg[0] != 0)
                global += wg[0] - (global % wg[0]);

            // extract profiling data for this kernel
            auto d = p.getProfilingData()->getMeasurements();
            auto& measurements = d[kernelName];
            REQUIRE(measurements.size() == 1);
            for (auto el : measurements)
            {
                REQUIRE(el.elapsed > 0);
                REQUIRE(el.work_group[0] == wg[0]);
                REQUIRE(el.work_group[1] == wg[1]);
                REQUIRE(el.work_group[2] == wg[2]);
                REQUIRE(el.work_items[0] == global);
                REQUIRE(el.work_items[1] == 0);
                REQUIRE(el.work_items[2] == 0);
            }
        }
    }
}