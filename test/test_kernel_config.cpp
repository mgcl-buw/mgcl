#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "../src/mgcl/problem.hpp"

TEST_CASE("default_kernel_config_is_created")
{
    int m, n, o;
    m = n = o = 4;

    auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
    mgcl::Problem p(m, n, o, f, v);

    // Test one kernel exemplarily
    REQUIRE(!p.getKernelConfig().at("jacobi_iter_27point_varying_stencil_1d").empty());
}

TEST_CASE("set_custom_kernel_config")
{
    int m, n, o;
    m = n = o = 4;

    auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
    mgcl::Problem p(m, n, o, f, v);

    auto& c = p.getKernelConfig();

    c["jacobi_iter_27point_varying_stencil_1d"].push_back({10, {10, 1, 1}});
    c["jacobi_iter_27point_varying_stencil_1d"].push_back({20, {20, 1, 1}});

    // Test one kernel exemplarily
    REQUIRE(!p.getKernelConfig().at("jacobi_iter_27point_varying_stencil_1d").empty());
    REQUIRE(p.getKernelConfig().at("jacobi_iter_27point_varying_stencil_1d").size() == 3);
    REQUIRE(p.getKernelConfig().at("jacobi_iter_27point_varying_stencil_1d")[1].first == 10);
    REQUIRE(p.getKernelConfig().at("jacobi_iter_27point_varying_stencil_1d")[1].second[0] == 10);
    REQUIRE(p.getKernelConfig().at("jacobi_iter_27point_varying_stencil_1d")[2].first == 20);
    REQUIRE(p.getKernelConfig().at("jacobi_iter_27point_varying_stencil_1d")[2].second[0] == 20);
}