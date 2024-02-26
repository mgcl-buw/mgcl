#include <catch2/catch_test_macros.hpp>
#include <memory>

#include "../src/mgcl/kernel_config.hpp"
#include "../src/mgcl/problem.hpp"

/**
 * @brief Tests if a default kernel config is created upon the creation of a Problem object.
 *
 */
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

/**
 * @brief Check if the kernel config can be edited.
 *
 */
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

/**
 * @brief Check if the work-group sizes for a given work-item count can be retrieved for a given kernel.
 *
 */
TEST_CASE("getWorkGroupSizeForKernelAndWiCount")
{
    int m, n, o;
    m = n = o = 4;

    auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
    mgcl::Problem p(m, n, o, f, v);

    auto& c = p.getKernelConfig();
    c["jacobi_iter_27point_varying_stencil_1d"].clear();
    c["jacobi_iter_27point_varying_stencil_1d"].push_back({10, {10, 1, 1}});
    c["jacobi_iter_27point_varying_stencil_1d"].push_back({20, {20, 1, 1}});

    // Check if correct wg sizes are returned when work-item count matches a boundary exactly.
    auto& wg10 = mgcl::conf::getWorkGroupSizeForKernelAndWiCount(c, "jacobi_iter_27point_varying_stencil_1d", 10);
    REQUIRE(wg10[0] == 10);
    REQUIRE(wg10[1] == 1);
    REQUIRE(wg10[2] == 1);

    // Check if correct wg sizes are returned when work-item count is between two boundaries.
    auto& wg15 = mgcl::conf::getWorkGroupSizeForKernelAndWiCount(c, "jacobi_iter_27point_varying_stencil_1d", 15);
    REQUIRE(wg15[0] == 10);
    REQUIRE(wg15[1] == 1);
    REQUIRE(wg15[2] == 1);

    // Check if correct wg sizes are returned when work-item count is bigger than the biggest boundary.
    auto& wg25 = mgcl::conf::getWorkGroupSizeForKernelAndWiCount(c, "jacobi_iter_27point_varying_stencil_1d", 25);
    REQUIRE(wg25[0] == 20);
    REQUIRE(wg25[1] == 1);
    REQUIRE(wg25[2] == 1);

    // Check if the wg sizes for the smallest boundary is returned, when work-item count is smaller than the smallest boundary.
    auto& wg5 = mgcl::conf::getWorkGroupSizeForKernelAndWiCount(c, "jacobi_iter_27point_varying_stencil_1d", 5);
    REQUIRE(wg5[0] == 10);
    REQUIRE(wg5[1] == 1);
    REQUIRE(wg5[2] == 1);

    // Check if an exception is thrown if the kernel name does not exist.
    REQUIRE_THROWS(mgcl::conf::getWorkGroupSizeForKernelAndWiCount(c, "does_not_exist", 5));
}