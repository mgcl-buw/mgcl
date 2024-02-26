#include "catch2/catch_test_macros.hpp"

#include "../src/mgcl/problem.hpp"

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
    auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
    auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
    mgcl::Problem p(m, n, o, f, v);

    p.setUseOpencl(true);
    p.setProfilingEnabled(true);

    // TODO
    FAIL();
}