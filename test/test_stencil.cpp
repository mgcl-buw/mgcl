#include <catch2/catch_test_macros.hpp>

#include <memory>

#include "../cuboid.hpp"
#include "../stencil.hpp"

TEST_CASE("StencilLaplace7p")
{
    auto c = std::make_shared<mgcl::Cuboid>(4, 4, 4);
    (*c)[1][1][1] = 8;
    (*c)[1][1][0] = 4;
    (*c)[1][1][2] = 16;
    (*c)[1][0][1] = 4;
    (*c)[1][2][1] = 2;
    (*c)[0][1][1] = 1;
    (*c)[2][1][1] = 2;

    mgcl::StencilLaplace7p stencil(c);

    double expectedFactor = 4.0 * 4.0;
    REQUIRE(expectedFactor == stencil.getStencilFactor());

    double expected = expectedFactor * (6.0 * 8 - 4 - 16 - 4 - 2 - 1 - 2);
    double sum = stencil.apply(1, 1, 1);
    REQUIRE(sum == expected);
}

TEST_CASE("StencilLaplace19p")
{
    auto c = std::make_shared<mgcl::Cuboid>(4, 4, 4);
    (*c)[1][1][1] = 8;
    (*c)[1][1][0] = 4;
    (*c)[1][1][2] = 16;
    (*c)[1][0][1] = 4;
    (*c)[1][2][1] = 2;
    (*c)[0][1][1] = 1;
    (*c)[2][1][1] = 2;

    (*c)[1][0][0] = 8;
    (*c)[1][0][2] = 4;
    (*c)[1][2][0] = 1;
    (*c)[1][2][2] = 2;

    (*c)[0][1][0] = 16;
    (*c)[0][1][2] = 32;
    (*c)[2][1][0] = 8;
    (*c)[2][1][2] = 4;

    (*c)[0][0][1] = 4;
    (*c)[0][2][1] = 4;
    (*c)[2][0][1] = 2;
    (*c)[2][2][1] = 1;

    mgcl::StencilLaplace19p stencil(c);

    double expectedFactor = (4.0 * 4.0) / 6.0;
    REQUIRE(expectedFactor == stencil.getStencilFactor());

    double expected = expectedFactor * (24.0 * 8 - 2.0 * (4 + 16 + 4 + 2 + 1 + 2) - 8 - 4 - 1 - 2 - 16 - 32 - 8 - 4 - 4 - 4 - 2 - 1);
    double sum = stencil.apply(1, 1, 1);
    REQUIRE(sum == expected);
}

TEST_CASE("StencilLaplace27p")
{
    auto c = std::make_shared<mgcl::Cuboid>(4, 4, 4);
    (*c)[1][1][1] = 8;
    (*c)[1][1][0] = 4;
    (*c)[1][1][2] = 16;
    (*c)[1][0][1] = 4;
    (*c)[1][2][1] = 2;
    (*c)[0][1][1] = 1;
    (*c)[2][1][1] = 2;

    (*c)[1][0][0] = 8;
    (*c)[1][0][2] = 4;
    (*c)[1][2][0] = 1;
    (*c)[1][2][2] = 2;

    (*c)[0][1][0] = 16;
    (*c)[0][1][2] = 32;
    (*c)[2][1][0] = 8;
    (*c)[2][1][2] = 4;

    (*c)[0][0][1] = 4;
    (*c)[0][2][1] = 4;
    (*c)[2][0][1] = 2;
    (*c)[2][2][1] = 1;

    (*c)[0][0][0] = 8;
    (*c)[0][0][2] = 4;
    (*c)[0][2][0] = 1;
    (*c)[0][2][2] = 2;
    (*c)[2][0][0] = 16;
    (*c)[2][0][2] = 4;
    (*c)[2][2][0] = 8;
    (*c)[2][2][2] = 2;

    mgcl::StencilLaplace27p stencil(c);

    double expectedFactor = (4.0 * 4.0) / 30.0;
    REQUIRE(expectedFactor == stencil.getStencilFactor());

    double expected = expectedFactor * (128.0 * 8 - 14.0 * (4 + 16 + 4 + 2 + 1 + 2) -
                                        3.0 * (8 + 4 + 1 + 2 + 16 + 32 + 8 + 4 + 4 + 4 + 2 + 1) -
                                        8 - 4 - 1 - 2 - 16 - 4 - 8 - 2);
    double sum = stencil.apply(1, 1, 1);
    REQUIRE(sum == expected);
}
