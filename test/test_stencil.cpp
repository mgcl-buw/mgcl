#include <catch2/catch_approx.hpp>
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

    double h = 1.0 / 4.0;
    mgcl::StencilLaplace7p stencil(h);

    double expectedFactor = 4.0 * 4.0;
    REQUIRE(expectedFactor == stencil.getStencilFactor());

    double expected = expectedFactor * (6.0 * 8 - 4 - 16 - 4 - 2 - 1 - 2);
    double sum = stencil.apply(*c, 1, 1, 1);
    REQUIRE(sum == Catch::Approx(expected));
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

    double h = 1.0 / 4.0;
    mgcl::StencilLaplace19p stencil(h);

    double expectedFactor = (4.0 * 4.0) / 6.0;
    REQUIRE(expectedFactor == stencil.getStencilFactor());

    double expected = expectedFactor * (24.0 * 8 - 2.0 * (4 + 16 + 4 + 2 + 1 + 2) - 8 - 4 - 1 - 2 - 16 - 32 - 8 - 4 - 4 - 4 - 2 - 1);
    double sum = stencil.apply(*c, 1, 1, 1);
    REQUIRE(sum == Catch::Approx(expected));
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

    double h = 1.0 / 4.0;
    mgcl::StencilLaplace27p stencil(h);

    double expectedFactor = (4.0 * 4.0) / 30.0;
    REQUIRE(expectedFactor == stencil.getStencilFactor());

    double expected = expectedFactor * (128.0 * 8 - 14.0 * (4 + 16 + 4 + 2 + 1 + 2) -
                                        3.0 * (8 + 4 + 1 + 2 + 16 + 32 + 8 + 4 + 4 + 4 + 2 + 1) -
                                        8 - 4 - 1 - 2 - 16 - 4 - 8 - 2);
    double sum = stencil.apply(*c, 1, 1, 1);
    REQUIRE(sum == Catch::Approx(expected));
}

TEST_CASE("StencilVarying7p")
{
    auto c = std::make_shared<mgcl::Cuboid>(4, 4, 4);
    (*c)[1][1][1] = 8;
    (*c)[1][1][0] = 4;
    (*c)[1][1][2] = 16;
    (*c)[1][0][1] = 4;
    (*c)[1][2][1] = 2;
    (*c)[0][1][1] = 1;
    (*c)[2][1][1] = 2;

    mgcl::StencilVarying7p stencil(c->getM(), c->getN(), c->getO());
    auto vals = stencil.getStencilValues();
    REQUIRE(vals->getDim1() == c->getM());
    REQUIRE(vals->getDim2() == c->getM());
    REQUIRE(vals->getDim3() == c->getM());
    REQUIRE(vals->getDim4() == 7);

    // fill varying stencil values with 7p Laplace stencil
    double h2inv = 4.0 * 4.0;
    for (int i = 0; i < vals->getDim1gh(); i++)
        for (int j = 0; j < vals->getDim2gh(); j++)
            for (int k = 0; k < vals->getDim3gh(); k++)
            {
                (*vals)[i][j][k][0] = 6.0 * h2inv;
                (*vals)[i][j][k][1] = -1.0 * h2inv;
                (*vals)[i][j][k][2] = -1.0 * h2inv;
                (*vals)[i][j][k][3] = -1.0 * h2inv;
                (*vals)[i][j][k][4] = -1.0 * h2inv;
                (*vals)[i][j][k][5] = -1.0 * h2inv;
                (*vals)[i][j][k][6] = -1.0 * h2inv;
            }

    double expected = h2inv * (6.0 * 8 - 4 - 16 - 4 - 2 - 1 - 2);
    double sum = stencil.apply(*c, 1, 1, 1);
    REQUIRE(sum == Catch::Approx(expected));
}

TEST_CASE("StencilVarying19p")
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

    mgcl::StencilVarying19p stencil(c->getM(), c->getN(), c->getO());
    auto vals = stencil.getStencilValues();
    REQUIRE(vals->getDim1() == c->getM());
    REQUIRE(vals->getDim2() == c->getM());
    REQUIRE(vals->getDim3() == c->getM());
    REQUIRE(vals->getDim4() == 19);

    // fill varying stencil with 19p Laplace stencil
    double h2inv = (4.0 * 4.0) / 6.0;
    for (int i = 0; i < vals->getDim1gh(); i++)
        for (int j = 0; j < vals->getDim2gh(); j++)
            for (int k = 0; k < vals->getDim3gh(); k++)
            {
                (*vals)[i][j][k][0] = 24.0 * h2inv;
                (*vals)[i][j][k][1] = -2.0 * h2inv;
                (*vals)[i][j][k][2] = -2.0 * h2inv;
                (*vals)[i][j][k][3] = -2.0 * h2inv;
                (*vals)[i][j][k][4] = -2.0 * h2inv;
                (*vals)[i][j][k][5] = -2.0 * h2inv;
                (*vals)[i][j][k][6] = -2.0 * h2inv;
                (*vals)[i][j][k][7] = -h2inv;
                (*vals)[i][j][k][8] = -h2inv;
                (*vals)[i][j][k][9] = -h2inv;
                (*vals)[i][j][k][10] = -h2inv;
                (*vals)[i][j][k][11] = -h2inv;
                (*vals)[i][j][k][12] = -h2inv;
                (*vals)[i][j][k][13] = -h2inv;
                (*vals)[i][j][k][14] = -h2inv;
                (*vals)[i][j][k][15] = -h2inv;
                (*vals)[i][j][k][16] = -h2inv;
                (*vals)[i][j][k][17] = -h2inv;
                (*vals)[i][j][k][18] = -h2inv;
            }

    double expected = h2inv * (24.0 * 8 - 2.0 * (4 + 16 + 4 + 2 + 1 + 2) - 8 - 4 - 1 - 2 - 16 - 32 - 8 - 4 - 4 - 4 - 2 - 1);
    double sum = stencil.apply(*c, 1, 1, 1);
    REQUIRE(sum == Catch::Approx(expected));
}

TEST_CASE("StencilVarying27p")
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

    mgcl::StencilVarying27p stencil(c->getM(), c->getN(), c->getO());
    auto vals = stencil.getStencilValues();
    REQUIRE(vals->getDim1() == c->getM());
    REQUIRE(vals->getDim2() == c->getM());
    REQUIRE(vals->getDim3() == c->getM());
    REQUIRE(vals->getDim4() == 27);

    // fill varying stencil with 19p Laplace stencil
    double h2inv = (4.0 * 4.0) / 30.0;
    for (int i = 0; i < vals->getDim1gh(); i++)
        for (int j = 0; j < vals->getDim2gh(); j++)
            for (int k = 0; k < vals->getDim3gh(); k++)
            {
                (*vals)[i][j][k][0] = 128.0 * h2inv;
                (*vals)[i][j][k][1] = -14.0 * h2inv;
                (*vals)[i][j][k][2] = -14.0 * h2inv;
                (*vals)[i][j][k][3] = -14.0 * h2inv;
                (*vals)[i][j][k][4] = -14.0 * h2inv;
                (*vals)[i][j][k][5] = -14.0 * h2inv;
                (*vals)[i][j][k][6] = -14.0 * h2inv;
                (*vals)[i][j][k][7] = -3.0 * h2inv;
                (*vals)[i][j][k][8] = -3.0 * h2inv;
                (*vals)[i][j][k][9] = -3.0 * h2inv;
                (*vals)[i][j][k][10] = -3.0 * h2inv;
                (*vals)[i][j][k][11] = -3.0 * h2inv;
                (*vals)[i][j][k][12] = -3.0 * h2inv;
                (*vals)[i][j][k][13] = -3.0 * h2inv;
                (*vals)[i][j][k][14] = -3.0 * h2inv;
                (*vals)[i][j][k][15] = -3.0 * h2inv;
                (*vals)[i][j][k][16] = -3.0 * h2inv;
                (*vals)[i][j][k][17] = -3.0 * h2inv;
                (*vals)[i][j][k][18] = -3.0 * h2inv;
                (*vals)[i][j][k][19] = -h2inv;
                (*vals)[i][j][k][20] = -h2inv;
                (*vals)[i][j][k][21] = -h2inv;
                (*vals)[i][j][k][22] = -h2inv;
                (*vals)[i][j][k][23] = -h2inv;
                (*vals)[i][j][k][24] = -h2inv;
                (*vals)[i][j][k][25] = -h2inv;
                (*vals)[i][j][k][26] = -h2inv;
            }

    double expected = h2inv * (128.0 * 8 - 14.0 * (4 + 16 + 4 + 2 + 1 + 2) -
                               3.0 * (8 + 4 + 1 + 2 + 16 + 32 + 8 + 4 + 4 + 4 + 2 + 1) -
                               8 - 4 - 1 - 2 - 16 - 4 - 8 - 2);
    double sum = stencil.apply(*c, 1, 1, 1);
    REQUIRE(sum == Catch::Approx(expected));
}

TEST_CASE("Stencil::clone")
{
    auto c = std::make_shared<mgcl::Cuboid>(8, 8, 8);
    (*c)[1][1][1] = 8;

    auto c2 = std::make_shared<mgcl::Cuboid>(4, 4, 4);
    (*c)[1][1][1] = 32;

    double h1 = 1.0 / c->getM();
    double h2 = 1.0 / c2->getM();

    SECTION("StencilLaplace7p")
    {
        mgcl::StencilLaplace7p stencil(h1);
        auto s2 = stencil.clone(c2->getM(), c2->getN(), c2->getO(), h2);

        REQUIRE(s2->getType() == mgcl::MGCL_LAPLACE_7POINT);
        REQUIRE(std::dynamic_pointer_cast<mgcl::StencilLaplace7p>(s2));
        REQUIRE(stencil.apply(*c, 1, 1, 1) != Catch::Approx(s2->apply(*c2, 1, 1, 1)));
    }

    SECTION("StencilLaplace19p")
    {
        mgcl::StencilLaplace19p stencil(h1);
        auto s2 = stencil.clone(c2->getM(), c2->getN(), c2->getO(), h2);

        REQUIRE(s2->getType() == mgcl::MGCL_LAPLACE_19POINT);
        REQUIRE(std::dynamic_pointer_cast<mgcl::StencilLaplace19p>(s2).get());
        REQUIRE(stencil.apply(*c, 1, 1, 1) != Catch::Approx(s2->apply(*c2, 1, 1, 1)));
    }

    SECTION("StencilLaplace27p")
    {
        mgcl::StencilLaplace27p stencil(h1);
        auto s2 = stencil.clone(c2->getM(), c2->getN(), c2->getO(), h2);

        REQUIRE(s2->getType() == mgcl::MGCL_LAPLACE_27POINT);
        REQUIRE(std::dynamic_pointer_cast<mgcl::StencilLaplace27p>(s2).get());
        REQUIRE(stencil.apply(*c, 1, 1, 1) != Catch::Approx(s2->apply(*c2, 1, 1, 1)));
    }

    SECTION("StencilVarying7p")
    {
        mgcl::StencilVarying7p stencil(c->getM(), c->getN(), c->getO());
        auto s2 = stencil.clone(c2->getM(), c2->getN(), c2->getO(), 1);

        REQUIRE(s2->getType() == mgcl::MGCL_VARYING_7POINT);
        REQUIRE(std::dynamic_pointer_cast<mgcl::StencilVarying7p>(s2).get());

        auto s2casted = std::dynamic_pointer_cast<mgcl::StencilVarying7p>(s2).get();
        (*stencil.getStencilValues())[1][1][1][0] = 3.0;
        (*s2casted->getStencilValues())[1][1][1][0] = 5.0;

        REQUIRE(stencil.apply(*c, 1, 1, 1) != Catch::Approx(s2->apply(*c2, 1, 1, 1)));
    }

    SECTION("StencilVarying19p")
    {
        mgcl::StencilVarying19p stencil(c->getM(), c->getN(), c->getO());
        auto s2 = stencil.clone(c2->getM(), c2->getN(), c2->getO(), 1);

        REQUIRE(s2->getType() == mgcl::MGCL_VARYING_19POINT);
        REQUIRE(std::dynamic_pointer_cast<mgcl::StencilVarying19p>(s2).get());

        auto s2casted = std::dynamic_pointer_cast<mgcl::StencilVarying19p>(s2).get();
        (*stencil.getStencilValues())[1][1][1][0] = 3.0;
        (*s2casted->getStencilValues())[1][1][1][0] = 5.0;

        REQUIRE(stencil.apply(*c, 1, 1, 1) != Catch::Approx(s2->apply(*c2, 1, 1, 1)));
    }

    SECTION("StencilVarying27p")
    {
        mgcl::StencilVarying27p stencil(c->getM(), c->getN(), c->getO());
        auto s2 = stencil.clone(c2->getM(), c2->getN(), c2->getO(), 1);

        REQUIRE(s2->getType() == mgcl::MGCL_VARYING_27POINT);
        REQUIRE(std::dynamic_pointer_cast<mgcl::StencilVarying27p>(s2).get());

        auto s2casted = std::dynamic_pointer_cast<mgcl::StencilVarying27p>(s2).get();
        (*stencil.getStencilValues())[1][1][1][0] = 3.0;
        (*s2casted->getStencilValues())[1][1][1][0] = 5.0;

        REQUIRE(stencil.apply(*c, 1, 1, 1) != Catch::Approx(s2->apply(*c2, 1, 1, 1)));
    }
}
