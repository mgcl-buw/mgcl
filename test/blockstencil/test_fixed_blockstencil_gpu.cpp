#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <iostream>

#include "../../src/mgcl/cuboid_bs.hpp"
#include "../../src/mgcl/level.hpp"
#include "../../src/mgcl/multigrid_engine.hpp"
#include "../../src/mgcl/stencil.hpp"
#include "../cli_args.hpp"
#include "../device_type_generator.hpp"
#include "../test_results.hpp"
#include "../test_utility.hpp"

TEST_CASE("FixedBlockstencilGpu_ctor")
{
    int blocksize = 2;
    auto v_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    auto f_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    mgcl::Problem p(1, 1, 1, f_dummy, v_dummy);
    p.setUseOpencl(true);
    p.getOpenCLHelper().setPreprocessorConstant("BLOCKSIZE", std::to_string(blocksize));
    p.init();

    int width = 3;
    mgcl::FixedBlockstencilGpu fs(width, blocksize, p.getContext());

    REQUIRE(fs.getBlocksize() == blocksize);
    REQUIRE(fs.getWidth() == width);
    REQUIRE(fs.getSize() == width * width * width * blocksize * blocksize);
    REQUIRE(fs.getBuf().getBuf() != nullptr);
}

TEST_CASE("FixedBlockstencilGpu_ctor_fs_given")
{
    int blocksize = 2;
    auto v_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    auto f_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    mgcl::Problem p(1, 1, 1, f_dummy, v_dummy);
    p.setUseOpencl(true);
    p.getOpenCLHelper().setPreprocessorConstant("BLOCKSIZE", std::to_string(blocksize));
    p.init();

    int width = 3;
    mgcl::FixedBlockstencil h_fs(width, blocksize);
    h_fs.fill1dIndex(false);
    mgcl::FixedBlockstencilGpu fs(h_fs, p.getContext(), p.getCommands());

    REQUIRE(fs.getBlocksize() == blocksize);
    REQUIRE(fs.getWidth() == width);
    REQUIRE(fs.getSize() == width * width * width * blocksize * blocksize);
    REQUIRE(fs.getBuf().getBuf() != nullptr);

    auto tmp = fs.read(p.getCommands(), true);
    REQUIRE(fs.isEqual(p.getCommands(), tmp));
}