#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <iostream>
#include <memory>

#include "../../src/mgcl/cuboid.hpp"
#include "../../src/mgcl/level.hpp"
#include "../../src/mgcl/multigrid_engine.hpp"
#include "../cli_args.hpp"
#include "../device_type_generator.hpp"
#include "../test_results.hpp"
#include "../test_utility.hpp"

std::shared_ptr<mgcl::Cuboid> residualTestInputF();
std::shared_ptr<mgcl::Cuboid> residualTestInputV();
std::shared_ptr<mgcl::Cuboid> residualTestOutputR();

TEST_CASE("seq_bs_residual")
{
    int m = 8;
    int n = 8;
    int o = 8;

    // TODO
}