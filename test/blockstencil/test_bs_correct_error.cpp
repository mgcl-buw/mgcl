#include "catch2/catch_test_macros.hpp"

#include "../../src/mgcl/multigrid_engine.hpp"
#include "../../src/mgcl/problem.hpp"
#include "../cli_args.hpp"
#include "../device_type_generator.hpp"
#include "../test_utility.hpp"

#include <catch2/generators/catch_generators.hpp>
#include <memory>

TEST_CASE("correct_error_bs")
{
    auto deviceType = GENERATE(mgcl_test::deviceTypes(CLI_ARGS::deviceTypes));

    int blocksize = 2;

    // create dummy problem
    auto v_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    auto f_dummy = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    mgcl::Problem p(1, 1, 1, f_dummy, v_dummy);
    p.setUseOpencl(true);
    p.getOpenCLHelper().setPreprocessorConstant("BLOCKSIZE", std::to_string(blocksize));
    p.init();

    int m = 4;
    int n = 4;
    int o = 4;
    int ghosts_m = 1;
    int ghosts_n = 1;
    int ghosts_o = 1;

    mgcl::CuboidBS h_v(m, n, o, ghosts_m, ghosts_n, ghosts_o, blocksize);
    mgcl::CuboidBS h_e(m, n, o, ghosts_m, ghosts_n, ghosts_o, blocksize);
    h_v.fill1dIndex(true);
    h_e.fill1dIndex(true);
    mgcl::CuboidBSGpu v(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, h_v);
    mgcl::CuboidBSGpu e(p.getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, h_e);

    mgcl::args::CorrectErrorBsOclArgs args{
        v, e,
        p.getProgram(), p.getCommands(), p.getContext(),
        nullptr, nullptr};

    mgcl::MultigridEngine::correctErrorBlockstencil(args);
    p.finish();

    // correct error seq
    for (int i = ghosts_m; i < m + ghosts_m; i++)
        for (int j = ghosts_n; j < n + ghosts_n; j++)
            for (int k = ghosts_o; k < o + ghosts_o; k++)
                for (size_t b = 0; b < blocksize; b++)
                    h_v[i][j][k][b] += h_e[i][j][k][b];

    auto v_act = v.read(p.getCommands(), nullptr, true);

    // h_v.dumpToFile("hv.txt");
    // v_act->dumpToFile("dv.txt");

    REQUIRE(v_act->isEqual(h_v));
}