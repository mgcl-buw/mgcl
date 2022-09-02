#include "catch2/benchmark/catch_benchmark.hpp"
#include "catch2/benchmark/catch_chronometer.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include "../problem.hpp"

TEST_CASE("mgcl benchmarks", "[!benchmark]")
{
    int N = GENERATE(16, 32, 64, 128);
    int m = N;
    int n = N;
    int o = N;

    BENCHMARK_ADVANCED("sequential random values")
    (Catch::Benchmark::Chronometer meter)
    {
        auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
        auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
        v->fillRandom();
        f->fillRandom();

        mgcl::Problem p(m, n, o, f, v);
        p.setSilent(true);
        p.init();
        meter.measure([&p]
                      { return p.solveSeq(); });
    };

    BENCHMARK_ADVANCED("OpenCL random values")
    (Catch::Benchmark::Chronometer meter)
    {
        auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
        auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
        v->fillRandom();
        f->fillRandom();

        mgcl::Problem p(m, n, o, f, v);
        p.setUseOpencl(true);
        p.setSilent(true);
        p.init();
        meter.measure([&p]
                      { return p.solve(); });
    };
}