#define ANKERL_NANOBENCH_IMPLEMENT
#include "../include/nanobench.h"

#include <chrono>
#include <vector>
using namespace std::chrono_literals;

#include "../cuboid.hpp"
#include "../problem.hpp"

void runSeq(ankerl::nanobench::Bench *b, std::string name, int grid);
void runGpu(ankerl::nanobench::Bench *b, std::string name, int grid);

int main()
{
    std::vector grids{16, 32, 64, 128};
    // int N = 16;

    for (auto g : grids)
    {
        ankerl::nanobench::Bench b;
        b.timeUnit(1ms, "ms")
            .warmup(3)
            .minEpochTime(50ms)
            .relative(true);

        runSeq(&b, std::string("Sequential random values, N = ").append(std::to_string(g)).c_str(), g);
        runGpu(&b, std::string("GPU random values, N = ").append(std::to_string(g)).c_str(), g);
    }
}

void runSeq(ankerl::nanobench::Bench *b, std::string name, int grid)
{
    int m = grid;
    int n = grid;
    int o = grid;

    auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
    auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
    v->fillRandom();
    f->fillRandom();

    mgcl::Problem p(m, n, o, f, v);
    // p.setSilent(true);
    p.init();

    b->run(name, [&]
           { p.solveSeq(); });
}

void runGpu(ankerl::nanobench::Bench *b, std::string name, int grid)
{
    int m = grid;
    int n = grid;
    int o = grid;

    auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
    auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
    v->fillRandom();
    f->fillRandom();

    mgcl::Problem p(m, n, o, f, v);
    p.setUseOpencl(true);
    p.setDeviceType(CL_DEVICE_TYPE_GPU);
    // p.setSilent(true);
    p.init();
    b->run(name, [&]
           { p.solve(); });
}