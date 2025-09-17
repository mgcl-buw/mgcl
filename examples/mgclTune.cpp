#include <cstddef>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include <mpi.h>

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/level.hpp"
#include "../src/mgcl/mgcl.hpp"
#include "../src/mgcl/multigrid_engine.hpp"
#include "../src/mgcl/problem.hpp"
#include "arg_parser.hpp"

struct BestKernelConf
{
    std::string kernelName;
    std::string gridSize;
    std::array<size_t, 3> work_groups;
};

void checkResidual(
    mgcl::conf::KernelConfig& conf, size_t N, int iters, bool verbose, mgcl::Problem& p, mgcl::Level& level,
    std::vector<BestKernelConf>& bestKernelConfs);
void checkJacobi(
    mgcl::conf::KernelConfig& conf, size_t N, int iters, bool verbose, mgcl::Problem& p, mgcl::Level& level,
    std::vector<BestKernelConf>& bestKernelConfs);

// Arguments:
// --grids <list>: Grids to be tested, e.g. --grids 4,8,16
//  --verbose, -v: Verbose output
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    std::vector<int> grids;
    int iters = 5;       // number of iterations for running a kernel
    bool verbose = true; // if true, every iteration will be printed

    mgcl_examples_helper::ArgParser parser;
    parser.registerFlag("verbose", "Verbose output", {"-v", "--verbose"});
    parser.registerIntList("grids", "Specify grids to be tested", {"--grids"});

    try
    {
        parser.parse(argc, argv);

        grids = parser.getIntList("--grids");
        if (grids.empty())
        {
            std::cout << "Must specify --grids <list>, e.g. --grids 4,8,16" << std::endl;
            return 1;
        }

        verbose = parser.isPresent("--verbose");
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        parser.printHelp();
        return 1;
    }

    std::cout << "Running with parameters:" << std::endl;
    std::cout << "  grids: {";
    for (int N : grids)
        std::cout << N << ",";
    std::cout << "}" << std::endl;

    // data structure to hold best timings for each kernel per grid
    std::vector<BestKernelConf> bestKernelConfs;

    // loop through grids
    for (auto N : grids)
    {
        int m = N;
        int n = N;
        int o = N;

        // Create sample problem using OpenCL
        auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
        auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
        mgcl::Problem p(N, N, N, f, v);
        p.setUseOpencl(true);
        p.setProfilingEnabled(true);
        p.setStencilType(mgcl::MGCL_VARYING);
        p.setSilent(true);

        // Fill random coefficients
        auto svptr = p.getStencilValues();
        auto& sv = *svptr;
        sv.fillRandom();

        p.init();

        auto& level = p.getLevelAt(0);

        auto& conf = p.getKernelConfig();

        if (verbose)
        {
            std::cout << std::endl
                      << "N: " << N << std::endl;
        }

        checkResidual(conf, N, iters, verbose, p, level, bestKernelConfs);
        checkJacobi(conf, N, iters, verbose, p, level, bestKernelConfs);
    }

    MPI_Finalize();
}

void checkResidual(
    mgcl::conf::KernelConfig& conf, size_t N, int iters, bool verbose, mgcl::Problem& p, mgcl::Level& level,
    std::vector<BestKernelConf>& bestKernelConfs)
{
    std::string kernelName = "residual_27point_varying_stencil";
    std::vector<size_t> wgs{32, 64, 128, 256, 512};
    std::map<size_t, unsigned long> bestTimePerWg;
    for (auto wg : wgs)
    {
        conf[kernelName] = {{N * N * N, {wg, 1, 1}}};

        for (int i = 0; i < iters; i++)
        {
            mgcl::MultigridEngine::residual(p, level, false);
        }

        auto pd = p.getProfilingData();
        // find minimum elapsed time
        auto mintime = std::min_element(pd->getMeasurements()[kernelName].begin(),
                                        pd->getMeasurements()[kernelName].end(),
                                        [](const auto& a, const auto& b)
                                        { return a.start_to_end < b.start_to_end; })
                           ->start_to_end;
        bestTimePerWg[wg] = mintime;

        // reset profiling data for next wg
        pd->getMeasurements().clear();
    }

    // iterate through bestTimePerWg and print
    if (verbose)
    {
        std::cout << kernelName << std::endl;
        for (const auto& n : bestTimePerWg)
            std::cout << "  wg " << std::setw(4) << n.first << ": " << std::setw(10) << n.second << " ns" << std::endl;
    }

    // find minimum wg across all wgs
    auto minwg = std::min_element(bestTimePerWg.begin(), bestTimePerWg.end(),
                                  [](const auto& a, const auto& b)
                                  { return a.second < b.second; })
                     ->first;

    // store in result
    bestKernelConfs.push_back({kernelName, std::to_string(N), {minwg, 0, 0}});
}

void checkJacobi(
    mgcl::conf::KernelConfig& conf, size_t N, int iters, bool verbose, mgcl::Problem& p, mgcl::Level& level,
    std::vector<BestKernelConf>& bestKernelConfs)
{
    std::string kernelName = "jacobi_iter_27point_varying_stencil_1d";
    std::vector<size_t> wgs{32, 64, 128, 256, 512};
    std::map<size_t, unsigned long> bestTimePerWg;
    for (auto wg : wgs)
    {
        conf[kernelName] = {{N * N * N, {wg, 1, 1}}};

        for (int i = 0; i < iters; i++)
        {
            mgcl::MultigridEngine::jacobi(p, level, 3, false);
        }

        auto pd = p.getProfilingData();
        // find minimum elapsed time
        auto mintime = std::min_element(pd->getMeasurements()[kernelName].begin(),
                                        pd->getMeasurements()[kernelName].end(),
                                        [](const auto& a, const auto& b)
                                        { return a.start_to_end < b.start_to_end; })
                           ->start_to_end;
        bestTimePerWg[wg] = mintime;

        // reset profiling data for next wg
        pd->getMeasurements().clear();
    }

    // iterate through bestTimePerWg and print
    if (verbose)
    {
        std::cout << kernelName << std::endl;
        for (const auto& n : bestTimePerWg)
            std::cout << "  wg " << std::setw(4) << n.first << ": " << std::setw(10) << n.second << " ns" << std::endl;
    }

    // find minimum wg across all wgs
    auto minwg = std::min_element(bestTimePerWg.begin(), bestTimePerWg.end(),
                                  [](const auto& a, const auto& b)
                                  { return a.second < b.second; })
                     ->first;

    // store in result
    bestKernelConfs.push_back({kernelName, std::to_string(N), {minwg, 0, 0}});
}