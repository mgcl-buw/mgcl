#ifndef MGCL_CLI_ARGS_HPP
#define MGCL_CLI_ARGS_HPP

#include <algorithm>
#include <string>
#include <vector>

namespace CLI_ARGS
{
    extern std::vector<int> grids;
    extern int minEpochIterations;
    extern int bench_epochs;
    extern int bench_iterations;
    extern int vCycleIterations;
    extern std::string outputPath;

    // 3-component vectors
    extern std::vector<int> gridsMin; // e.g. {4,4,4}
    extern std::vector<int> gridsMax; // e.g. {64,64,32}

    extern int nu1;
    extern int nu2;
    extern int blocksize;

    extern std::vector<int> jacobiIters;
    extern std::vector<int> jacobiStepsPerIter;

    extern std::vector<int> elements;

    extern bool checkResults;
    extern bool enableKernelProfiling;
    extern bool useBinaryFile;
}

#endif // MGCL_CLI_ARGS_HPP
