#include <catch2/catch_session.hpp>

#include "cli_args.hpp"
#include <iostream>

#include "mpi.h"

// cmd args as global variables
std::vector<int> CLI_ARGS::grids;
int CLI_ARGS::minEpochIterations = 0;
int CLI_ARGS::bench_epochs = 11;
int CLI_ARGS::bench_iterations = 1;
int CLI_ARGS::vCycleIterations = 10;
std::string CLI_ARGS::outputPath = ".";
std::vector<int> CLI_ARGS::gridsMin{0, 0, 0};    // e.g. {4,4,4}
std::vector<int> CLI_ARGS::gridsMax{-1, -1, -1}; // e.g. {64,64,32}
int CLI_ARGS::nu1 = 2;
int CLI_ARGS::nu2 = 2;
std::vector<int> CLI_ARGS::jacobiIters;
std::vector<int> CLI_ARGS::jacobiStepsPerIter;
std::vector<int> CLI_ARGS::elements;
bool CLI_ARGS::checkResults = false;
bool CLI_ARGS::enableKernelProfiling = false;

// helper functions
std::vector<int> split_int(std::string s, std::string delimiter);

// Custom main function for benchmarks that takes additional command line arguments, which then are defined as
// global variables since catch does not support accessing args in tests atm.
// Valid arguments are:
// --grids 8,16,32
// --minEpochIterations N
// --gridsMin 4,4,4
// --gridsMax 32,32,32
//
// If gridsMin and gridsMax are set, each grid size between those to limits that is a power of 2 is tested.
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int mpi_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    Catch::Session session; // There must be exactly one instance

    std::string grids;
    int minEpochIterations = CLI_ARGS::minEpochIterations;
    int vCycleIterations = CLI_ARGS::vCycleIterations;
    std::string outputPath = ".";
    std::string jacobiIters;        // only relevant for Jacobi multiple iters test
    std::string jacobiStepsPerIter; // only relevant for Jacobi multiple iters test

    std::string elements; // only relevant for data transfer

    std::string gridsMin;
    std::string gridsMax;

    int bench_epochs = CLI_ARGS::bench_epochs;
    int bench_iterations = CLI_ARGS::bench_iterations;

    int nu1 = 2;
    int nu2 = 2;

    // Build a new parser on top of Catch2's
    using namespace Catch::Clara;
    auto cli = session.cli()                            // Get Catch2's command line parser
               | Opt(grids, "grids")                    // bind variable to a new option, with a hint string
                     ["--grids"]                        // the option names it will respond to
               ("grids seperated by ',', i.e. 8,16,32") // description string for the help output

               | Opt(minEpochIterations, "minEpochIterations")      // bind variable to a new option, with a hint string
                     ["--minEpochIterations"]                       // the option names it will respond to
               ("minEpochIterations, 0 is auto, see nanobench doc") // description string for the help output

               | Opt(bench_epochs, "bench_epochs")               // bind variable to a new option, with a hint string
                     ["--benchEpochs"]                           // the option names it will respond to
               ("benchEpochs, 11 is default, see nanobench doc") // description string for the help output

               | Opt(bench_iterations, "bench_iterations")     // bind variable to a new option, with a hint string
                     ["--benchIters"]                          // the option names it will respond to
               ("benchIters, 1 is default, see nanobench doc") // description string for the help output

               | Opt(vCycleIterations, "vCycleIterations") // bind variable to a new option, with a hint string
                     ["--vci"]["--vCycleIterations"]       // the option names it will respond to
               ("vCycleIterations, 10 is default")         // description string for the help output

               | Opt(outputPath, "outputPath")                                        // bind variable to a new option, with a hint string
                     ["--outputPath"]                                                 // the option names it will respond to
               ("specify a path where any output should be written to. Default is .") // description string for the help output

               | Opt(vCycleIterations, "nu1")        // bind variable to a new option, with a hint string
                     ["--nu1"]                       // the option names it will respond to
               ("pre-smoothing steps, 2 is default") // description string for the help output

               | Opt(vCycleIterations, "nu2")         // bind variable to a new option, with a hint string
                     ["--nu2"]                        // the option names it will respond to
               ("post-smoothing steps, 2 is default") // description string for the help output

               | Opt(CLI_ARGS::checkResults)                                       // bind variable to a new option, with a hint string
                     ["--checkResults"]                                            // the option names it will respond to
               ("if true, results will be checked for benchmarks supporting this") // description string for the help output

               | Opt(CLI_ARGS::enableKernelProfiling)        // bind variable to a new option, with a hint string
                     ["--enableKernelProfiling"]             // the option names it will respond to
               ("if true, kernel profiling will be enabled") // description string for the help output

               | Opt(jacobiIters, "jacobiIters")                                                                          // bind variable to a new option, with a hint string
                     ["--jacobiIters"]                                                                                    // the option names it will respond to
               ("jacobi iterations to test seperated by ',', i.e. 1,3,10. Only relevant for Jacobi multiple iters bench") // description string for the help output

               | Opt(jacobiStepsPerIter, "jacobiStepsPerIter")                                                                                          // bind variable to a new option, with a hint string
                     ["--jacobiStepsPerIter"]["--spi"]                                                                                                  // the option names it will respond to
               ("jacobi steps per iteration wihtout ghost update to test seperated by ',', i.e. 1,3,10. Only relevant for Jacobi multiple iters bench") // description string for the help output

               | Opt(elements, "elements")                                                                              // bind variable to a new option, with a hint string
                     ["--elements"]                                                                                     // the option names it will respond to
               ("number of elements to send seperated by ',', i.e. 16384,32768. Only relevant for data transfer tests") // description string for the help output

               | Opt(gridsMin, "gridsMin")                                          // bind variable to a new option, with a hint string
                     ["--gridsMin"]                                                 // the option names it will respond to
               ("minimum size of grids to be tested, seperated by ',', e.g. 4,4,4") // description string for the help output

               | Opt(gridsMax, "gridsMax")                                              // bind variable to a new option, with a hint string
                     ["--gridsMax"]                                                     // the option names it will respond to
               ("maximum size of grids to be tested, seperated by ',', e.g. 64,64,32"); // description string for the help output

    // Now pass the new composite back to Catch2 so it uses that
    session.cli(cli);

    // Let Catch2 (using Clara) parse the command line
    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) // Indicates a command line error
        return returnCode;

    if (!grids.empty())
    {
        if (mpi_rank == 0)
            std::cout << "grids: " << grids << std::endl;
        CLI_ARGS::grids = split_int(grids, ",");
    }

    if (!jacobiIters.empty())
    {
        if (mpi_rank == 0)
            std::cout << "jacobiIters: " << jacobiIters << std::endl;
        CLI_ARGS::jacobiIters = split_int(jacobiIters, ",");
    }

    if (!jacobiStepsPerIter.empty())
    {
        if (mpi_rank == 0)
            std::cout << "jacobiStepsPerIter: " << jacobiStepsPerIter << std::endl;
        CLI_ARGS::jacobiStepsPerIter = split_int(jacobiStepsPerIter, ",");
    }

    if (minEpochIterations > 0)
    {
        if (mpi_rank == 0)
            std::cout << "minEpochIterations: " << minEpochIterations << std::endl;
        CLI_ARGS::minEpochIterations = minEpochIterations;
    }

    if (bench_epochs > 0)
    {
        if (mpi_rank == 0)
            std::cout << "bench_epochs: " << bench_epochs << std::endl;
        CLI_ARGS::bench_epochs = bench_epochs;
    }

    if (bench_iterations > 0)
    {
        if (mpi_rank == 0)
            std::cout << "bench_iterations: " << bench_iterations << std::endl;
        CLI_ARGS::bench_iterations = bench_iterations;
    }

    if (!elements.empty())
    {
        if (mpi_rank == 0)
            std::cout << "elements: " << elements << std::endl;
        CLI_ARGS::elements = split_int(elements, ",");
    }

    if (vCycleIterations != 10)
    {
        if (mpi_rank == 0)
            std::cout << "vCycleIterations: " << vCycleIterations << std::endl;
        CLI_ARGS::vCycleIterations = vCycleIterations;
    }

    if (nu1 != 2)
    {
        if (mpi_rank == 0)
            std::cout << "nu1: " << nu1 << std::endl;
        CLI_ARGS::nu1 = nu1;
    }

    if (nu2 != 2)
    {
        if (mpi_rank == 0)
            std::cout << "nu2: " << nu2 << std::endl;
        CLI_ARGS::nu2 = nu2;
    }

    if (!outputPath.empty() && outputPath != ".")
    {
        // Add trailing slash if not given
        if (outputPath.back() != '/')
            outputPath.append("/");

        if (mpi_rank == 0)
            std::cout << "outputPath: " << outputPath << std::endl;
        CLI_ARGS::outputPath = outputPath;
    }

    if (!gridsMin.empty())
    {
        if (mpi_rank == 0)
            std::cout << "gridsMin: " << gridsMin << std::endl;
        CLI_ARGS::gridsMin = split_int(gridsMin, ",");
    }

    if (!gridsMax.empty())
    {
        if (mpi_rank == 0)
            std::cout << "gridsMax: " << gridsMax << std::endl;
        CLI_ARGS::gridsMax = split_int(gridsMax, ",");
    }

    std::cout << "checkResults: " << CLI_ARGS::checkResults << std::endl;
    std::cout << "enableKernelProfiling: " << CLI_ARGS::enableKernelProfiling << std::endl;

    int result = session.run();

    MPI_Finalize();

    return result;
}

std::vector<int> split_int(std::string s, std::string delimiter)
{
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<int> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos)
    {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(std::stoi(token));
    }

    res.push_back(std::stoi(s.substr(pos_start)));
    return res;
}
