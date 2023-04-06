#include <catch2/catch_session.hpp>

#include "cli_args.hpp"
#include <iostream>

// cmd args as global variables
std::vector<int> CLI_ARGS::grids;
int CLI_ARGS::minEpochIterations = 0;
int CLI_ARGS::vCycleIterations = 10;

// helper functions
std::vector<int> split_int(std::string s, std::string delimiter);

// Custom main function for benchmarks that takes additional command line arguments, which then are defined as
// global variables since catch does not support accessing args in tests atm.
// Valid arguments are:
// --grids 8,16,32
// --minEpochIterations N
int main(int argc, char *argv[])
{
    // CLI_ARGS::parseArgs(argc, argv);

    // int result = Catch::Session().run(argc, argv);

    // return result;

    Catch::Session session; // There must be exactly one instance

    std::string grids;
    int minEpochIterations = CLI_ARGS::minEpochIterations;
    int vCycleIterations = CLI_ARGS::vCycleIterations;

    // Build a new parser on top of Catch2's
    using namespace Catch::Clara;
    auto cli = session.cli()                            // Get Catch2's command line parser
               | Opt(grids, "grids")                    // bind variable to a new option, with a hint string
                     ["--grids"]                        // the option names it will respond to
               ("grids seperated by ',', i.e. 8,16,32") // description string for the help output

               | Opt(minEpochIterations, "minEpochIterations")      // bind variable to a new option, with a hint string
                     ["--minEpochIterations"]                       // the option names it will respond to
               ("minEpochIterations, 0 is auto, see nanobench doc") // description string for the help output

               | Opt(vCycleIterations, "vCycleIterations") // bind variable to a new option, with a hint string
                     ["--vCycleIterations"]                // the option names it will respond to
               ("vCycleIterations, 10 is default");        // description string for the help output

    // Now pass the new composite back to Catch2 so it uses that
    session.cli(cli);

    // Let Catch2 (using Clara) parse the command line
    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) // Indicates a command line error
        return returnCode;

    // if set on the command line then 'height' is now set at this point
    if (!grids.empty())
    {
        std::cout << "grids: " << grids << std::endl;
        CLI_ARGS::grids = split_int(grids, ",");
    }

    if (minEpochIterations > 0)
    {
        std::cout << "minEpochIterations: " << minEpochIterations << std::endl;
        CLI_ARGS::minEpochIterations = minEpochIterations;
    }

    if (vCycleIterations != 10)
    {
        std::cout << "vCycleIterations: " << vCycleIterations << std::endl;
        CLI_ARGS::vCycleIterations = vCycleIterations;
    }

    return session.run();
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
