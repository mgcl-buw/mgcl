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

    extern std::vector<int> jacobiIters;
    extern std::vector<int> jacobiStepsPerIter;

    extern std::vector<int> elements;

    extern bool checkResults;
}

// // Simple wrapper for command line args which are static variables.
// class CLI_ARGS
// {
// public:
//     static std::vector<int> grids;

//     // parses args, ignore unrecognized ones.
//     static void parseArgs(int argc, char *argv[])
//     {
//         if (cmdOptionExists(argv, argc + argv, "--grids"))
//         {
//             std::string g = getCmdOption(argv, argv + argc, "--grids");
//             grids = split_int(g, ",");
//         }
//     }

// private:
//     // taken from https://stackoverflow.com/a/868894/4108363
//     static std::string getCmdOption(char **begin, char **end, const std::string &option)
//     {
//         char **itr = std::find(begin, end, option);
//         if (itr != end && ++itr != end)
//         {
//             return std::string(*itr);
//         }
//         return "";
//     }

//     static bool cmdOptionExists(char **begin, char **end, const std::string &option)
//     {
//         return std::find(begin, end, option) != end;
//     }

//     // taken from https://stackoverflow.com/a/46931770/4108363
//     static std::vector<std::string> split(std::string s, std::string delimiter)
//     {
//         size_t pos_start = 0, pos_end, delim_len = delimiter.length();
//         std::string token;
//         std::vector<std::string> res;

//         while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos)
//         {
//             token = s.substr(pos_start, pos_end - pos_start);
//             pos_start = pos_end + delim_len;
//             res.push_back(token);
//         }

//         res.push_back(s.substr(pos_start));
//         return res;
//     }

//     static std::vector<int> split_int(std::string s, std::string delimiter)
//     {
//         size_t pos_start = 0, pos_end, delim_len = delimiter.length();
//         std::string token;
//         std::vector<int> res;

//         while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos)
//         {
//             token = s.substr(pos_start, pos_end - pos_start);
//             pos_start = pos_end + delim_len;
//             res.push_back(std::stoi(token));
//         }

//         res.push_back(std::stoi(s.substr(pos_start)));
//         return res;
//     }
// };

#endif // MGCL_CLI_ARGS_HPP
