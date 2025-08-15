#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <catch2/catch_session.hpp>

#include "../cli_args.hpp"
#include "../test_utility.hpp"
#include "mpi.h"

#include <iostream>

std::vector<cl_device_type> CLI_ARGS::deviceTypes;

// Initializes MPI, runs Catch2 tests and finalizes MPI.
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int mpi_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    Catch::Session session; // There must be exactly one instance

    std::string inputDeviceTypes = "";

    using namespace Catch::Clara;
    auto cli = session.cli()                                                                  // Get Catch2's command line parser
               | Opt(inputDeviceTypes, "deviceTypes")                                         // bind variable to a new option, with a hint string
                     ["--deviceTypes"]                                                        // the option names it will respond to
               ("deviceTypes for OpenCL seperated by ',', e.g. 'gpu,cpu'. Default is 'gpu'"); // description string for the help output

    // Now pass the new composite back to Catch2 so it uses that
    session.cli(cli);

    // Let Catch2 (using Clara) parse the command line
    int returnCode = session.applyCommandLine(argc, argv);
    if (returnCode != 0) // Indicates a command line error
        return returnCode;

    // Set device types
    if (inputDeviceTypes.empty())
    {
        CLI_ARGS::deviceTypes.push_back(CL_DEVICE_TYPE_DEFAULT);
    }

    if (inputDeviceTypes.find("gpu") != std::string::npos)
    {
        if (mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_GPU))
            CLI_ARGS::deviceTypes.push_back(CL_DEVICE_TYPE_GPU);
        else
            std::cout << "GPU device type given as argument, but is not available on system." << std::endl;
    }

    if (inputDeviceTypes.find("cpu") != std::string::npos)
    {
        if (mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_CPU))
            CLI_ARGS::deviceTypes.push_back(CL_DEVICE_TYPE_CPU);
        else
            std::cout << "CPU device type given as argument, but is not available on system." << std::endl;
    }

    if (mpi_rank == 0)
    {
        std::cout << "Using the following OpenCL device types: " << std::endl;
        for (auto deviceType : CLI_ARGS::deviceTypes)
            if (deviceType == CL_DEVICE_TYPE_GPU)
                std::cout << "  GPU" << std::endl;
            else if (deviceType == CL_DEVICE_TYPE_CPU)
                std::cout << "  CPU" << std::endl;
    }

    int result = session.run();

    MPI_Finalize();

    return result;
}
