
#include <iostream>
#include <memory>

#include "arg_parser.hpp"
#include "mpi.h"

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/problem.hpp"

/*
 * This program creates a problem using OpenCL, which then prints the used devices.
 * This is mainly useful for checking which process uses which device in a multi-device setup.
 * Arguments:
 * --device-name <name> Name of the device to use. If not set, the first available device will be used.
 * --device-type <(cpu|gpu)> Type of OpenCL device to use. Optional.
 * --device-strategy <(first|distribute)> How the OpenCL device shall be selected. distribute means that each mpi
 *    process gets its own device. Defaults to first.
 */
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int mpi_size;
    int mpi_rank;
    int mpi_dims[3] = {0, 0, 0};

    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Dims_create(mpi_size, 3, mpi_dims);

    std::string deviceName = "";
    cl_device_type deviceType = CL_DEVICE_TYPE_DEFAULT;
    std::string deviceTypeStr = "default";

    mgcl::OCL_DEVICE_STRATEGY deviceStrategy = mgcl::OCL_DEVICE_STRATEGY::ALWAYS_FIRST;

    mgcl_examples_helper::ArgParser parser;
    parser.registerEnumValue("device-type", "Choose device type", {"cpu", "gpu"}, {"--dt"});
    parser.registerValue("device-name", "Name of the device to use (partial names are allowed)", {"--dn"});
    parser.registerEnumValue("device-strategy", "Choose device strategy", {"first", "distribute"}, {"--ds"});

    try
    {
        parser.parse(argc, argv);

        if (parser.isPresent("--device-type"))
        {
            deviceTypeStr = parser.getValue("--device-type");
            if (deviceTypeStr == "cpu")
                deviceType = CL_DEVICE_TYPE_CPU;
            else if (deviceTypeStr == "gpu")
                deviceType = CL_DEVICE_TYPE_GPU;
        }

        if (parser.isPresent("--device-name"))
        {
            deviceName = parser.getValue("--device-name");
        }

        if (parser.isPresent("--device-strategy"))
        {
            std::string deviceStrategyStr = parser.getValue("--device-strategy");
            if (deviceStrategyStr == "first")
                deviceStrategy = mgcl::OCL_DEVICE_STRATEGY::ALWAYS_FIRST;
            else if (deviceStrategyStr == "distribute")
                deviceStrategy = mgcl::OCL_DEVICE_STRATEGY::DISTRIBUTE_EVENLY;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        parser.printHelp();
        return 1;
    }

    int N = 4;

    auto f = std::make_shared<mgcl::Cuboid>(N, N, N);
    auto v = std::make_shared<mgcl::Cuboid>(N, N, N);
    mgcl::Problem p(N, N, N, f, v);
    p.setUseOpencl(true);
    p.setDeviceName(deviceName);
    p.setDeviceType(deviceType);
    p.setDeviceStrategy(deviceStrategy);

    for (int i = 0; i < mpi_size; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == mpi_rank)
        {
            std::cout << "available devices on rank " << i << ": (dims: " << mpi_dims[0] << "," << mpi_dims[1] << "," << mpi_dims[2] << ")" << std::endl
                      << p.getOpenCLHelper().availableDevicesInfo()
                      << "-----" << std::endl;
            // p.getOpenCLHelper().init();
            std::cout << std::flush;
        }
    }

    for (int i = 0; i < mpi_size; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == mpi_rank)
        {
            std::cout << "initializing OpenCLHelper on rank " << i << ": " << std::endl;
            p.getOpenCLHelper().init(mpi_rank);
            std::cout << "-----" << std::endl;
            std::cout << std::flush;
        }
    }

    MPI_Finalize();

    return 0;
}
