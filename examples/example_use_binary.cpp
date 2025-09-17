/**
 * This example demonstrates the usage of a prebuilt kernel binary. Only does the setup, not actually solving.
 *
 * The following cli arguments are supported:
 * --device-name <name>: Name of the device to use. If not set, the first available device will be used. Optional.
 * --device-type <(cpu|gpu)>: Type of OpenCL device to use. Optional.
 */

#include <iostream>
#include <string>
#include <vector>

#include "../src/mgcl/problem.hpp"
#include "arg_parser.hpp"

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    using std::min;

    std::vector<int> gridsTBT;
    bool periodic = true;
    std::string deviceName = "";
    std::string deviceTypeStr = "default";
    cl_device_type deviceType = CL_DEVICE_TYPE_DEFAULT;
    mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_VARYING;

    // parse input
    mgcl_examples_helper::ArgParser parser;
    parser.registerValue("device-name", "Name of the device to use (partial names are allowed)", {"--dn"});
    parser.registerEnumValue("device-type", "Choose device type", {"cpu", "gpu"}, {"--dt"});

    try
    {
        parser.parse(argc, argv);

        if (parser.isPresent("--device-name"))
            deviceName = parser.getValue("--device-name");
        if (parser.isPresent("--device-type"))
        {
            deviceTypeStr = parser.getValue("--device-type");
            if (deviceTypeStr == "cpu")
                deviceType = CL_DEVICE_TYPE_CPU;
            else if (deviceTypeStr == "gpu")
                deviceType = CL_DEVICE_TYPE_GPU;
            else
                throw "Invalid device type. Must be 'cpu' or 'gpu'";
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        parser.printHelp();
        return 1;
    }

    // Init some random data
    auto v = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    auto f = std::make_shared<mgcl::Cuboid>(1, 1, 1);
    v->fillRandom();
    f->fillRandom();

    mgcl::Problem p(1, 1, 1, f, v);
    p.setUseOpencl(true);
    p.setDeviceName(deviceName);
    p.setDeviceType(deviceType);
    p.setStencilType(stencilType);
    p.setSilent(false);

    std::string binaryFileName = "binary_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + ".bin";

    auto& oclh = p.getOpenCLHelper();
    oclh.setBinaryFile(binaryFileName);

    std::chrono::microseconds bestDuration = std::chrono::microseconds::max();
    for (int i = 0; i < 10; ++i)
    {
        // remove test.bin if it exists
        std::remove(binaryFileName.c_str());

        auto start = std::chrono::high_resolution_clock::now();
        oclh.init();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        if (duration < bestDuration)
        {
            bestDuration = duration;
        }
    }
    std::cout << "Best OpenCL init with building kernels: " << bestDuration.count() << " us" << std::endl;

    bestDuration = std::chrono::microseconds::max();
    for (int i = 0; i < 10; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        oclh.init();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        if (duration < bestDuration)
        {
            bestDuration = duration;
        }
    }
    std::cout << "Best OpenCL init with loading kernels from binary: " << bestDuration.count() << " us" << std::endl;

    // remove test.bin again
    std::remove(binaryFileName.c_str());

    return 0;
}