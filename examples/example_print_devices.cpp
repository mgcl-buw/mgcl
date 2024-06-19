
#include <iostream>
#include <memory>

#include "mpi.h"

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/problem.hpp"

// taken from https://stackoverflow.com/a/868894/4108363
class InputParser
{
public:
    InputParser(int& argc, char** argv)
    {
        for (int i = 1; i < argc; ++i)
            this->tokens.push_back(std::string(argv[i]));
    }

    /// @author iain
    const std::string& getCmdOption(const std::string& option) const
    {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->tokens.begin(), this->tokens.end(), option);
        if (itr != this->tokens.end() && ++itr != this->tokens.end())
        {
            return *itr;
        }
        static const std::string empty_string("");
        return empty_string;
    }

    /// @author iain
    bool cmdOptionExists(const std::string& option) const
    {
        return std::find(this->tokens.begin(), this->tokens.end(), option) != this->tokens.end();
    }

private:
    std::vector<std::string> tokens;
};

/*
 * This program creates a problem using OpenCL, which then prints the used devices.
 * This is mainly useful for checking which process uses which device in a multi-device setup.
 * Arguments:
 * --device-name <name> Name of the device to use. If not set, the first available device will be used.
 * --device-type <(cpu|gpu)> Type of OpenCL device to use. Optional.
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

    InputParser input(argc, argv);
    std::string deviceName = "";
    cl_device_type deviceType = CL_DEVICE_TYPE_DEFAULT;

    if (input.cmdOptionExists("--device-name"))
        deviceName = input.getCmdOption("--device-name");

    if (input.cmdOptionExists("--device-type"))
    {
        std::string deviceTypeStr = input.getCmdOption("--device-type");
        if (deviceTypeStr == "cpu")
            deviceType = CL_DEVICE_TYPE_CPU;
        else if (deviceTypeStr == "gpu")
            deviceType = CL_DEVICE_TYPE_GPU;
        else
            throw "Invalid device type. Must be 'cpu' or 'gpu'";
    }

    int N = 16;

    auto f = std::make_shared<mgcl::Cuboid>(N, N, N);
    auto v = std::make_shared<mgcl::Cuboid>(N, N, N);
    mgcl::Problem p(N, N, N, f, v);
    p.setUseOpencl(true);
    p.setDeviceName(deviceName);
    p.setDeviceType(deviceType);

    for (int i = 0; i < mpi_size; i++)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (i == mpi_rank)
        {
            std::cout << "on rank " << i << ": " << std::endl;
            std::cout << "  > dims: " << mpi_dims[0] << "," << mpi_dims[1] << "," << mpi_dims[2] << std::endl
                      << "  > ";
            p.getOpenCLHelper().init();
            std::cout << std::flush;
        }
    }

    MPI_Finalize();

    return 0;
}
