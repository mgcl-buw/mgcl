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

// forward declarations
static std::vector<std::string> split(std::string s, std::string delimiter);
static std::vector<int> split_int(std::string s, std::string delimiter);

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
    InputParser input(argc, argv);

    if (input.cmdOptionExists("--device-name"))
        deviceName = input.getCmdOption("--device-name");

    if (input.cmdOptionExists("--device-type"))
    {
        deviceTypeStr = input.getCmdOption("--device-type");
        if (deviceTypeStr == "cpu")
            deviceType = CL_DEVICE_TYPE_CPU;
        else if (deviceTypeStr == "gpu")
            deviceType = CL_DEVICE_TYPE_GPU;
        else
            throw "Invalid device type. Must be 'cpu' or 'gpu'";
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

// taken from https://stackoverflow.com/a/46931770/4108363
static std::vector<std::string> split(std::string s, std::string delimiter)
{
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos)
    {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }

    res.push_back(s.substr(pos_start));
    return res;
}

static std::vector<int> split_int(std::string s, std::string delimiter)
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
