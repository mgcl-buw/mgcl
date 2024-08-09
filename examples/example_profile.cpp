/**
 * This example demonstrates the usage of mgcl's kernel profiling feature.
 *
 * The following cli arguments are supported:
 * --grids <list>: Grids to be tested, e.g. --grids 4,8,16. Mandatory.
 * --device-name <name>: Name of the device to use. If not set, the first available device will be used. Optional.
 * --device-type <(cpu|gpu)>: Type of OpenCL device to use. Optional.
 * -np, --non-periodic: If set, the problem will not be periodic but Dirichlet bc's will be used.
 * --stencil-type <l7|l19|l27|var>: Stencil type that shall be used. l(7,19,27): Use Laplace stencil with given size.
 *   var: Use varying stencil (default)
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

    if (!input.cmdOptionExists("--grids"))
    {
        throw "Must specify --grids argument, e.g. --grids 4,8,16";
    }
    else
    {
        gridsTBT = split_int(input.getCmdOption("--grids"), ",");
    }

    ;

    if (input.cmdOptionExists("-np") || input.cmdOptionExists("--non-periodic"))
        periodic = false;

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

    if (input.cmdOptionExists("--stencil-type"))
    {
        std::string st = input.getCmdOption("--stencil-type");
        if (st == "l7")
            stencilType = mgcl::MGCL_LAPLACE_7POINT;
        else if (st == "l19")
            stencilType = mgcl::MGCL_LAPLACE_19POINT;
        else if (st == "l27")
            stencilType = mgcl::MGCL_LAPLACE_27POINT;
    }

    for (auto g : gridsTBT)
    {
        int m = g;
        int n = g;
        int o = g;

        // Init some random data
        auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
        auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
        v->fillRandom();
        f->fillRandom();

        // Create problem, set mpi communicator (needed for topoogy information) and sove.
        mgcl::Problem p(m, n, o, f, v, m, n, o);
        p.setUseOpencl(true);
        p.setDeviceName(deviceName);
        p.setDeviceType(deviceType);
        p.setStencilType(stencilType);
        p.setProfilingEnabled(true);
        p.setSilent(true);
        if (stencilType == mgcl::MGCL_VARYING)
        {
            auto& s = *p.getStencilValues();

            // fill with 7-point Laplace
            double h2inv = m * m;
            for (int i = 0; i < s.getMgh(); i++)
                for (int j = 0; j < s.getNgh(); j++)
                    for (int k = 0; k < s.getOgh(); k++)
                    {
                        // 7-point Laplace
                        s[0][1][1][i][j][k] = -h2inv;
                        s[1][0][1][i][j][k] = -h2inv;
                        s[1][1][0][i][j][k] = -h2inv;
                        s[1][1][1][i][j][k] = 6 * h2inv;
                        s[1][1][2][i][j][k] = -h2inv;
                        s[1][2][1][i][j][k] = -h2inv;
                        s[2][1][1][i][j][k] = -h2inv;
                    }
        }
        p.setNu1(2);
        p.setNu2(2);

        std::cout << "Solving for m,n,o: " << m << "," << n << "," << o << std::endl;
        p.solve();
        p.getProfilingData()->printBestTimingsPerKernel();

        std::cout << std::endl;
    }

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
