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

    mgcl_examples_helper::ArgParser parser;
    parser.registerFlag("non-periodic", "Disable periodic behavior", {"-np"});
    parser.registerIntList("grids", "Specify a list of integers", {"--grids"});
    parser.registerValue("device-name", "Name of the device to use (partial names are allowed)", {"--dn"});
    parser.registerEnumValue("device-type", "Choose device type", {"cpu", "gpu"}, {"--dt"});
    parser.registerEnumValue("stencil-type", "Stencil type", {"l7", "l19", "l27", "var"}, {"--st"});

    try
    {
        parser.parse(argc, argv);

        gridsTBT = parser.getIntList("--grids");
        periodic = !parser.isPresent("--non-periodic");
        deviceName = parser.getValue("--device-name");
        deviceTypeStr = parser.getValue("--device-type");
        if (deviceTypeStr == "cpu")
            deviceType = CL_DEVICE_TYPE_CPU;
        else if (deviceTypeStr == "gpu")
            deviceType = CL_DEVICE_TYPE_GPU;
        else
            throw "Invalid device type. Must be 'cpu' or 'gpu'";

        std::string st = parser.getValue("--stencil-type");
        if (st == "l7")
            stencilType = mgcl::MGCL_LAPLACE_7POINT;
        else if (st == "l19")
            stencilType = mgcl::MGCL_LAPLACE_19POINT;
        else if (st == "l27")
            stencilType = mgcl::MGCL_LAPLACE_27POINT;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        parser.printHelp();
        return 1;
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