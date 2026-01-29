/**
 * This example illustrates the usage of mgcl with MPI, especially setting up the domain.
 */

#include "mpi.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/kernel_config.hpp"
#include "../src/mgcl/problem.hpp"
#include "arg_parser.hpp"

// forward declarations
static std::vector<std::string> split(std::string s, std::string delimiter);
static std::vector<int> split_int(std::string s, std::string delimiter);

using std::min;

/**
 * This program is an example of using mgcl with MPI. The domain will be divided into blocks.
 *
 * Arguments:
 * -N <x>[,<y>,<z>], i.e. -N 32 for a grid of size 32^3 or -N 16,32,64 for a grid of size 16x32x64. Default is 16^3
 * -np, --non-periodic If set, the problem will not be periodic but Dirichlet bc's will be used.
 * --device-name <name> Name of the device to use. If not set, the first available device will be used.
 * --device-type <(cpu|gpu)> Type of OpenCL device to use. Optional.
 * --stencil-type <l7|l19|l27|var> Stencil type that shall be used. l(7,19,27): Use Laplace stencil with given size.
 *   var: Use varying stencil (default)
 * -seq, -sequential Run in sequential mode on host (no OpenCL)
 */
int main(int argc, char* argv[])
{
    int m = 16;
    int n = 16;
    int o = 16;
    bool periodic = true;
    bool sequential = false;
    std::string deviceName = "";
    std::string deviceTypeStr = "default";
    cl_device_type deviceType = CL_DEVICE_TYPE_ALL;
    mgcl::MGCL_STENCIL stencilType = mgcl::MGCL_VARYING;
    std::string stencilTypeStr = "var";

    mgcl_examples_helper::ArgParser parser;
    parser.registerFlag("sequential", "Run in sequential mode on host", {"--seq"});
    parser.registerFlag("non-periodic", "Disable periodic behavior", {"--np"});
    parser.registerIntList("N", "Specify a list of integers", {"--N"});
    parser.registerEnumValue("device-type", "Choose device type", {"cpu", "gpu"}, {"--dt"});
    parser.registerValue("device-name", "Name of the device to use (partial names are allowed)", {"--dn"});
    parser.registerEnumValue("stencil-type", "Stencil type", {"l7", "l19", "l27", "var"}, {"--st"});

    try
    {
        parser.parse(argc, argv);

        periodic = !parser.isPresent("--np");
        sequential = parser.isPresent("--seq");

        if (parser.isPresent("--N"))
        {
            auto values = parser.getIntList("--N");
            if (values.size() == 1)
                m = n = o = values[0];
            else if (values.size() == 2)
            {
                m = values[0];
                n = o = values[1];
            }
            else if (values.size() == 3)
            {
                m = values[0];
                n = values[1];
                o = values[2];
            }
        }

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

        if (parser.isPresent("--stencil-type"))
        {
            stencilTypeStr = parser.getValue("--stencil-type");
            if (stencilTypeStr == "l7")
                stencilType = mgcl::MGCL_LAPLACE_7POINT;
            else if (stencilTypeStr == "l19")
                stencilType = mgcl::MGCL_LAPLACE_19POINT;
            else if (stencilTypeStr == "l27")
                stencilType = mgcl::MGCL_LAPLACE_27POINT;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        parser.printHelp();
        return 1;
    }

    /* MPI variables */
    int mpi_size;
    int mpi_rank;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    int mpi_dims[3] = {0, 0, 0};
    int mpi_periods[3] = {periodic, periodic, periodic};
    int mpi_coords[3];

    /* Initializing MPI */
    MPI_Init(&argc, &argv);

    /* Initialize cartesian process grid */
    MPI_Comm_size(mpi_comm, &mpi_size);
    MPI_Dims_create(mpi_size, 3, mpi_dims);
    MPI_Cart_create(mpi_comm, 3, mpi_dims, mpi_periods, 1, &mpi_comm);
    MPI_Comm_rank(mpi_comm, &mpi_rank);
    MPI_Cart_coords(mpi_comm, mpi_rank, 3, mpi_coords);

    /* Initialize start and end for local grid */
    int m_start = (m / mpi_dims[0]) * mpi_coords[0] + min(mpi_coords[0], (m % mpi_dims[0]));
    int m_end = (m / mpi_dims[0]) * (mpi_coords[0] + 1) + min(mpi_coords[0] + 1, (m % mpi_dims[0])) - 1;
    int n_start = (n / mpi_dims[1]) * mpi_coords[1] + min(mpi_coords[1], (n % mpi_dims[1]));
    int n_end = (n / mpi_dims[1]) * (mpi_coords[1] + 1) + min(mpi_coords[1] + 1, (n % mpi_dims[1])) - 1;
    int o_start = (o / mpi_dims[2]) * mpi_coords[2] + min(mpi_coords[2], (o % mpi_dims[2]));
    int o_end = (o / mpi_dims[2]) * (mpi_coords[2] + 1) + min(mpi_coords[2] + 1, (o % mpi_dims[2])) - 1;

    int ml = (m_end - m_start) + 1;
    int nl = (n_end - n_start) + 1;
    int ol = (o_end - o_start) + 1;

    if (mpi_rank == 0)
    {
        std::cout << "Arguments:" << std::endl;
        std::cout << "  m,n,o: " << m << "," << n << "," << o << "," << std::endl;
        std::cout << "  periodic: " << std::boolalpha << periodic << std::endl;
        std::cout << "  sequential: " << std::boolalpha << sequential << std::endl;
        std::cout << "  stencilType: " << stencilTypeStr << std::endl;
        std::cout << "  rank;ms;me;ns;ne;os;oe;ml;nl;ol" << std::endl;
    }

    MPI_Barrier(mpi_comm);

    for (int i = 0; i < mpi_size; i++)
    {
        MPI_Barrier(mpi_comm);
        if (i == mpi_rank)
        {
            std::cout << "  " << mpi_rank << ";"
                      << m_start << ";" << m_end << ";"
                      << n_start << ";" << n_end << ";"
                      << o_start << ";" << o_end << ";"
                      << ml << ";" << nl << ";" << ol
                      << std::endl;
        }
    }

    MPI_Barrier(mpi_comm);
    std::cout << "Generating random data..." << std::endl;

    // Init some random data
    int ghin = 1;
    auto v = std::make_shared<mgcl::Cuboid>(ml, nl, ol, ghin, ghin, ghin);
    auto f = std::make_shared<mgcl::Cuboid>(ml, nl, ol, ghin, ghin, ghin);
    v->fillRandom();
    f->fillRandom();

    // Create problem, set mpi communicator (needed for topology information) and solve.
    mgcl::Problem p(ml, nl, ol, f, v, m, n, o);
    p.setMpiComm(mpi_comm);
    if (!sequential)
    {
        p.setUseOpencl(true);
        p.setDeviceName(deviceName);
        p.setDeviceType(deviceType);
    }
    p.setGhostsIn(ghin);
    p.setBc(periodic ? mgcl::BC::PERIODIC : mgcl::BC::DIRICHLET);
    p.setStencilType(stencilType);
    if (stencilType == mgcl::MGCL_VARYING)
    {
        auto& s = *p.createStencilValues();

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
    p.setNu1(1);
    p.setNu2(1);
    p.setMaxiterVcycles(1);

    // Optional: Edit the kernel configuration, i.e. set work-group sizes for each kernel. Below are the default values,
    // taken from kernel_config.cpp. Just fill in the desired work-group sizes. Make sure to not change the kernel names
    // and respect the kernel dimensionality. E.g. the jacobi_iter_27point_varying_stencil_1d kernel is a 1d kernel
    // where only the first dimension is used.
    auto& conf = p.getKernelConfig();

    // Jacobi kernels
    // conf["jacobi_iter_27point_varying_stencil_1d"] = mgcl::conf::KernelWorkgroupSizes{{1, {512, 1, 1}}};
    // conf["jacobi_iter_7point"] = mgcl::conf::KernelWorkgroupSizes{{1, {1, 64, 1}}};
    // conf["jacobi_iter_19point"] = mgcl::conf::KernelWorkgroupSizes{{1, {1, 64, 1}}};
    // conf["jacobi_iter_27point"] = mgcl::conf::KernelWorkgroupSizes{{1, {1, 64, 1}}};

    // // Residual kernels
    // conf["residual_27point_varying_stencil"] = mgcl::conf::KernelWorkgroupSizes{{1, {512, 1, 1}}};
    // conf["residual_7point"] = mgcl::conf::KernelWorkgroupSizes{{1, {512, 1, 1}}};
    // conf["residual_19point"] = mgcl::conf::KernelWorkgroupSizes{{1, {512, 1, 1}}};
    // conf["residual_27point"] = mgcl::conf::KernelWorkgroupSizes{{1, {512, 1, 1}}};
    // conf["residual_squared"] = mgcl::conf::KernelWorkgroupSizes{{1, {512, 1, 1}}};

    // // Ghost update kernels
    // conf["update_ghosts_periodic"] = mgcl::conf::KernelWorkgroupSizes{{1, {4, 4, 4}}};
    // conf["extract_border_planes"] = mgcl::conf::KernelWorkgroupSizes{{1, {32, 1, 1}}};
    // conf["paste_ghosts_from_border_planes"] = mgcl::conf::KernelWorkgroupSizes{{1, {32, 1, 1}}};

    // // Copy buffer kernels
    // conf["copy_input_data"] = mgcl::conf::KernelWorkgroupSizes{{1, {4, 4, 4}}};
    // conf["copy_output_data"] = mgcl::conf::KernelWorkgroupSizes{{1, {4, 4, 4}}};

    // // V-cycle kernels
    // conf["correct_error"] = mgcl::conf::KernelWorkgroupSizes{{1, {4, 4, 4}}};
    // conf["restrict_to_coarse"] = mgcl::conf::KernelWorkgroupSizes{{1, {4, 4, 4}}};
    // conf["prolongate_to_fine"] = mgcl::conf::KernelWorkgroupSizes{{1, {4, 4, 4}}};

    // // Stencil kernels
    // conf["update_ghosts_varying_stencil"] = mgcl::conf::KernelWorkgroupSizes{{1, {4, 4, 4}}};
    // conf["mult_stencils_var_var"] = mgcl::conf::KernelWorkgroupSizes{{1, {4, 4, 4}}};
    // conf["mult_stencils_var_fix"] = mgcl::conf::KernelWorkgroupSizes{{1, {4, 4, 4}}};
    // conf["mult_stencils_fix_var"] = mgcl::conf::KernelWorkgroupSizes{{1, {4, 4, 4}}};
    // conf["cut_stencils_w7_to_w3"] = mgcl::conf::KernelWorkgroupSizes{{1, {4, 4, 4}}};

    // // Utility kernels
    // conf["sum_partial_global_eq_x_num_elements"] = mgcl::conf::KernelWorkgroupSizes{{1, {256, 1, 1}}};
    // // c["sum_finish"] = mgcl::conf::KernelWorkgroupSizes{{1, {4, 4, 4}}}; // Launches only 1 wi
    // conf["max_partial_global_eq_x_num_elements"] = mgcl::conf::KernelWorkgroupSizes{{1, {256, 1, 1}}};
    // // c["max_finish"] = mgcl::conf::KernelWorkgroupSizes{{1, {4, 4, 4}}}; // Launches only 1 wi
    // conf["max_abs_partial_global_eq_x_num_elements"] = mgcl::conf::KernelWorkgroupSizes{{1, {256, 1, 1}}};

    std::cout << "Initializing problem..." << std::endl;
    p.init();
    MPI_Barrier(p.getMpiComm());

    std::cout << "Solving..." << std::endl;
    p.solve(true);

    MPI_Finalize();

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
