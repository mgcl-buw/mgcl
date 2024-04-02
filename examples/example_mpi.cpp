/**
 * @file example_mpi.cpp
 * @author Simon Hoffmann
 * @brief This example illustrates the usage of mgcl with MPI, especially setting up the domain.
 * @version 0.1
 * @date 2023-07-24
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "mpi.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "../src/mgcl/cuboid.hpp"
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

using std::min;

/**
 * This program is an example of using mgcl with MPI. The domain will be divided into blocks.
 *
 * Arguments:
 * -N <x>[,<y>,<z>], i.e. -N 32 for a grid of size 32^3 or -N 16,32,64 for a grid of size 16x32x64. Default is 16^3
 * -np, --non-periodic If set, the problem will not be periodic but Dirichlet bc's will be used.
 */
int main(int argc, char* argv[])
{
    int m = 16;
    int n = 16;
    int o = 16;
    bool periodic = true;

    // parse input
    InputParser input(argc, argv);

    if (input.cmdOptionExists("-N"))
    {
        auto sizes = split_int(input.getCmdOption("-N"), ",");
        if (sizes.size() == 1)
            m = n = o = sizes[0];
        else if (sizes.size() == 2)
        {
            m = sizes[0];
            n = o = sizes[1];
        }
        else if (sizes.size() == 3)
        {
            m = sizes[0];
            n = sizes[1];
            o = sizes[2];
        }
    }

    if (input.cmdOptionExists("-np") || input.cmdOptionExists("--non-periodic"))
        periodic = false;

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
        std::cout << "  periodic: " << periodic << std::endl
                  << std::endl;
        std::cout << "rank;ms;me;ns;ne;os;oe;ml;nl;ol" << std::endl;
    }

    MPI_Barrier(mpi_comm);

    std::cout << mpi_rank << ";"
              << m_start << ";" << m_end << ";"
              << n_start << ";" << n_end << ";"
              << o_start << ";" << o_end << ";"
              << ml << ";" << nl << ";" << ol
              << std::endl;

    // Init some random data
    auto v = std::make_shared<mgcl::Cuboid>(ml, nl, ol);
    auto f = std::make_shared<mgcl::Cuboid>(ml, nl, ol);
    v->fillRandom();
    f->fillRandom();

    // Create problem, set mpi communicator (needed for topology information) and solve.
    mgcl::Problem p(ml, nl, ol, f, v, m, n, o);
    p.setMpiComm(mpi_comm);
    p.setUseOpencl(true);
    p.setStencilType(mgcl::MGCL_VARYING);
    p.getStencilValues()->fillRandom();
    p.setNu1(2);
    p.setNu2(2);
    p.solve();

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
