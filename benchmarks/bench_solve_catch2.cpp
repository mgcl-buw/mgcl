#include "nanobench.h"

#include "catch2/benchmark/catch_benchmark.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std::chrono_literals;

#include "../cuboid.hpp"
#include "../problem.hpp"
#include "../test/test_utility.hpp"
#include "bench_render_templates.hpp"
#include "pmg_utility.hpp"

#include "../thirdparty/mgcl_c/mgcl.hpp"
#include "../thirdparty/pmg/mg.h"

TEST_CASE("mgcl catch2 bench console: solve", "[!benchmark][solveWithCatch2]")
{
    std::vector grids{16, 32, 64, 128};
    for (auto N : grids)
    {
        // int N = 16;
        int m = N;
        int n = N;
        int o = N;

        // Problem parameters
        double tol = 1e-20;
        int nu1 = 2;
        int nu2 = 2;
        double omega = 0.8;
        int maxIterVCycles = 30;

        auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
        f->fillRandom(0, 10);

        // if (N >= 128)
        //     b.epochs(3);

        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

            mgcl::Problem p(m, n, o, f, v);
            p.setMaxiterVcycles(maxIterVCycles);
            p.setIgnoreTol(true);
            p.setSilent(true);
            p.setNu1(nu1);
            p.setNu2(nu2);
            p.setOmega(omega);
            // p.init();

            BENCHMARK(std::string("sequential random values, N = ").append(std::to_string(N)).c_str())
            {
                return p.solveSeq();
            };
        }

        if (mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_GPU))
        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

            mgcl::Problem p(m, n, o, f, v);
            p.setMaxiterVcycles(maxIterVCycles);
            p.setIgnoreTol(true);
            p.setUseOpencl(true);
            p.setDeviceType(CL_DEVICE_TYPE_GPU);
            p.setSilent(true);
            p.setNu1(nu1);
            p.setNu2(nu2);
            p.setOmega(omega);

            if (mgcl_test::TestUtility::deviceAvailable("Quadro", p.getDeviceType()))
                p.setDeviceName("Quadro");

            // p.init();
            BENCHMARK(std::string("opencl gpu random values, N = ").append(std::to_string(N)).c_str())
            {
                p.solve();
            };
        }

        {
            // pmg

            // setup MPI
            MPI_Comm mpi_comm_cart = *init_mpi_for_pmg();

            if (mpi_comm_cart == MPI_COMM_NULL)
            {
                std::cout << "mpi_comm_cart is null! Cannot test against pmg." << std::endl;
            }
            else
            {
                auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

                // init 7-point stencil for pmg's jacobi
                int size = 7;
                double h2 = 1.0 / (double)(N * N);
                double *values = new double[size]();
                int *xoff = new int[size]();
                int *yoff = new int[size]();
                int *zoff = new int[size]();

                values[0] = 6.0 / h2;
                for (int i = 1; i <= 6; i++)
                    values[i] = (-1.0) / h2;

                xoff[0] = 0;
                xoff[1] = 1;
                xoff[2] = -1;
                for (int i = 3; i <= 6; i++)
                    xoff[i] = 0;

                for (int i = 0; i <= 2; i++)
                    yoff[i] = 0;
                yoff[3] = 1;
                yoff[4] = -1;
                yoff[5] = 0;
                yoff[6] = 0;

                for (int i = 0; i <= 4; i++)
                    zoff[i] = 0;
                zoff[5] = 1;
                zoff[6] = -1;

                // run with a tolerance that will never be reached thus all vcycle iters are executed
                BENCHMARK(std::string("pmg random values, N = ").append(std::to_string(N)).c_str())
                {
                    mg(v->getData(), f->getData(), maxIterVCycles, tol, m, n, o,
                       0, m - 1, 0, n - 1, 0, o - 1,
                       1, nu1, nu2, omega, size, values, xoff, yoff, zoff, mpi_comm_cart, 1);
                };
            }
        }

        {
            // old mgcl c implementation seq

            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

            mgcl_config *conf;
            mgcl_generate_config(&conf);

            conf->v = v->getData();
            conf->f = f->getData();
            conf->m = N;
            conf->n = N;
            conf->o = N;
            conf->ghosts_in = 0;
            conf->nu1 = nu1;
            conf->nu2 = nu2;
            conf->omega = omega;
            conf->maxiter_vcycles = maxIterVCycles;
            conf->silent = 1;
            conf->ignoreTol = 1;

            BENCHMARK(std::string("old mgcl_c seq random values, N = ").append(std::to_string(N)).c_str())
            {
                mgcl_c_mgcl_seq(conf);
            };
        }

        if (mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_GPU))
        {
            // old mgcl c implementation ocl

            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

            mgcl_config *conf;
            mgcl_generate_config(&conf);

            conf->v = v->getData();
            conf->f = f->getData();
            conf->m = N;
            conf->n = N;
            conf->o = N;
            conf->ghosts_in = 0;
            conf->nu1 = nu1;
            conf->nu2 = nu2;
            conf->omega = omega;
            conf->maxiter_vcycles = maxIterVCycles;
            conf->device_type = CL_DEVICE_TYPE_GPU;
            // conf->kernel_dir = ".";

            if (mgcl_test::TestUtility::deviceAvailable("Quadro", CL_DEVICE_TYPE_GPU))
                conf->device_name = "Quadro";

            conf->silent = 1;
            conf->ignoreTol = 1;
            conf->use_opencl = 1;
            conf->read_results = 1;

            BENCHMARK(std::string("old mgcl_c ocl random values, N = ").append(std::to_string(N)).c_str())
            {
                mgcl_c_mgcl(conf);
            };
        }
    }
    MPI_Finalize();
}
