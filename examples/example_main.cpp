
#include <iostream>
#include <memory>

#include "../src/cuboid.hpp"
#include "../src/mgcl.hpp"
#include "../src/problem.hpp"
#include "../test/test_utility.hpp"
#include "../thirdparty/mgcl_c/mgcl.hpp"

enum RUNS
{
    SEQ,
    OCL,
    SEQ_OCL
};

enum IMPL
{
    OLD,
    NEW,
    OLD_NEW
};

// arguments:
// 1. arg: int, grid size. Default is 16
// 2. arg: string, determines if seq, ocl or both shall be called. Valid values: seq, ocl, seq+ocl. Default is seq
// 3. arg: string, determines if old or new mgcl shall be used. Valid values: new, old, new+old. Default is new+old.
int main(int argc, char** argv)
{
    int N = 16;
    RUNS runs = SEQ;
    IMPL impl = OLD_NEW;

    if (argc >= 2)
        N = atoi(argv[1]);

    if (argc >= 3)
    {
        if (std::string(argv[2]) == "seq")
            runs = SEQ;
        else if (std::string(argv[2]) == "ocl")
            runs = OCL;
        else if (std::string(argv[2]) == "seq+ocl")
            runs = SEQ_OCL;
    }

    if (argc >= 4)
    {
        if (std::string(argv[3]) == "new")
            impl = NEW;
        else if (std::string(argv[3]) == "old")
            impl = OLD;
        else if (std::string(argv[3]) == "new+old")
            impl = OLD_NEW;
    }

    std::cout << "Running with parameters:" << std::endl;
    std::cout << "  N = " << N << std::endl;

    int m = N;
    int n = N;
    int o = N;

    // Problem parameters
    double tol = 1e-7;
    int nu1 = 2;
    int nu2 = 2;
    double omega = 0.8;
    int maxIterVCycles = 30;

    auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
    // f->fillRandom(0, 10);

    // periodic function 4th order on right hand side
    double h = 1.0 / static_cast<double>(N);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
            {
                double zs = i * h;
                double ys = j * h;
                double xs = k * h;
                double xs2 = xs * xs;
                double ys2 = ys * ys;
                double zs2 = zs * zs;
                double xsm1_2 = (xs - 1) * (xs - 1);
                double ysm1_2 = (ys - 1) * (ys - 1);
                double zsm1_2 = (zs - 1) * (zs - 1);
                double xs3 = xs * xs * xs;
                double ys3 = ys * ys * ys;
                double zs3 = zs * zs * zs;
                double xsm1_3 = (xs - 1) * (xs - 1) * (xs - 1);
                double ysm1_3 = (ys - 1) * (ys - 1) * (ys - 1);
                double zsm1_3 = (zs - 1) * (zs - 1) * (zs - 1);
                double xs4 = xs * xs * xs * xs;
                double ys4 = ys * ys * ys * ys;
                double zs4 = zs * zs * zs * zs;
                double xsm1_4 = (xs - 1) * (xs - 1) * (xs - 1) * (xs - 1);
                double ysm1_4 = (ys - 1) * (ys - 1) * (ys - 1) * (ys - 1);
                double zsm1_4 = (zs - 1) * (zs - 1) * (zs - 1) * (zs - 1);
                (*f)[i][j][k] =
                    -1000000 *
                    (12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_2 + 12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_2 * zsm1_4 +
                     12 * xs4 * ys4 * zs4 * xsm1_2 * ysm1_4 * zsm1_4 + 32 * xs4 * ys4 * zs3 * xsm1_4 * ysm1_4 * zsm1_3 +
                     12 * xs4 * ys4 * zs2 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs4 * ys3 * zs4 * xsm1_4 * ysm1_3 * zsm1_4 +
                     12 * xs4 * ys2 * zs4 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs3 * ys4 * zs4 * xsm1_3 * ysm1_4 * zsm1_4 +
                     12 * xs2 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_4);
            }

    if ((impl == NEW || impl == OLD_NEW) && (runs == SEQ || runs == SEQ_OCL))
    {
        std::cout << "Running mgcl seq..." << std::endl;
        auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

        mgcl::Problem p(m, n, o, f, v);
        p.setMaxiterVcycles(maxIterVCycles);
        p.setIgnoreTol(false);
        p.setSilent(false);
        p.setNu1(nu1);
        p.setNu2(nu2);
        p.setOmega(omega);
        // p.init();

        p.solveSeq();
    }

    if ((impl == NEW || impl == OLD_NEW) && (runs == OCL || runs == SEQ_OCL) && mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_GPU))
    {
        std::cout << "Running mgcl OpenCL gpu..." << std::endl;
        auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

        mgcl::Problem p(m, n, o, f, v);
        p.setMaxiterVcycles(maxIterVCycles);
        p.setIgnoreTol(false);
        p.setUseOpencl(true);
        p.setDeviceType(CL_DEVICE_TYPE_GPU);
        p.setSilent(false);
        p.setNu1(nu1);
        p.setNu2(nu2);
        p.setOmega(omega);

        if (mgcl_test::TestUtility::deviceAvailable("Quadro", p.getDeviceType()))
            p.setDeviceName("Quadro");

        // p.init();
        p.solve();
    }

    if ((impl == OLD || impl == OLD_NEW) && (runs == SEQ || runs == SEQ_OCL))
    {
        // old mgcl c implementation seq
        std::cout << "Running old mgcl seq..." << std::endl;
        auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

        mgcl_config* conf;
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

        mgcl_c_mgcl_seq(conf);
    }

    if ((impl == OLD || impl == OLD_NEW) && (runs == OCL || runs == SEQ_OCL) && mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_GPU))
    {
        // old mgcl c implementation ocl
        std::cout << "Running old mgcl OpenCL gpu..." << std::endl;
        auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

        mgcl_config* conf;
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

        mgcl_c_mgcl(conf);
    }

    return 0;
}
