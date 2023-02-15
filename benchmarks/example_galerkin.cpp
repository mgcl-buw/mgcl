
#include <iostream>
#include <memory>

#include "../src/cuboid.hpp"
#include "../src/mgcl.hpp"
#include "../src/problem.hpp"
#include "../test/test_utility.hpp"

enum RUNS
{
    SEQ,
    OCL,
    SEQ_OCL
};

// arguments:
// 1. arg: int, grid size. Default is 16
// 2. arg: string, determines if seq, ocl or both shall be called. Valid values: seq, ocl, seq+ocl. Default is seq
int main(int argc, char **argv)
{
    int N = 16;
    RUNS runs = SEQ;

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

    if (runs == SEQ || runs == SEQ_OCL)
    {
        std::cout << "Initializing mgcl seq with varying stencil..." << std::endl;
        auto v = std::make_shared<mgcl::Cuboid>(m, n, o);

        mgcl::Problem p(m, n, o, f, v);
        p.setMaxiterVcycles(maxIterVCycles);
        p.setIgnoreTol(false);
        p.setSilent(false);
        p.setNu1(nu1);
        p.setNu2(nu2);
        p.setOmega(omega);

        p.setStencilType(mgcl::MGCL_VARYING_7POINT);
        auto &s = *p.getStencilValues();

        // fill with 7-point Laplace
        for (int i = 0; i < s.getDim1gh(); i++)
            for (int j = 0; j < s.getDim2gh(); j++)
                for (int k = 0; k < s.getDim3gh(); k++)
                {
                    // 7-point Laplace
                    s[i][j][k][0][1][1] = 1;
                    s[i][j][k][1][0][1] = 1;
                    s[i][j][k][1][1][0] = 1;
                    s[i][j][k][1][1][1] = -6;
                    s[i][j][k][1][1][2] = 1;
                    s[i][j][k][1][2][1] = 1;
                    s[i][j][k][2][1][1] = 1;
                }

        p.init();

        p.solveSeq();
    }

    if ((runs == OCL || runs == SEQ_OCL) && mgcl_test::TestUtility::deviceAvailable("", CL_DEVICE_TYPE_GPU))
    {
        std::cout << "Initializing mgcl OpenCL gpu..." << std::endl;
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

        p.init();
        // p.solve();
    }

    return 0;
}
