#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <vector>
using namespace std::chrono_literals;

#include "../src/mgcl/cuboid.hpp"
#include "../src/mgcl/cuboid_gpu.hpp"
#include "../src/mgcl/multigrid_engine.hpp"
#include "../src/mgcl/opencl_helper.hpp"
#include "../src/mgcl/problem.hpp"
#include "../src/mgcl/stencil.hpp"
#include "../src/mgcl/util.hpp"
#include "../test/ocl_wrapper.hpp"
#include "../test/test_utility.hpp"
#include "bench_render_templates.hpp"
#include "cli_args.hpp"

struct Result
{
    std::string name;
    double minTime;
    double m;
    double n;
    double o;
    double locm;
    double locn;
    double loco;
};

std::string getKernelOptimizationsFilePath()
{
    std::string filePath = __FILE__;
    std::string dirPath = filePath.substr(0, filePath.rfind("/"));
    return dirPath.append("/kernel_optimizations.cl");
}

void runResidualBench(std::vector<std::vector<int>> gridsTBT, std::vector<std::vector<int>> localsTBT,
                      std::vector<Result>& minTimes,
                      int ghosts, bool return_residual, std::string kernelName, int kernelDim);
std::shared_ptr<mgcl::Cuboid> runJacobiBench(std::vector<std::vector<int>> gridsTBT, std::vector<std::vector<int>> localsTBT,
                                             std::vector<Result>& minTimes,
                                             int ghosts, bool return_residual, std::string kernelName, int kernelDim,
                                             int maxiter, int stepsPerIter, double omega, bool checkResults);

// helper functions
void create4thOrderProblem(mgcl::Cuboid& v, mgcl::Cuboid& f, mgcl::Cuboid& solution);
void sv_fill_27p_laplace(mgcl::VaryingStencil& s);

/*
 * The benchmarks in this file aim to find the optimal execution parameters, i.e. work group sizes, for each step of
 * the multigrid method for a single GPU without MPI. These steps are:
 * - residual
 * - jacobi
 * - ghost update
 * - restriction
 * - prolongation
 * The galerkin operator, and thus stencil arithmetic, is benchmarked separately. See bech_stencil_arithmetic.cpp and
 * bech_stencil_arithmetic_kernels.cpp.
 *
 * The implementation is a copy of residual from 11-02-2023
 */
TEST_CASE("exec_params_residual")
{
    if (CLI_ARGS::grids.size() == 0 && (CLI_ARGS::gridsMin.size() == 0 || CLI_ARGS::gridsMax.size() == 0))
        throw "Need to specify at least one local grid size, e.g. using --grids 4,8,16 or --gridsMin 4,4,4 AND --gridsMax 32,32,32";

    // build grids to be tested from CLI args
    std::vector<std::vector<int>> gridsTBT;
    for (auto N : CLI_ARGS::grids)
        gridsTBT.push_back({N, N, N});
    if (CLI_ARGS::gridsMin.size() > 0 && CLI_ARGS::gridsMax.size() > 0)
        for (int m = CLI_ARGS::gridsMin[0]; m <= CLI_ARGS::gridsMax[0]; m *= 2)
            for (int n = CLI_ARGS::gridsMin[1]; n <= CLI_ARGS::gridsMax[1]; n *= 2)
                for (int o = CLI_ARGS::gridsMin[2]; o <= CLI_ARGS::gridsMax[2]; o *= 2)
                    gridsTBT.push_back({m, n, o});

    std::cout << "Testing the following grid sizes" << std::endl;
    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];
        // std::cout << "  local size: " << m << "," << n << "," << o << ", global sizes: "
        //           << m * mpi_dims[0] << "," << n * mpi_dims[1] << "," << o * mpi_dims[2] << std::endl;
        std::cout << "  " << m << "," << n << "," << o << std::endl;
    }

    std::vector<std::vector<int>> localsTBT3d = {
        {32, 1, 1}, {1, 1, 32}, {64, 1, 1}, {1, 1, 64}, {128, 1, 1}, {4, 4, 4}, {4, 4, 8} //
    };
    std::vector<std::vector<int>> localsTBT1d = {
        {16, 1, 1}, {32, 1, 1}, {64, 1, 1}, {128, 1, 1}, {256, 1, 1}, {512, 1, 1}, //
    };
    std::cout << "Testing the following local sizes for 3d" << std::endl;
    for (auto lo : localsTBT3d)
    {
        int m = lo[0];
        int n = lo[1];
        int o = lo[2];
        std::cout << "  " << m << "," << n << "," << o << std::endl;
    }
    std::cout << "Testing the following local sizes for 1d" << std::endl;
    for (auto lo : localsTBT1d)
    {
        int m = lo[0];
        std::cout << "  " << m << std::endl;
    }

    int ghosts = 1;
    bool return_residual = true;

    std::vector<Result> minTimes;
    runResidualBench(gridsTBT, localsTBT3d, minTimes, ghosts, return_residual, "residual_27point_varying_stencil_3d_one_wi_per_cell", 3);
    runResidualBench(gridsTBT, localsTBT1d, minTimes, ghosts, return_residual, "residual_27point_varying_stencil_1d_one_wi_per_cell", 1);

    std::stringstream ss;
    ss << "name;m;n;o;locm;locn;loco;minTimeInMs" << std::endl;
    for (auto r : minTimes)
    {
        ss << r.name << ";" << r.m << ";" << r.n << ";" << r.o << ";" << //
            r.locm << ";" << r.locn << ";" << r.loco << ";" << std::setprecision(17) << r.minTime << std::endl;
    }
    std::string output = ss.str();
    std::replace(output.begin(), output.end(), '.', ',');
    std::cout << output;
}

/*
 * The benchmarks in this file aim to find the optimal execution parameters, i.e. work group sizes, for each step of
 * the multigrid method for a single GPU without MPI. These steps are:
 * - residual
 * - jacobi
 * - ghost update
 * - restriction
 * - prolongation
 * The galerkin operator, and thus stencil arithmetic, is benchmarked separately. See bech_stencil_arithmetic.cpp and
 * bech_stencil_arithmetic_kernels.cpp.
 *
 * The implementation is a copy of residual from 11-02-2023
 */
TEST_CASE("exec_params_jacobi")
{
    std::cout << "Hint: Use --nu1 to specify jacobi iterations at the command line." << std::endl;

    if (CLI_ARGS::grids.size() == 0 && (CLI_ARGS::gridsMin.size() == 0 || CLI_ARGS::gridsMax.size() == 0))
        throw "Need to specify at least one local grid size, e.g. using --grids 4,8,16 or --gridsMin 4,4,4 AND --gridsMax 32,32,32";

    // build grids to be tested from CLI args
    std::vector<std::vector<int>> gridsTBT;
    for (auto N : CLI_ARGS::grids)
        gridsTBT.push_back({N, N, N});
    if (CLI_ARGS::gridsMin.size() > 0 && CLI_ARGS::gridsMax.size() > 0)
        for (int m = CLI_ARGS::gridsMin[0]; m <= CLI_ARGS::gridsMax[0]; m *= 2)
            for (int n = CLI_ARGS::gridsMin[1]; n <= CLI_ARGS::gridsMax[1]; n *= 2)
                for (int o = CLI_ARGS::gridsMin[2]; o <= CLI_ARGS::gridsMax[2]; o *= 2)
                    gridsTBT.push_back({m, n, o});

    std::cout << "Testing the following grid sizes" << std::endl;
    for (auto gr : gridsTBT)
    {
        int m = gr[0];
        int n = gr[1];
        int o = gr[2];
        // std::cout << "  local size: " << m << "," << n << "," << o << ", global sizes: "
        //           << m * mpi_dims[0] << "," << n * mpi_dims[1] << "," << o * mpi_dims[2] << std::endl;
        std::cout << "  " << m << "," << n << "," << o << std::endl;
    }

    std::vector<std::vector<int>> localsTBT3d = {
        {1, 1, 16}, {1, 1, 32}, {1, 4, 32}, {1, 1, 64}, {4, 4, 4}, {4, 4, 8}, {8, 8, 8} //
    };
    std::vector<std::vector<int>> localsTBT2d = {
        {1, 16, 1}, {1, 32, 1}, {4, 8, 1}, {8, 4, 1}, {4, 4, 1}, {4, 16, 1} //
    };
    std::vector<std::vector<int>> localsTBT1d = {
        {16, 1, 1}, {32, 1, 1}, {64, 1, 1}, {128, 1, 1}, {256, 1, 1}, {512, 1, 1}, //
    };
    std::cout << "Testing the following local sizes for 3d" << std::endl;
    for (auto lo : localsTBT3d)
    {
        int m = lo[0];
        int n = lo[1];
        int o = lo[2];
        std::cout << "  " << m << "," << n << "," << o << std::endl;
    }
    std::cout << "Testing the following local sizes for 2d" << std::endl;
    for (auto lo : localsTBT2d)
    {
        int m = lo[0];
        int n = lo[1];
        std::cout << "  " << m << "," << n << std::endl;
    }
    std::cout << "Testing the following local sizes for 1d" << std::endl;
    for (auto lo : localsTBT1d)
    {
        int m = lo[0];
        std::cout << "  " << m << std::endl;
    }

    int ghosts = 1;
    bool return_residual = true;

    std::vector<Result> minTimes;
    runJacobiBench(gridsTBT, localsTBT2d, minTimes, ghosts, return_residual, "jacobi_iter_27point_varying_stencil", 2,
                   CLI_ARGS::nu1, 1, 0.8, false);
    runJacobiBench(gridsTBT, localsTBT3d, minTimes, ghosts, return_residual, "jacobi_iter_27point_varying_stencil_3d", 3,
                   CLI_ARGS::nu1, 1, 0.8, false);

    std::stringstream ss;
    ss << "name;m;n;o;locm;locn;loco;minTimeInMs" << std::endl;
    for (auto r : minTimes)
    {
        ss << r.name << ";" << r.m << ";" << r.n << ";" << r.o << ";" << //
            r.locm << ";" << r.locn << ";" << r.loco << ";" << std::setprecision(17) << r.minTime << std::endl;
    }
    std::string output = ss.str();
    std::replace(output.begin(), output.end(), '.', ',');
    std::cout << output;
}

TEST_CASE("exec_params_jacobi_check_results")
{

    std::vector<std::vector<int>> gridsTBT = {{32, 32, 32}};
    std::vector<std::vector<int>> localsTBT3d = {
        {1, 1, 32} //
    };
    std::vector<std::vector<int>> localsTBT2d = {
        {1, 32, 1} //
    };
    std::vector<std::vector<int>> localsTBT1d = {
        {32, 1, 1} //
    };

    int ghosts = 1;
    bool return_residual = true;

    std::vector<Result> minTimes;
    std::cout << "Running jacobi_iter_27point_varying_stencil ..." << std::endl;
    auto v1 = runJacobiBench(gridsTBT, localsTBT2d, minTimes, ghosts, return_residual, "jacobi_iter_27point_varying_stencil", 2,
                             CLI_ARGS::nu1, 1, 0.8, true);
    std::cout << "Running jacobi_iter_27point_varying_stencil_3d ..." << std::endl;
    auto v2 = runJacobiBench(gridsTBT, localsTBT3d, minTimes, ghosts, return_residual, "jacobi_iter_27point_varying_stencil_3d", 3,
                             CLI_ARGS::nu1, 1, 0.8, true);

    if (!v1->isEqual(*v2))
        std::cerr << "solution not ok for kernel jacobi_iter_27point_varying_stencil_3d" << std::endl;
    else
        std::cerr << "solution ok for kernel jacobi_iter_27point_varying_stencil_3d" << std::endl;
}

/**
 * @brief Benchmarks different grid sizes and work-group sizes for a given residual kernel.
 *
 * @param gridsTBT grids to be tested
 * @param localsTBT wg sizes to be tested
 * @param benchname name of the benchmark
 * @param minTimes array of Result, minimum timings of this benchmark will be appended
 * @param kernelName name of the kernel in kernel_optimizations.cl file
 * @param kernelDim dimensions of the kernel
 */
void runResidualBench(std::vector<std::vector<int>> gridsTBT, std::vector<std::vector<int>> localsTBT,
                      std::vector<Result>& minTimes, int ghosts, bool return_residual,
                      std::string kernelName, int kernelDim)
{

    ankerl::nanobench::Bench b;
    b.timeUnit(1ms, "ms")
        .minEpochTime(100ms);

    for (auto gr : gridsTBT)
        for (auto lo : localsTBT)
        {
            int m = gr[0];
            int n = gr[1];
            int o = gr[2];

            double hm = 1.0 / (double)m;
            double hn = 1.0 / (double)n;
            double ho = 1.0 / (double)o;

            auto v = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            auto r = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            mgcl::VaryingStencil sv(m, n, o, 3, 2, 2, 2);
            // int mgh = m + 2 * ghosts;
            // int ngh = n + 2 * ghosts;
            // int ogh = o + 2 * ghosts;

            mgcl::Problem problem(m, n, o, f, v);
            problem.setUseOpencl(true);
            problem.setSilent(true);
            problem.setKernelFile(getKernelOptimizationsFilePath());
            problem.getOpenCLHelper().init();
            mgcl::OpenCLHelper& oclh = problem.getOpenCLHelper();
            cl_context context = oclh.getContext();
            cl_command_queue commands = oclh.getCommands();
            cl_program program = oclh.getProgram();

            mgcl::CuboidGpu dVIn_cuboid(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, *v);
            mgcl::CuboidGpu dF_cuboid(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, *f);
            mgcl::CuboidGpu dR_cuboid(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, *r);
            mgcl::VaryingStencilGpu dsv = mgcl::VaryingStencilGpu(m, n, o, 3, 2, context, commands);
            dsv.fill(sv, commands, true);

            int moff = 0;
            int noff = 0;
            int ooff = 0;

            int err;
            int mgh = m + 2 * ghosts;
            int ngh = n + 2 * ghosts;
            int ogh = o + 2 * ghosts;
            double res = 0.0;

            double h2 = hm * hm;
            double h2inv = 1.0 / h2; // divisor of the stencil, inverted to use * instead of / in kernel

            CAPTURE(ghosts, moff, noff, ooff);
            // check if off is too small (i.e. start < 0)
            // TODO refactor to use GPUCuboid and check against v.getGhosts
            if (moff <= -ghosts || noff <= -ghosts || ooff <= -ghosts)
                throw "moff, noff and ooff must not be <= -ghosts";

            // check if off is too large (i.e. start > end)
            if (moff * 2 >= m || noff * 2 >= n || ooff * 2 >= o)
                throw "2*moff, 2*noff and 2*ooff must not be >= m, n or o";

            std::string name = std::string(kernelName)
                                   .append("_")
                                   .append(std::to_string(m))
                                   .append("x")
                                   .append(std::to_string(n))
                                   .append("x")
                                   .append(std::to_string(o))
                                   .append("_")
                                   .append(std::to_string(lo[0]))
                                   .append("x")
                                   .append(std::to_string(lo[1]))
                                   .append("x")
                                   .append(std::to_string(lo[2]));
            b.run(name.c_str(), [&]
                  {
                      // Create the compute kernel from the program
                      const char* kernel_name = kernelName.c_str();

                      // Create the compute kernel from the program
                      cl_kernel kernel = clCreateKernel(problem.getOpenCLHelper().getProgram(), kernel_name, &err);
                      mgcl::mgclCheckError(err, "Creating kernel");

                      cl_mem dF = dF_cuboid.getBuffer();
                      cl_mem dVIn = dVIn_cuboid.getBuffer();
                      cl_mem dR = dR_cuboid.getBuffer();
                      // assign kernel arguments
                      int pos = 0;
                      auto svbuf = dsv.getBuf();
                      int svgh = dsv.getGh();
                      err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dVIn);
                      err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dF);
                      err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dR);
                      err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &svbuf);
                      err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
                      err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
                      err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
                      err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts);
                      err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svgh);
                      err |= clSetKernelArg(kernel, ++pos, sizeof(int), &moff);
                      err |= clSetKernelArg(kernel, ++pos, sizeof(int), &noff);
                      err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ooff);

                      mgcl::mgclCheckError(err, "Setting residual kernel arguments");

                      // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
                      // set 3d sizes initially
                      size_t global[3] = {static_cast<size_t>(mgh), static_cast<size_t>(ngh), static_cast<size_t>(ogh)};
                      size_t local[3] = {static_cast<size_t>(lo[0]), static_cast<size_t>(lo[1]), static_cast<size_t>(lo[2])};
                      if (kernelDim == 1)
                      {
                          global[0] = static_cast<size_t>(mgh * ngh * ogh);
                          local[0] = static_cast<size_t>(lo[0]);
                          global[1] = static_cast<size_t>(0);
                          local[1] = static_cast<size_t>(1);
                          global[2] = static_cast<size_t>(0);
                          local[2] = static_cast<size_t>(1);
                      }

                      for (int i = 0; i < 3 && i < kernelDim; i++)
                          if (global[i] % local[i] != 0)
                          {
                              // printf("padding global size %d from %ld to ", i, global[i]);
                              global[i] += local[i] - (global[i] % local[i]);
                              // printf("%ld (multiple of %ld)\n", global[i], local[i]);
                          }

                      if (kernelDim == 1)
                          err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel, 1, NULL, &global[0], &local[0], 0, NULL, NULL);
                      else if (kernelDim == 3)
                          err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel, 3, NULL, global, local, 0, NULL, NULL);
                      mgcl::mgclCheckError(err, "Enqueueing residual kernel");

                      if (problem.isPeriodic())
                      {
                          err = mgcl::MultigridEngine::updateGhosts(problem, dR_cuboid, nullptr, true);
                          mgcl::mgclCheckError(err, "Updating ghosts of r");
                      }

                      // calculate residual's 2-norm. Square elements on device and sum up on host
                      if (return_residual)
                      {
                          // calculate 2-Norm
                          mgcl::Cuboid rsq(mgh, ngh, ogh);
                          int pointer_flag = problem.getOpenCLHelper().getDeviceType() == CL_DEVICE_TYPE_GPU ? CL_MEM_COPY_HOST_PTR : CL_MEM_USE_HOST_PTR;
                          mgcl::CuboidGpu dRsquares(problem.getOpenCLHelper().getContext(), CL_MEM_WRITE_ONLY | pointer_flag, rsq);

                          // Create the compute kernel from the program
                          cl_kernel kernel_square = clCreateKernel(program, "residual_squared", &err);
                          mgcl::mgclCheckError(err, "Creating residual squared kernel");

                          pos = 0;
                          err = clSetKernelArg(kernel_square, pos, sizeof(cl_mem), &dR);
                          err |= clSetKernelArg(kernel_square, ++pos, sizeof(cl_mem), &dRsquares);
                          err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &m);
                          err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &n);
                          err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &o);
                          err |= clSetKernelArg(kernel_square, ++pos, sizeof(int), &ghosts);
                          mgcl::mgclCheckError(err, "Setting residual squared kernel arguments");

                          err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel_square, 3, NULL, global, local, 0, NULL, NULL);
                          mgcl::mgclCheckError(err, "Enqueueing residual squared kernel");

                          // sum up residual squares
                          ankerl::nanobench::doNotOptimizeAway(res = sqrt(mgcl::util::sum(dRsquares, problem.getProgram(), problem.getCommands(), true)));

                          clReleaseKernel(kernel_square);
                      }

                      clReleaseKernel(kernel); // TODO maybe clFinish before release?
                      oclh.finish();           //
                  });

            // Get minimum of all epochs in ms
            double min = 1000000;
            for (auto r : b.results())
                if (r.minimum(ankerl::nanobench::Result::Measure::elapsed) < min)
                    min = r.minimum(ankerl::nanobench::Result::Measure::elapsed) * 1000.0 /** 1000.0 * 1000.0*/;

            Result result;
            result.name = name;
            result.minTime = min;
            result.m = m;
            result.n = n;
            result.o = o;
            result.locm = lo[0];
            result.locn = lo[1];
            result.loco = lo[2];
            minTimes.push_back(result);
        }
}

/**
 * @brief Benchmarks Jacobi on the GPU.
 *
 * @param gridsTBT
 * @param localsTBT
 * @param minTimes
 * @param ghosts
 * @param return_residual
 * @param kernelName
 * @param kernelDim
 * @param maxiter
 * @param stepsPerIter
 * @param omega
 * @param checkResults If true, resuts will be checked and Jacobi is called only once.
 * @returns Result v if checkResults is true
 */
std::shared_ptr<mgcl::Cuboid> runJacobiBench(std::vector<std::vector<int>> gridsTBT, std::vector<std::vector<int>> localsTBT,
                                             std::vector<Result>& minTimes,
                                             int ghosts, bool return_residual, std::string kernelName, int kernelDim,
                                             int maxiter, int stepsPerIter, double omega, bool checkResults)
{
    ankerl::nanobench::Bench b;
    b.timeUnit(1ms, "ms")
        .minEpochTime(100ms);

    // set number of epochs and epoch iterations to 1 if results shall be checked.
    if (checkResults)
        b.epochs(1).epochIterations(1).output(nullptr);

    int store_res = 0;
    int idx_start = 0;

    for (auto gr : gridsTBT)
        for (auto lo : localsTBT)
        {
            int m = gr[0];
            int n = gr[1];
            int o = gr[2];

            double hm = 1.0 / (double)m;
            double hn = 1.0 / (double)n;
            double ho = 1.0 / (double)o;

            auto v = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            auto r = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            auto solution = std::make_shared<mgcl::Cuboid>(m, n, o, ghosts, ghosts, ghosts);
            create4thOrderProblem(*v, *f, *solution);
            mgcl::VaryingStencil sv(m, n, o, 3, 2, 2, 2);
            sv_fill_27p_laplace(sv);
            // int mgh = m + 2 * ghosts;
            // int ngh = n + 2 * ghosts;
            // int ogh = o + 2 * ghosts;

            mgcl::Problem problem(m, n, o, f, v);
            problem.setUseOpencl(true);
            problem.setSilent(true);
            problem.setKernelFile(getKernelOptimizationsFilePath());
            problem.getOpenCLHelper().init();
            mgcl::OpenCLHelper& oclh = problem.getOpenCLHelper();
            cl_context context = oclh.getContext();
            cl_command_queue commands = oclh.getCommands();
            cl_program program = oclh.getProgram();

            mgcl::CuboidGpu dVIn_cuboid(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, *v);
            mgcl::CuboidGpu dVOut_cuboid(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, *v);
            mgcl::CuboidGpu dF_cuboid(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, *f);
            mgcl::CuboidGpu dR_cuboid(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, *r);
            mgcl::VaryingStencilGpu dsv = mgcl::VaryingStencilGpu(m, n, o, 3, 2, context, commands);
            dsv.fill(sv, commands, true);

            int moff = 0;
            int noff = 0;
            int ooff = 0;

            int err;
            int mgh = m + 2 * ghosts;
            int ngh = n + 2 * ghosts;
            int ogh = o + 2 * ghosts;
            double res = 0.0;

            double h2 = hm * hm;
            double h2inv = 1.0 / h2; // divisor of the stencil, inverted to use * instead of / in kernel

            CAPTURE(ghosts, moff, noff, ooff);
            std::string name = std::string(kernelName)
                                   .append("_")
                                   .append(std::to_string(m))
                                   .append("x")
                                   .append(std::to_string(n))
                                   .append("x")
                                   .append(std::to_string(o))
                                   .append("_")
                                   .append(std::to_string(lo[0]))
                                   .append("x")
                                   .append(std::to_string(lo[1]))
                                   .append("x")
                                   .append(std::to_string(lo[2]));

            b.run(name.c_str(), [&]
                  {
                      // decrease stepsPerIter if it's less than maxIter
                      if (maxiter < stepsPerIter)
                          stepsPerIter = maxiter;

                      // Ghosts only need to be updated in the periodic case, so set stepsPerIter = 1 for non-periodic.
                      // TODO adjust for MPI
                      if (!problem.isPeriodic())
                          stepsPerIter = 1;

                      // Check if amount of ghost cells is large enough
                      if (ghosts < stepsPerIter)
                      {
                          throw "#ghosts must be >= stepsPerIter!";
                      }

                      // Create the compute kernel from the program
                      // const char* kernel_name;
                      // kernel_name = "jacobi_iter_27point_varying_stencil";

                      cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &err);
                      mgcl::mgclCheckError(err, "Creating kernel");

                      cl_mem dVIn = dVIn_cuboid.getBuffer();
                      cl_mem dVOut = dVOut_cuboid.getBuffer();
                      cl_mem dF = dF_cuboid.getBuffer();
                      cl_mem dR = dR_cuboid.getBuffer();

                      // assign kernel arguments
                      int pos = 0;
                      auto svbuf = dsv.getBuf();
                      int svgh = dsv.getGh();
                      err = clSetKernelArg(kernel, pos, sizeof(cl_mem), &dVIn);
                      err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dVOut);
                      err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dF);
                      err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &dR);
                      err |= clSetKernelArg(kernel, ++pos, sizeof(cl_mem), &svbuf);
                      err |= clSetKernelArg(kernel, ++pos, sizeof(double), &omega);
                      err |= clSetKernelArg(kernel, ++pos, sizeof(int), &mgh);
                      err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ngh);
                      err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ogh);
                      err |= clSetKernelArg(kernel, ++pos, sizeof(int), &ghosts);
                      err |= clSetKernelArg(kernel, ++pos, sizeof(int), &svgh);
                      err |= clSetKernelArg(kernel, ++pos, sizeof(int), &idx_start);
                      err |= clSetKernelArg(kernel, ++pos, sizeof(int), &store_res);
                      mgcl::mgclCheckError(err, "Setting kernel arguments");

                      // one work-item per cell (including ghost cells). Pad global sizes to fit to local sizes
                      // set 3d sizes initially
                      size_t global[3] = {static_cast<size_t>(mgh), static_cast<size_t>(ngh), static_cast<size_t>(ogh)};
                      size_t local[3] = {static_cast<size_t>(lo[0]), static_cast<size_t>(lo[1]), static_cast<size_t>(lo[2])};
                      if (kernelDim == 1)
                      {
                          global[0] = static_cast<size_t>(mgh * ngh * ogh);
                          local[0] = static_cast<size_t>(lo[0]);
                          global[1] = static_cast<size_t>(0);
                          local[1] = static_cast<size_t>(1);
                          global[2] = static_cast<size_t>(0);
                          local[2] = static_cast<size_t>(1);
                      }
                      else if (kernelDim == 2)
                      {
                          global[0] = static_cast<size_t>(mgh);
                          global[1] = static_cast<size_t>(ngh);
                          global[2] = static_cast<size_t>(0);
                          local[0] = static_cast<size_t>(lo[0]);
                          local[1] = static_cast<size_t>(lo[1]);
                          local[2] = static_cast<size_t>(1);
                      }

                      for (int i = 0; i < 3 && i < kernelDim; i++)
                          if (global[i] % local[i] != 0)
                          {
                              // printf("padding global size %d from %ld to ", i, global[i]);
                              global[i] += local[i] - (global[i] % local[i]);
                              // printf("%ld (multiple of %ld)\n", global[i], local[i]);
                          }

                      int globalIter = 0;
                      while (globalIter < maxiter)
                      {
                          // Update ghosts of current input v
                          if (globalIter % 2 == 1)
                          {
                              err = mgcl::MultigridEngine::updateGhosts(problem, dVOut_cuboid,
                                                                        nullptr, true);
                              mgcl::mgclCheckError(err, "Updating ghosts");
                          }
                          else
                          {
                              err = mgcl::MultigridEngine::updateGhosts(problem, dVIn_cuboid,
                                                                        nullptr, true);
                              mgcl::mgclCheckError(err, "Updating ghosts");
                          }

                          // if stepsPerIter > 1, multiple iterations can be done without updating ghosts in-between
                          for (int innerIter = 0; innerIter < stepsPerIter && globalIter < maxiter; innerIter++, globalIter++)
                          {
                              // damped/weighted iteration formula: u_(m+1) = u_(m) + omega * D^-1 * r_(m)

                              // switch arguments dVIn -> dVOut to use latest values in next iteration
                              if (globalIter % 2 == 1)
                              {
                                  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dVIn);
                                  err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &dVOut);
                                  mgcl::mgclCheckError(err, "Setting kernel arguments");
                              }
                              else
                              {
                                  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dVIn);
                                  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dVOut);
                                  mgcl::mgclCheckError(err, "Setting kernel arguments");
                              }

                              // set flag to store residual in last iteration
                              if (globalIter == maxiter - 1)
                              {
                                  store_res = 1;
                                  err = clSetKernelArg(kernel, pos, sizeof(int), &store_res);
                                  mgcl::mgclCheckError(err, "Setting kernel arguments");
                              }

                              // recalculate and set idx_start
                              idx_start = ghosts - ((stepsPerIter - innerIter) - 1);
                              err = clSetKernelArg(kernel, pos - 1, sizeof(int), &idx_start);
                              mgcl::mgclCheckError(err, "Setting kernel arguments");

                              if (kernelDim == 2)
                                  err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel, 2, NULL, global, local, 0, NULL, NULL);
                              else if (kernelDim == 3)
                                  err = clEnqueueNDRangeKernel(problem.getOpenCLHelper().getCommands(), kernel, 3, NULL, global, local, 0, NULL, NULL);
                              mgcl::mgclCheckError(err, "Enqueueing kernel");
                          }
                      }

                      // if (store_res)
                      // {
                      //     // TODO check for mpi
                      //     err = mgcl::MultigridEngine::updateGhosts(problem, dR_cuboid, nullptr, true);
                      //     mgcl::mgclCheckError(err, "Updating ghosts of dR");
                      // }

                      // copy result into dVIn if needed
                      if (maxiter % 2 == 1)
                          dVOut_cuboid.copyTo(problem.getOpenCLHelper().getCommands(), dVIn_cuboid);

                      // Update ghosts of dVIn
                      err = mgcl::MultigridEngine::updateGhosts(problem, dVIn_cuboid, nullptr, true);
                      mgcl::mgclCheckError(err, "Updating ghosts");

                      // calculate residual and its norm
                      // if (return_residual)
                      // {
                      //     // update residual to use current approximation v
                      //     res = mgcl::MultigridEngine::residual(problem, level, true);
                      // }

                      clReleaseKernel(kernel); //
                      oclh.finish();           //
                  });

            if (checkResults)
            {
                dVIn_cuboid.read(commands, v.get(), true);
                return v;
            }

            // Get minimum of all epochs in ms
            double min = 1000000;
            for (auto r : b.results())
                if (r.minimum(ankerl::nanobench::Result::Measure::elapsed) < min)
                    min = r.minimum(ankerl::nanobench::Result::Measure::elapsed) * 1000.0 /** 1000.0 * 1000.0*/;

            Result result;
            result.name = name;
            result.minTime = min;
            result.m = m;
            result.n = n;
            result.o = o;
            result.locm = lo[0];
            result.locn = lo[1];
            result.loco = lo[2];
            minTimes.push_back(result);
        }

    return nullptr;
}

// fills v, f and solution with a periodic 4th order problem. Dimensions and ghosts of all 3 inputs must match.
void create4thOrderProblem(mgcl::Cuboid& v, mgcl::Cuboid& f, mgcl::Cuboid& solution)
{
    if (v.getM() != f.getM() || v.getN() != f.getN() || v.getO() != f.getO() || v.getM() != solution.getM() || v.getN() != solution.getN() || v.getO() != solution.getO())
        throw "Dimensions must match.";

    if (v.getGhostsM() != f.getGhostsM() || v.getGhostsN() != f.getGhostsN() || v.getGhostsO() != f.getGhostsO() || v.getGhostsM() != solution.getGhostsM() || v.getGhostsN() != solution.getGhostsN() || v.getGhostsO() != solution.getGhostsO())
        throw "Ghosts must match.";

    double hm = 1.0 / v.getM();
    double hn = 1.0 / v.getN();
    double ho = 1.0 / v.getO();

    for (int i = v.getGhostsM(); i < v.getM() + v.getGhostsM(); i++)
        for (int j = v.getGhostsN(); j < v.getN() + v.getGhostsN(); j++)
            for (int k = v.getGhostsO(); k < v.getO() + v.getGhostsO(); k++)
            {
                double zs = i * hm;
                double ys = j * hn;
                double xs = k * ho;
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
                v[i][j][k] = 0;
                solution[i][j][k] = 1000000 * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) * (xs * (xs - 1)) *
                                    (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) * (ys * (ys - 1)) *
                                    (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1)) * (zs * (zs - 1));
                f[i][j][k] =
                    -1000000 *
                    (12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_2 + 12 * xs4 * ys4 * zs4 * xsm1_4 * ysm1_2 * zsm1_4 +
                     12 * xs4 * ys4 * zs4 * xsm1_2 * ysm1_4 * zsm1_4 + 32 * xs4 * ys4 * zs3 * xsm1_4 * ysm1_4 * zsm1_3 +
                     12 * xs4 * ys4 * zs2 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs4 * ys3 * zs4 * xsm1_4 * ysm1_3 * zsm1_4 +
                     12 * xs4 * ys2 * zs4 * xsm1_4 * ysm1_4 * zsm1_4 + 32 * xs3 * ys4 * zs4 * xsm1_3 * ysm1_4 * zsm1_4 +
                     12 * xs2 * ys4 * zs4 * xsm1_4 * ysm1_4 * zsm1_4);
            }
}

void sv_fill_27p_laplace(mgcl::VaryingStencil& s)
{
    double hm = 1.0 / s.getDim1();
    double factor = 1.0 / (30.0 * hm * hm);

    // Fill with 27-point Laplace
    for (int i = 0; i < s.getDim1gh(); i++)
        for (int j = 0; j < s.getDim2gh(); j++)
            for (int k = 0; k < s.getDim3gh(); k++)
            {
                // 27-point Laplace
                // center
                s[i][j][k][1][1][1] = factor * 128.0;

                // adjacent to center
                s[i][j][k][0][1][1] = factor * -1.0;
                s[i][j][k][1][0][1] = factor * -1.0;
                s[i][j][k][1][1][0] = factor * -1.0;
                s[i][j][k][1][1][2] = factor * -1.0;
                s[i][j][k][1][2][1] = factor * -1.0;
                s[i][j][k][2][1][1] = factor * -1.0;

                // diagonally adjacent to center
                s[i][j][k][1][0][0] = -3.0 * factor;
                s[i][j][k][1][0][2] = -3.0 * factor;
                s[i][j][k][1][2][0] = -3.0 * factor;
                s[i][j][k][1][2][2] = -3.0 * factor;
                s[i][j][k][0][1][0] = -3.0 * factor;
                s[i][j][k][0][1][2] = -3.0 * factor;
                s[i][j][k][2][1][0] = -3.0 * factor;
                s[i][j][k][2][1][2] = -3.0 * factor;
                s[i][j][k][0][0][1] = -3.0 * factor;
                s[i][j][k][0][2][1] = -3.0 * factor;
                s[i][j][k][2][0][1] = -3.0 * factor;
                s[i][j][k][2][2][1] = -3.0 * factor;

                // corners
                s[i][j][k][0][0][0] = -1.0 * factor;
                s[i][j][k][0][0][2] = -1.0 * factor;
                s[i][j][k][0][2][0] = -1.0 * factor;
                s[i][j][k][0][2][2] = -1.0 * factor;
                s[i][j][k][2][0][0] = -1.0 * factor;
                s[i][j][k][2][0][2] = -1.0 * factor;
                s[i][j][k][2][2][0] = -1.0 * factor;
                s[i][j][k][2][2][2] = -1.0 * factor;
            }
}
