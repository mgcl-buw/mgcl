// #include "nanobench.h"

// #include "catch2/catch_test_macros.hpp"
// #include "catch2/generators/catch_generators.hpp"

// #include <chrono>
// #include <fstream>
// #include <iostream>
// #include <vector>
// using namespace std::chrono_literals;

// #include "../src/cuboid.hpp"
// #include "../src/problem.hpp"
// #include "../src/stencil.hpp"
// #include "../test/test_utility.hpp"
// #include "bench_render_templates.hpp"

// TEST_CASE("bench_stencil apply vs direct", "[!benchmark][stencil][console][applyvsdirect]")
// {
//     int N = GENERATE(16, 32, 64, 128);
//     // int N = 16;
//     int m = N;
//     int n = N;
//     int o = N;
//     double h = 1.0 / (double)N;

//     ankerl::nanobench::Bench b;
//     b.timeUnit(1us, "us")
//         .epochs(11)
//         // .epochIterations(1)
//         .minEpochTime(100ms)
//         .relative(true);

//     mgcl::Cuboid v(m, n, o);
//     v.fillRandom();

//     // int i = 1;
//     // int j = 1;
//     // int k = 1;

//     SECTION("Laplace7p")
//     {
//         mgcl::Stencil *stencil = new mgcl::StencilLaplace7p(h);
//         double stencilFactor = 1.0 / (h * h);

//         b.run(std::string("apply 7p, N = ").append(std::to_string(N)).c_str(),
//               [&]
//               {
//                   for (int i = 1; i < m - 1; i++)
//                       for (int j = 1; j < n - 1; j++)
//                           for (int k = 1; k < o - 1; k++)
//                           {
//                               ankerl::nanobench::doNotOptimizeAway(stencil->apply(v, i, j, k));
//                           }
//               });

//         b.run(std::string("direct 7p, N = ").append(std::to_string(N)).c_str(),
//               [&]
//               {
//                   for (int i = 1; i < m - 1; i++)
//                       for (int j = 1; j < n - 1; j++)
//                           for (int k = 1; k < o - 1; k++)
//                           {
//                               // clang-format off
//                 double sum = (6.0 * v[i][j][k]
//                     - v[i][j][k - 1] - v[i][j][k + 1]
//                     - v[i][j - 1][k] - v[i][j + 1][k]
//                     - v[i - 1][j][k] - v[i + 1][j][k]
//                     ) * stencilFactor;
//                 ankerl::nanobench::doNotOptimizeAway(sum);
//                               // clang-format on
//                           }
//               });

//         delete stencil;
//     }

//     SECTION("Laplace19p")
//     {
//         mgcl::Stencil *stencil = new mgcl::StencilLaplace19p(h);
//         double stencilFactor = 1.0 / (6.0 * h * h);

//         b.run(std::string("apply 19p, N = ").append(std::to_string(N)).c_str(),
//               [&]
//               {
//                   for (int i = 1; i < m - 1; i++)
//                       for (int j = 1; j < n - 1; j++)
//                           for (int k = 1; k < o - 1; k++)
//                           {
//                               ankerl::nanobench::doNotOptimizeAway(stencil->apply(v, i, j, k));
//                           }
//               });

//         b.run(std::string("direct 19p, N = ").append(std::to_string(N)).c_str(),
//               [&]
//               {
//                   for (int i = 1; i < m - 1; i++)
//                       for (int j = 1; j < n - 1; j++)
//                           for (int k = 1; k < o - 1; k++)
//                           {
//                               // clang-format off
//                   double sum = (24.0 * v[i][j][k]
//                         - 2.0 * v[i][j][k - 1] - 2.0 * v[i][j][k + 1]
//                         - 2.0 * v[i][j - 1][k] - 2.0 * v[i][j + 1][k]
//                         - 2.0 * v[i - 1][j][k] - 2.0 * v[i + 1][j][k]

//                         - v[i][j - 1][k - 1] - v[i][j - 1][k + 1]
//                         - v[i][j + 1][k - 1] - v[i][j + 1][k + 1]
//                         - v[i - 1][j][k - 1] - v[i - 1][j][k + 1]
//                         - v[i + 1][j][k - 1] - v[i + 1][j][k + 1]
//                         - v[i - 1][j - 1][k] - v[i - 1][j + 1][k]
//                         - v[i + 1][j - 1][k] - v[i + 1][j + 1][k]
//                             ) * stencilFactor;
//                       ankerl::nanobench::doNotOptimizeAway(sum);
//                               // clang-format on
//                           }
//               });

//         delete stencil;
//     }

//     SECTION("Laplace27p")
//     {
//         mgcl::Stencil *stencil = new mgcl::StencilLaplace27p(h);
//         double stencilFactor = 1.0 / (30.0 * h * h);

//         b.run(std::string("apply 27p, N = ").append(std::to_string(N)).c_str(),
//               [&]
//               {
//                   for (int i = 1; i < m - 1; i++)
//                       for (int j = 1; j < n - 1; j++)
//                           for (int k = 1; k < o - 1; k++)
//                           {
//                               ankerl::nanobench::doNotOptimizeAway(stencil->apply(v, i, j, k));
//                           }
//               });

//         b.run(std::string("direct 27p, N = ").append(std::to_string(N)).c_str(),
//               [&]
//               {
//                   for (int i = 1; i < m - 1; i++)
//                       for (int j = 1; j < n - 1; j++)
//                           for (int k = 1; k < o - 1; k++)
//                           {
//                               // clang-format off
//                   double sum =
//                         (128.0 * v[i][j][k]
//                         - 14.0 * v[i][j][k - 1] - 14.0 * v[i][j][k + 1]
//                         - 14.0 * v[i][j - 1][k] - 14.0 * v[i][j + 1][k]
//                         - 14.0 * v[i - 1][j][k] - 14.0 * v[i + 1][j][k]

//                         - 3.0 * v[i][j - 1][k - 1] - 3.0 * v[i][j - 1][k + 1]
//                         - 3.0 * v[i][j + 1][k - 1] - 3.0 * v[i][j + 1][k + 1]
//                         - 3.0 * v[i - 1][j][k - 1] - 3.0 * v[i - 1][j][k + 1]
//                         - 3.0 * v[i + 1][j][k - 1] - 3.0 * v[i + 1][j][k + 1]
//                         - 3.0 * v[i - 1][j - 1][k] - 3.0 * v[i - 1][j + 1][k]
//                         - 3.0 * v[i + 1][j - 1][k] - 3.0 * v[i + 1][j + 1][k]

//                         - v[i - 1][j - 1][k - 1] - v[i - 1][j - 1][k + 1]
//                         - v[i - 1][j + 1][k - 1] - v[i - 1][j + 1][k + 1]
//                         - v[i + 1][j - 1][k - 1] - v[i + 1][j - 1][k + 1]
//                         - v[i + 1][j + 1][k - 1] - v[i + 1][j + 1][k + 1]
//                         ) * stencilFactor;
//                       ankerl::nanobench::doNotOptimizeAway(sum);
//                               // clang-format on
//                           }
//               });

//         delete stencil;
//     }
//     std::cout << "---------" << std::endl;
// }

// TEST_CASE("bench_stencil types", "[!benchmark][stencil][console][types]")
// {
//     int N = GENERATE(16, 32, 64);
//     // int N = 16;
//     int m = N;
//     int n = N;
//     int o = N;
//     double h = 1.0 / (double)N;

//     ankerl::nanobench::Bench b;
//     b.timeUnit(1us, "us")
//         .epochs(11)
//         // .epochIterations(1)
//         .minEpochTime(100ms)
//         .relative(true);

//     mgcl::Cuboid v(m, n, o);
//     v.fillRandom();

//     {
//         mgcl::Stencil *stencil = new mgcl::StencilLaplace7p(h);

//         b.run(std::string("Laplace 7p, N = ").append(std::to_string(N)).c_str(),
//               [&]
//               {
//                   for (int i = 1; i < m - 1; i++)
//                       for (int j = 1; j < n - 1; j++)
//                           for (int k = 1; k < o - 1; k++)
//                           {
//                               ankerl::nanobench::doNotOptimizeAway(stencil->apply(v, i, j, k));
//                           }
//               });

//         delete stencil;
//     }

//     {
//         mgcl::Stencil *stencil = new mgcl::StencilLaplace19p(h);

//         b.run(std::string("Laplace 19p, N = ").append(std::to_string(N)).c_str(),
//               [&]
//               {
//                   for (int i = 1; i < m - 1; i++)
//                       for (int j = 1; j < n - 1; j++)
//                           for (int k = 1; k < o - 1; k++)
//                           {
//                               ankerl::nanobench::doNotOptimizeAway(stencil->apply(v, i, j, k));
//                           }
//               });

//         delete stencil;
//     }

//     {
//         mgcl::Stencil *stencil = new mgcl::StencilLaplace27p(h);

//         b.run(std::string("Laplace 27p, N = ").append(std::to_string(N)).c_str(),
//               [&]
//               {
//                   for (int i = 1; i < m - 1; i++)
//                       for (int j = 1; j < n - 1; j++)
//                           for (int k = 1; k < o - 1; k++)
//                           {
//                               ankerl::nanobench::doNotOptimizeAway(stencil->apply(v, i, j, k));
//                           }
//               });

//         delete stencil;
//     }

//     {
//         mgcl::StencilVarying7p *stencil = new mgcl::StencilVarying7p(m, n, o);
//         stencil->getStencilValues()->fillRandom();

//         b.run(std::string("Varying 7p, N = ").append(std::to_string(N)).c_str(),
//               [&]
//               {
//                   for (int i = 1; i < m - 1; i++)
//                       for (int j = 1; j < n - 1; j++)
//                           for (int k = 1; k < o - 1; k++)
//                           {
//                               ankerl::nanobench::doNotOptimizeAway(stencil->apply(v, i, j, k));
//                           }
//               });

//         delete stencil;
//     }

//     {
//         mgcl::StencilVarying19p *stencil = new mgcl::StencilVarying19p(m, n, o);
//         stencil->getStencilValues()->fillRandom();

//         b.run(std::string("Varying 19p, N = ").append(std::to_string(N)).c_str(),
//               [&]
//               {
//                   for (int i = 1; i < m - 1; i++)
//                       for (int j = 1; j < n - 1; j++)
//                           for (int k = 1; k < o - 1; k++)
//                           {
//                               ankerl::nanobench::doNotOptimizeAway(stencil->apply(v, i, j, k));
//                           }
//               });

//         delete stencil;
//     }

//     {
//         mgcl::StencilVarying27p *stencil = new mgcl::StencilVarying27p(m, n, o);
//         stencil->getStencilValues()->fillRandom();

//         b.run(std::string("Varying 27p, N = ").append(std::to_string(N)).c_str(),
//               [&]
//               {
//                   for (int i = 1; i < m - 1; i++)
//                       for (int j = 1; j < n - 1; j++)
//                           for (int k = 1; k < o - 1; k++)
//                           {
//                               ankerl::nanobench::doNotOptimizeAway(stencil->apply(v, i, j, k));
//                           }
//               });

//         delete stencil;
//     }
// }
