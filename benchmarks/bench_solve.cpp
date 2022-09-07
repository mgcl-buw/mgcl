#define ANKERL_NANOBENCH_IMPLEMENT
#include "nanobench.h"

#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"

#include <chrono>
#include <vector>
using namespace std::chrono_literals;

#include "../cuboid.hpp"
#include "../problem.hpp"
#include "../test/test_utility.hpp"

namespace mgcl_test
{
    /**
     * @brief Mustache template for generating a line plot of sequential vs. opencl. Heavily depends on the name of the
     * benchmark test, e.g. for N it tries to match 'N = \d+' and for differentiating between seq and ocl it tries to
     * match "sequential" or "opencl" respectively.
     *
     */
    std::string htmlLine = R"DELIM(<html>
<head>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <div id="myDivAbs"></div>
    <div id="myDivRel"></div>
    <script>
        var seq = {
            x: [],
            y: [],
            type: 'scatter',
            name: 'Sequential'
        };
        var ocl = {
            x: [],
            y: [],
            type: 'scatter',
            name: 'OpenCL'
        };

        // group seq and ocl runs
        let name;
        let N;
        let isSeq;
        let isOcl;
        let allNs = new Set();

        {{#result}}
        name = '{{name}}'.toLowerCase();
        N = null;
        if (name.match(/n = (\d*)/)) {
            N = name.match(/n = (\d*)/)[1];
        }

        isSeq = !!name.match(/.*sequential.*/);
        isOcl = !!name.match(/.*opencl.*/);

        if (isSeq && N) {
            allNs.add(N);
            seq.x.push(N);
            seq.y.push({{minimum(elapsed)}});
        }
        else if (isOcl && N) {
            allNs.add(N);
            ocl.x.push(N);
            ocl.y.push({{minimum(elapsed)}});
        }
        {{/result}}

        // plot absolute
        var data = [seq, ocl];
        var title = 'Absolute';
        var layout = { title: { text: title }, showlegend: true, 
            xaxis: { title: 'grid size', tickvals: [...allNs] },
            yaxis: { title: 'time per unit', rangemode: 'tozero', autorange: true, type: 'log' }
        };
        Plotly.newPlot('myDivAbs', data, layout, {responsive: true});

        // plot relative
        var dataRel = {
            x: [],
            y: [],
            type: 'scatter',
            name: 'Sequential / OpenCL'
        };

        for (let i = 0; i < seq.x.length; i++) {
            dataRel.x.push(seq.x[i]);
            dataRel.y.push(seq.y[i] / ocl.y[i]);
        }

        title = 'Relative';
        layout = { title: { text: title }, showlegend: true, 
            xaxis: { title: 'grid size', tickvals: [...allNs] },
            yaxis: { title: 'Sequential / OpenCL', rangemode: 'tozero', autorange: true }
        };
        Plotly.newPlot('myDivRel', [dataRel], layout);
    </script>
</body>
</html>)DELIM";
}

TEST_CASE("mgcl benchmarks console", "[!benchmark]")
{
    int N = GENERATE(16, 32, 64, 128);
    // int N = 16;
    int m = N;
    int n = N;
    int o = N;

    ankerl::nanobench::Bench b;
    b.timeUnit(1ms, "ms")
        .warmup(3)
        .minEpochTime(100ms)
        .relative(true);

    SECTION(std::string("N = ").append(std::to_string(N)).c_str())
    {
        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
            v->fillRandom(0, 10);
            f->fillRandom(0, 10);

            mgcl::Problem p(m, n, o, f, v);
            p.setSilent(true);
            p.init();

            b.run(std::string("sequential random values, N = ").append(std::to_string(N)).c_str(), [&]
                  { p.solveSeq(); });
        }

        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
            v->fillRandom(0, 10);
            f->fillRandom(0, 10);

            mgcl::Problem p(m, n, o, f, v);
            p.setUseOpencl(true);
            p.setDeviceType(CL_DEVICE_TYPE_GPU);
            p.setSilent(true);

            mgcl_test::TestUtility tu;
            if (tu.deviceAvailable("Quadro", p.getDeviceType()))
                p.setDeviceName("Quadro");

            p.init();
            b.run(std::string("opencl random values, N = ").append(std::to_string(N)).c_str(), [&]
                  { p.solve(); });
        }
    }
}

TEST_CASE("mgcl benchmarks lineplot", "[!benchmark]")
{
    ankerl::nanobench::Bench b;
    b.timeUnit(1ms, "ms")
        .warmup(3)
        .minEpochTime(100ms);

    std::vector<int> grids{16, 32, 64, 128};
    for (auto N : grids)
    {
        int m = N;
        int n = N;
        int o = N;

        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
            v->fillRandom(0, 10);
            f->fillRandom(0, 10);

            mgcl::Problem p(m, n, o, f, v);
            p.setSilent(true);
            p.init();

            b.run(std::string("sequential random values, N = ").append(std::to_string(N)).c_str(), [&]
                  { p.solveSeq(); });
        }

        {
            auto v = std::make_shared<mgcl::Cuboid>(m, n, o);
            auto f = std::make_shared<mgcl::Cuboid>(m, n, o);
            v->fillRandom(0, 10);
            f->fillRandom(0, 10);

            mgcl::Problem p(m, n, o, f, v);
            p.setUseOpencl(true);
            p.setDeviceType(CL_DEVICE_TYPE_GPU);
            p.setSilent(true);

            mgcl_test::TestUtility tu;
            if (tu.deviceAvailable("Quadro", p.getDeviceType()))
                p.setDeviceName("Quadro");

            p.init();
            b.run(std::string("opencl random values, N = ").append(std::to_string(N)).c_str(), [&]
                  { p.solve(); });
        }
    }

    std::ofstream renderOut(std::string("benchresults_lineplot").append(".html"));
    ankerl::nanobench::render(mgcl_test::htmlLine, b, renderOut);
}
