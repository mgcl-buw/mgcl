#ifndef __BENCH_RENDER_TEMPLATES_H__
#define __BENCH_RENDER_TEMPLATES_H__

#include <string>

namespace mgcl_test
{
    /**
     * @brief Mustache template for generating a line plot of sequential vs. opencl. Heavily depends on the name of the
     * benchmark test, e.g. for N it tries to match 'N = \d+' and for differentiating between seq and ocl it tries to
     * match "sequential" or "opencl" respectively.
     *
     */
    inline std::string htmlLineComparingN()
    {
        return R"DELIM(<html>
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

    /**
     * @brief Mustache template for generating a line plot of sequential vs. opencl for varying vcycle iteration counts.
     * Heavily depends on the name of the benchmark test, e.g. for N it tries to match 'N = \d+' and for
     * differentiating between seq and ocl it tries to match "sequential" or "opencl" respectively.
     *
     */
    inline std::string htmlLineComparingVcycleIters()
    {
        return R"DELIM(<html>
<head>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/regression/2.0.1/regression.min.js"></script>
</head>
<body>
    <div id="myDivAbs"></div>
    <div id="divFitSeq"></div>
    <div id="divFitOcl"></div>
    <div id="myDivRel"></div>
    <div id="divFitRel"></div>
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
        let N = null;
        let iter = null;
        let isSeq;
        let isOcl;
        let allNs = new Set();
        let allIters = new Set();

        {{#result}}
        name = '{{name}}'.toLowerCase();
        if (name.match(/n = (\d+)/)) {
            N = name.match(/n = (\d+)/)[1];
        }

        if (name.match(/vcycle iters = (\d+)/)) {
            iter = name.match(/vcycle iters = (\d+)/)[1];
        }

        isSeq = !!name.match(/.*seq.*/);
        isOcl = !!name.match(/.*ocl.*/);

        if (isSeq && N && iter) {
            allNs.add(parseInt(N));
            allIters.add(parseInt(iter));
            seq.x.push(parseInt(iter));
            seq.y.push({{minimum(elapsed)}});
        }
        else if (isOcl && N && iter) {
            allNs.add(parseInt(N));
            allIters.add(parseInt(iter));
            ocl.x.push(parseInt(iter));
            ocl.y.push({{minimum(elapsed)}});
        }
        {{/result}}

        // fit absolute curves
        const seqPairs = seq.x.reduce((prev, curr, idx) => [...prev, [seq.x[idx], seq.y[idx]]], []);
        const oclPairs = ocl.x.reduce((prev, curr, idx) => [...prev, [ocl.x[idx], ocl.y[idx]]], []);
        const seqRegression = regression.linear(seqPairs, { order: 1, precision: 9 });
        const oclRegression = regression.linear(oclPairs, { order: 1, precision: 9 });
        const seqRegLine = {
            x: [seq.x[0], seq.x[seq.x.length - 1]],
            y: [seqRegression.equation[0] * seq.x[0] + seqRegression.equation[1], seqRegression.equation[0] * seq.x[seq.x.length - 1] + seqRegression.equation[1]],
            mode: 'lines', name: 'SequentialFit', line: { dash: 'dot' }
        };
        const oclRegLine = {
            x: [ocl.x[0], ocl.x[ocl.x.length - 1]],
            y: [oclRegression.equation[0] * ocl.x[0] + oclRegression.equation[1], oclRegression.equation[0] * ocl.x[ocl.x.length - 1] + oclRegression.equation[1]],
            mode: 'lines', name: 'OpenCLFit', line: { dash: 'dot' }
        };

        // plot absolute
        var data = [seq, ocl, seqRegLine, oclRegLine];
        var title = 'Absolute, N = ' + N;
        var layout = {
            title: { text: title }, showlegend: true,
            xaxis: { title: 'grid size', tickvals: [...allIters] },
            yaxis: { title: 'time per unit', rangemode: 'tozero', autorange: true }
        };
        Plotly.newPlot('myDivAbs', data, layout, { responsive: true });

        document.getElementById("divFitSeq").innerText = "Sequential Fit: y = " + seqRegression.equation[0] + " * x + " + seqRegression.equation[1];
        document.getElementById("divFitOcl").innerText = "    OpenCL Fit: y = " + oclRegression.equation[0] + " * x + " + oclRegression.equation[1];

        // fit relative curves
        const relPairs = seq.x.reduce((prev, curr, idx) => [...prev, [seq.x[idx], seq.y[idx] / ocl.y[idx]]], []);
        const relRegression = regression.linear(relPairs, { order: 1, precision: 9 });
        const relRegLine = {
            x: [seq.x[0], seq.x[seq.x.length - 1]],
            y: [relRegression.equation[0] * seq.x[0] + relRegression.equation[1], relRegression.equation[0] * seq.x[seq.x.length - 1] + relRegression.equation[1]],
            mode: 'lines', name: 'RelativeFit', line: { dash: 'dot' }
        };

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

        title = 'Relative, N = ' + N;
        layout = {
            title: { text: title }, showlegend: true,
            xaxis: { title: 'grid size', tickvals: [...allIters] },
            yaxis: { title: 'Sequential / OpenCL', rangemode: 'tozero', autorange: true }
        };
        Plotly.newPlot('myDivRel', [dataRel, relRegLine], layout);

        document.getElementById("divFitRel").innerText = "Relative Fit: y = " + relRegression.equation[0] + " * x + " + relRegression.equation[1];
    </script>
</body>
</html>)DELIM";
    }
}
#endif // __BENCH_RENDER_TEMPLATES_H__