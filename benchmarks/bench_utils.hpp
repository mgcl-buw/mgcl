/**
 * @file bench_utils.hpp
 * @brief Utility functions for benchmarks, for example, getting the minimum time of a measurement.
 * @date 2024-03-04
 *
 */
#ifndef __BENCH_UTILS_H__
#define __BENCH_UTILS_H__

#include <nanobench.h>

/**
 * @brief Returns minimum time of measurements for all epochs.
 *
 * @param b Benchmark object the tests ran on.
 * @param startidx Starting index of measurement to consider. Useful, when e.g. multiple test are run on the
 * same Bench object.
 */
inline double benchMinTime(ankerl::nanobench::Bench& b, int startidx = 0)
{
    // Get minimum of all epochs in ms
    double minTime = std::numeric_limits<double>::max();
    for (int i = startidx; i < b.results().size(); i++)
    {
        auto& r = b.results()[i];
        if (r.minimum(ankerl::nanobench::Result::Measure::elapsed) < minTime)
            minTime = r.minimum(ankerl::nanobench::Result::Measure::elapsed) * 1000.0;
    }
    return minTime;
}

#endif // __BENCH_UTILS_H__