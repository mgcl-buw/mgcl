#ifndef __PROFILING_DATA_H__
#define __PROFILING_DATA_H__

/**
 * @file profiling_data.hpp
 * @author Simon Hoffmann (shoffmann@uni-wuppertal.de)
 * @brief This file contains the class ProfilingData, which gets filled with kernel
 * profiling information, if profiling is enabled in Problem. One measurement is stored
 * inside the struct ProfilingMeasurement.
 * @version 0.1
 * @date 2024-02-21
 */

#include <map>
#include <string>
#include <vector>

namespace mgcl
{
    /**
     * @brief Holds results of one profiled kernel call.
     */
    struct ProfilingMeasurement
    {
        unsigned long elapsed; // Elapsed time from kernel begin to end in ns
        size_t work_items[3];
        size_t work_group[3];
    };

    using MeasurementsPerKernel = std::map<std::string, std::vector<ProfilingMeasurement>>;

    /**
     * @brief Collection of ProfilingMeasurements. New measurements can be added by getting the underlying
     * map via getMeasurements() and using the regular map syntax.
     *
     */
    class ProfilingData
    {
    private:
        MeasurementsPerKernel measurements;

    public:
        MeasurementsPerKernel& getMeasurements()
        {
            return measurements;
        }
    };
}

#endif // __PROFILING_DATA_H__