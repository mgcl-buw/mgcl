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

#include <array>
#include <map>
#include <string>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace mgcl
{
    /**
     * @brief Holds results of one profiled kernel call.
     */
    struct ProfilingMeasurement
    {
        cl_ulong elapsed; // Elapsed time from kernel begin to end in ns
        cl_ulong inQueue; // Time spent from enqueuing to kernel start in ns
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

        void addMeasurement(cl_command_queue commands, cl_event ev,
                            std::string kernelName,
                            std::array<size_t, 3> global, std::array<size_t, 3> local);

        void printBestTimingsPerKernel();

        friend std::ostream& operator<<(std::ostream& os, const ProfilingData& pd);
    };
}

#endif // __PROFILING_DATA_H__