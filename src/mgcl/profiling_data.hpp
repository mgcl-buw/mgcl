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
#include <iostream>
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
     * There are four Profiling ticks in OpenCL: CL_PROFILING_COMMAND_*
     * - QUEUED: Time when the command was enqueued using OpenCL API
     * - SUBMIT: Time when the command was submitted to the device
     * - START: Time when the command started executing on the device
     * - END: Time when the command finished executing on the device
     * So, the process goes as: QUEUED -> SUBMIT -> START -> END.
     * @arg item-description
     */
    struct ProfilingMeasurement
    {
        cl_ulong queue_to_submit; // SUBMIT - QUEUED in ns
        cl_ulong submit_to_start; // START - SUBMIT in ns
        cl_ulong start_to_end;    // END - START in ns
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

        // Helper to compare work_items and work_group
        using GroupKey = std::tuple<std::array<size_t, 3>, std::array<size_t, 3>>;

        inline GroupKey make_group_key(const ProfilingMeasurement& m)
        {
            std::array<size_t, 3> wi = {m.work_items[0], m.work_items[1], m.work_items[2]};
            std::array<size_t, 3> wg = {m.work_group[0], m.work_group[1], m.work_group[2]};
            return std::make_tuple(wi, wg);
        }

    public:
        MeasurementsPerKernel& getMeasurements()
        {
            return measurements;
        }

        void addMeasurement(cl_command_queue commands, cl_event ev,
                            std::string kernelName,
                            std::array<size_t, 3> global, std::array<size_t, 3> local);

        void printBestTimingsPerKernel(std::ostream& os = std::cout);
        void printBestTimingsPerKernelAsCsv(std::ostream& os = std::cout);
        void printMeasurementsAsCsv(std::ostream& os = std::cout);

        friend std::ostream& operator<<(std::ostream& os, const ProfilingData& pd);
    };
}

#endif // __PROFILING_DATA_H__