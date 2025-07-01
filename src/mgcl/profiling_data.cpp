#include "profiling_data.hpp"
#include "opencl_helper.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <string>

namespace mgcl
{

    /**
     * @brief Prints measurements to ostream.
     *
     */
    std::ostream& operator<<(std::ostream& os, const ProfilingData& pd)
    {
        for (const auto& entry : pd.measurements)
        {
            auto kernelName = entry.first;
            os << kernelName << std::endl;
            for (const auto& m : entry.second)
            {
                os << "* kernel runtime in ns: " << m.elapsed << std::endl;
                os << "   time in queue in ns: " << m.inQueue << std::endl;
                os << "            work-items: " << m.work_items[0] << "," << m.work_items[1] << "," << m.work_items[2] << std::endl;
                os << "            work-group: " << m.work_group[0] << "," << m.work_group[1] << "," << m.work_group[2] << std::endl;
            }
        }
        return os;
    }

    void ProfilingData::addMeasurement(cl_command_queue commands, cl_event ev,
                                       std::string kernelName,
                                       std::array<size_t, 3> global, std::array<size_t, 3> local)
    {
        clFinish(commands);

        cl_ulong start_time, end_time;
        mgclCheckError(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL), "clGetEventProfilingInfo");
        mgclCheckError(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL), "clGetEventProfilingInfo");
        cl_ulong execution_time_ns = end_time - start_time;

        mgclCheckError(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &start_time, NULL), "clGetEventProfilingInfo");
        mgclCheckError(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &end_time, NULL), "clGetEventProfilingInfo");
        cl_ulong queue_time_in_ns = end_time - start_time;

        measurements[kernelName].push_back(ProfilingMeasurement{
            execution_time_ns,
            queue_time_in_ns,
            {global[0], global[1], global[2]},
            {local[0], local[1], local[2]}});
    }

    /**
     * @brief Finds and prints measurement with minimum elapsed time per kernel. Does not respect different
     * work-group sizes or different work-item counts.
     *
     */
    void ProfilingData::printBestTimingsPerKernel(std::ostream& os)
    {
        size_t maxKernelNameLength = 0;
        size_t maxElapsedLength = std::string("time [ns]").length();
        size_t maxQueueLength = std::string("queue [ns]").length();
        size_t maxWiLength = std::string("work-items").length();
        size_t maxWgLength = std::string("work-group").length();
        for (const auto& entry : measurements)
        {
            maxKernelNameLength = std::max(maxKernelNameLength, entry.first.length());
            for (const auto& m : entry.second)
            {
                maxElapsedLength = std::max(maxElapsedLength, std::to_string(m.elapsed).length());
                maxQueueLength = std::max(maxQueueLength, std::to_string(m.inQueue).length());
                maxWiLength = std::max(maxWiLength, std::to_string(m.work_items[0]).append(",").append(std::to_string(m.work_items[1]).append(",").append(std::to_string(m.work_items[2]))).length());
                maxWgLength = std::max(maxWgLength, std::to_string(m.work_group[0]).append(",").append(std::to_string(m.work_group[1]).append(",").append(std::to_string(m.work_group[2]))).length());
            }
        }

        os << std::setw(maxKernelNameLength + 2) << "kernel"
           << std::setw(maxElapsedLength + 2) << "time [ns]"
           << std::setw(maxQueueLength + 2) << "queue [ns]"
           << std::setw(maxWiLength + 2) << "work-items"
           << std::setw(maxWgLength + 2) << "work-group" << std::endl;

        // Iterate over all measurements per kernel
        for (const auto& entry : measurements)
        {
            // Create map per work-item count with measurements for the current kernel
            std::map<int, std::vector<ProfilingMeasurement>> tmp;
            for (const auto& m : entry.second)
                tmp[m.work_items[0] * (m.work_items[1] > 0 ? m.work_items[1] : 1) * (m.work_items[2] > 0 ? m.work_items[2] : 1)].push_back(m);

            // Find and print minimum elapsed time per work-item count for the current kernel
            for (const auto& m : tmp)
            {
                auto it = std::min_element(
                    m.second.begin(),
                    m.second.end(),
                    [](const ProfilingMeasurement& a, const ProfilingMeasurement& b)
                    {
                        return a.elapsed < b.elapsed;
                    });

                os << std::setw(maxKernelNameLength + 2) << entry.first
                   << std::setw(maxElapsedLength + 2) << it->elapsed
                   << std::setw(maxQueueLength + 2) << it->inQueue
                   << std::setw(maxWiLength + 2) << std::to_string(it->work_items[0]).append(",").append(std::to_string(it->work_items[1])).append(",").append(std::to_string(it->work_items[2]))
                   << std::setw(maxWgLength + 2) << std::to_string(it->work_group[0]).append(",").append(std::to_string(it->work_group[1])).append(",").append(std::to_string(it->work_group[2]))
                   << std::endl;
            }
        }
    }
} // namespace mgcl
