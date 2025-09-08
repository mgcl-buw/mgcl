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
                os << "* queued to submit [ns]: " << m.queue_to_submit << std::endl;
                os << "   queued to start [ns]: " << m.submit_to_start << std::endl;
                os << "      start to end [ns]: " << m.start_to_end << std::endl;
                os << "             work-items: " << m.work_items[0] << "," << m.work_items[1] << "," << m.work_items[2] << std::endl;
                os << "             work-group: " << m.work_group[0] << "," << m.work_group[1] << "," << m.work_group[2] << std::endl;
            }
        }
        return os;
    }

    void ProfilingData::addMeasurement(cl_command_queue commands, cl_event ev,
                                       std::string kernelName,
                                       std::array<size_t, 3> global, std::array<size_t, 3> local)
    {
        clFinish(commands);

        cl_ulong start, end, queued, submit;
        mgclCheckError(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queued, NULL), "clGetEventProfilingInfo");
        mgclCheckError(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submit, NULL), "clGetEventProfilingInfo");
        mgclCheckError(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL), "clGetEventProfilingInfo");
        mgclCheckError(clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL), "clGetEventProfilingInfo");

        measurements[kernelName].push_back(ProfilingMeasurement{
            submit - queued, // Time from QUEUED to SUBMIT
            start - submit,  // Time from SUBMIT to START
            end - start,     // Time from START to END
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
        size_t maxQueueLength = std::string("queue to submit [ns]").length();
        size_t maxSubmitLength = std::string("submit to start [ns]").length();
        size_t maxElapsedLength = std::string("start to end [ns]").length();
        size_t maxWiLength = std::string("work-items").length();
        size_t maxWgLength = std::string("work-group").length();
        for (const auto& entry : measurements)
        {
            maxKernelNameLength = std::max(maxKernelNameLength, entry.first.length());
            for (const auto& m : entry.second)
            {
                maxElapsedLength = std::max(maxElapsedLength, std::to_string(m.start_to_end).length());
                maxQueueLength = std::max(maxQueueLength, std::to_string(m.queue_to_submit).length());
                maxSubmitLength = std::max(maxSubmitLength, std::to_string(m.submit_to_start).length());
                maxWiLength = std::max(maxWiLength, std::to_string(m.work_items[0]).append(",").append(std::to_string(m.work_items[1]).append(",").append(std::to_string(m.work_items[2]))).length());
                maxWgLength = std::max(maxWgLength, std::to_string(m.work_group[0]).append(",").append(std::to_string(m.work_group[1]).append(",").append(std::to_string(m.work_group[2]))).length());
            }
        }

        os << "| " << std::setw(maxKernelNameLength) << "kernel"
           << " | " << std::setw(maxQueueLength) << "queue to submit [ns]"
           << " | " << std::setw(maxSubmitLength) << "submit to start [ns]"
           << " | " << std::setw(maxElapsedLength) << "start to end [ns]"
           << " | " << std::setw(maxWiLength) << "work-items"
           << " | " << std::setw(maxWgLength) << "work-group"
           << " |" << std::endl;
        os << "|--" << std::string(maxKernelNameLength, '-')
           << "|--" << std::string(maxQueueLength, '-')
           << "|--" << std::string(maxSubmitLength, '-')
           << "|--" << std::string(maxElapsedLength, '-')
           << "|--" << std::string(maxWiLength, '-')
           << "|--" << std::string(maxWgLength, '-') << "|" << std::endl;

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
                        return a.start_to_end < b.start_to_end;
                    });

                os << "| " << std::setw(maxKernelNameLength) << entry.first
                   << " | " << std::setw(maxQueueLength) << it->queue_to_submit
                   << " | " << std::setw(maxSubmitLength) << it->submit_to_start
                   << " | " << std::setw(maxElapsedLength) << it->start_to_end
                   << " | " << std::setw(maxWiLength) << std::to_string(it->work_items[0]).append(",").append(std::to_string(it->work_items[1])).append(",").append(std::to_string(it->work_items[2]))
                   << " | " << std::setw(maxWgLength) << std::to_string(it->work_group[0]).append(",").append(std::to_string(it->work_group[1])).append(",").append(std::to_string(it->work_group[2]))
                   << " |" << std::endl;
            }
        }
    }

    void ProfilingData::printBestTimingsPerKernelAsCsv(std::ostream& os)
    {
        os << "kernel;queue_to_submit_ns;submit_to_start_ns;start_to_end_ns;work_items_x;work_items_y;work_items_z;work_group_x;work_group_y;work_group_z" << std::endl;
        for (const auto& entry : measurements)
        {
            auto kernelName = entry.first;
            auto it = std::min_element(
                entry.second.begin(),
                entry.second.end(),
                [](const ProfilingMeasurement& a, const ProfilingMeasurement& b)
                {
                    return a.start_to_end < b.start_to_end;
                });
            os << kernelName << ";"
               << it->queue_to_submit << ";"
               << it->submit_to_start << ";"
               << it->start_to_end << ";"
               << it->work_items[0] << ";" << it->work_items[1] << ";" << it->work_items[2] << ";"
               << it->work_group[0] << ";" << it->work_group[1] << ";" << it->work_group[2] << std::endl;
        }

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
                        return a.start_to_end < b.start_to_end;
                    });

                os << entry.first << ";"
                   << it->queue_to_submit << ";"
                   << it->submit_to_start << ";"
                   << it->start_to_end << ";"
                   << it->work_items[0] << ";" << it->work_items[1] << ";" << it->work_items[2] << ";"
                   << it->work_group[0] << ";" << it->work_group[1] << ";" << it->work_group[2] << std::endl;
            }
        }
    }

    void ProfilingData::printMeasurementsAsCsv(std::ostream& os)
    {
        os << "kernel;queue_to_submit_ns;submit_to_start_ns;start_to_end_ns;work_items_x;work_items_y;work_items_z;work_group_x;work_group_y;work_group_z" << std::endl;
        for (const auto& entry : measurements)
        {
            auto kernelName = entry.first;
            for (const auto& m : entry.second)
            {
                os << kernelName << ";"
                   << m.queue_to_submit << ";"
                   << m.submit_to_start << ";"
                   << m.start_to_end << ";"
                   << m.work_items[0] << ";" << m.work_items[1] << ";" << m.work_items[2] << ";"
                   << m.work_group[0] << ";" << m.work_group[1] << ";" << m.work_group[2] << std::endl;
            }
        }
    }
} // namespace mgcl
