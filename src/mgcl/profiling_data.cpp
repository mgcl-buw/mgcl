#include "profiling_data.hpp"
#include "opencl_helper.hpp"

#include <iostream>

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
                os << "* time in ns: " << m.elapsed << std::endl;
                os << "  work-items: " << m.work_items[0] << "," << m.work_items[1] << "," << m.work_items[2] << std::endl;
                os << "  work-group: " << m.work_group[0] << "," << m.work_group[1] << "," << m.work_group[2] << std::endl;
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

        measurements[kernelName].push_back(ProfilingMeasurement{
            execution_time_ns,
            {global[0], global[1], global[2]},
            {local[0], local[1], local[2]}});
    }
} // namespace mgcl
