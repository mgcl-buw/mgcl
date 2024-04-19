#pragma once

#include "nanobench.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <mpi.h>

namespace bench_util
{
    struct Result
    {
        std::string name;
        double minTime;
        double medianTime;
        double avgTime;
        double medianAbsolutePercentError;
        int m;
        int n;
        int o;
    };

    struct ResultMpi
    {
        std::string name;
        double minTime;
        double medianTime;
        double avgTime;
        double medianAbsolutePercentError;
        int m;
        int n;
        int o;
        int mglob;
        int nglob;
        int oglob;
        int gpus; // GPU-count
        int LT;   // mpiLevelThreshold
    };

    struct ResultJacobiMpi
    {
        std::string name;
        double minTime;
        double medianTime;
        double avgTime;
        double medianAbsolutePercentError;
        int mloc;
        int nloc;
        int oloc;
        int mglob;
        int nglob;
        int oglob;
        int spi; // steps per iteration
        int iters;
        int gpus; // GPU-count, equals mpi proc count
    };

    struct ResultGhostUpdateMpi
    {
        std::string name;
        double minTime;
        double medianTime;
        double avgTime;
        double medianAbsolutePercentError;
        int mloc;
        int nloc;
        int oloc;
        int mglob;
        int nglob;
        int oglob;
        int ghosts; // amount of ghost cells in one direction
    };

    /**
     * @brief Get minimum time from measurements for a specific benchmark name in ms.
     *
     * @param b
     * @param name Name of the bench to filter for.
     * @return double
     */
    inline double getMinTime(ankerl::nanobench::Bench& b, const std::string& name)
    {
        double min = std::numeric_limits<double>::max();
        for (auto& r : b.results())
        {
            if (r.config().mBenchmarkName == name && r.minimum(ankerl::nanobench::Result::Measure::elapsed) < min)
                min = r.minimum(ankerl::nanobench::Result::Measure::elapsed) /* * 1000.0 * 1000.0*/;
        }
        return min * 1000.0;
    }

    /**
     * @brief Get median time from measurements starting from idxstart in ms.
     *
     * @param b
     * @param name Name of the bench to filter for.
     * @return double
     */
    inline double getMedianTime(ankerl::nanobench::Bench& b, const std::string& name)
    {
        double median = std::numeric_limits<double>::max();
        for (auto& r : b.results())
        {
            if (r.config().mBenchmarkName == name && r.median(ankerl::nanobench::Result::Measure::elapsed) < median)
                median = r.median(ankerl::nanobench::Result::Measure::elapsed) /* * 1000.0 * 1000.0*/;
        }
        return median * 1000.0;
    }

    /**
     * @brief Get average time from measurements starting from idxstart in ms.
     *
     * @param b
     * @param name Name of the bench to filter for.
     * @return double
     */
    inline double getAvgTime(ankerl::nanobench::Bench& b, const std::string& name)
    {
        double avg = std::numeric_limits<double>::max();
        for (auto& r : b.results())
        {
            if (r.config().mBenchmarkName == name && r.average(ankerl::nanobench::Result::Measure::elapsed) < avg)
                avg = r.average(ankerl::nanobench::Result::Measure::elapsed) /* * 1000.0 * 1000.0*/;
        }
        return avg * 1000.0;
    }

    /**
     * @brief Returns median absolute percent error for a specific benchmark name, or -1 if not found.
     * see https://nanobench.ankerl.com/reference.html#render-mustache-like-templates
     *
     * @param b
     * @param name Name of the bench to filter for.
     * @return double
     */
    inline double getMedianAbsolutePercentError(ankerl::nanobench::Bench& b, const std::string& name)
    {
        for (auto& r : b.results())
        {
            if (r.config().mBenchmarkName == name)
                return r.medianAbsolutePercentError(ankerl::nanobench::Result::Measure::elapsed) * 100.0;
        }
        return -1;
    }

    inline void printCsvFormat(std::vector<Result> minTimes)
    {
        std::stringstream ss;
        ss << std::endl;
        ss << "***DATASTART***" << std::endl;
        ss << "name;m;n;o;minTimeInMs" << std::endl;
        for (auto r : minTimes)
        {
            ss << r.name << ";" << r.m << ";" << r.n << ";" << r.o << ";" << std::setprecision(17) << r.minTime << std::endl;
        }
        ss << "***DATAEND***" << std::endl;
        std::string output = ss.str();
        std::replace(output.begin(), output.end(), '.', ',');
        std::cout << output;
    }

    inline void printCsvFormat(std::vector<ResultMpi> minTimes, MPI_Comm mpi_comm, int mpi_rank)
    {
        // print min times
        MPI_Barrier(mpi_comm);
        if (mpi_rank == 0)
        {
            std::stringstream ss;
            ss << std::endl;
            ss << "***DATASTART***" << std::endl;
            ss << "gpus;LT;name;m;n;o;mglob;nglob;oglob;minTimeInMs;medianTimeInMs;avgTimeInMs;err%" << std::endl;
            for (auto r : minTimes)
            {
                ss << r.gpus << ";" << r.LT << ";"
                   << r.name << ";" << r.m << ";" << r.n << ";" << r.o << ";"
                   << r.mglob << ";" << r.nglob << ";" << r.oglob
                   << ";" << std::setprecision(17) << r.minTime
                   << ";" << std::setprecision(17) << r.medianTime
                   << ";" << std::setprecision(17) << r.avgTime
                   << ";" << std::setprecision(4) << r.medianAbsolutePercentError << "%"
                   << std::endl;
            }
            ss << "***DATAEND***" << std::endl;
            std::string output = ss.str();
            std::replace(output.begin(), output.end(), '.', ',');
            std::cout << output;
        }
        MPI_Barrier(mpi_comm);
    }

    inline void printCsvFormat(std::vector<ResultJacobiMpi> results, MPI_Comm mpi_comm, int mpi_rank)
    {
        // print min times
        MPI_Barrier(mpi_comm);
        if (mpi_rank == 0)
        {
            std::stringstream ss;
            ss << std::endl;
            ss << "***DATASTART***" << std::endl;
            // ss << "gpus;spi;iters;name;mloc;nloc;oloc;mglob;nglob;oglob;minTimeInMs;medianTimeInMs;avgTimeInMs;err%" << std::endl;
            ss << "gpus;spi;iters;mloc;nloc;oloc;mglob;nglob;oglob;minTimeInMs;medianTimeInMs;avgTimeInMs;err%" << std::endl;
            for (auto r : results)
            {
                ss << r.gpus << ";" << r.spi << ";" << r.iters << ";"
                   /* << r.name << ";" */
                   << r.mloc << ";" << r.nloc << ";" << r.oloc << ";"
                   << r.mglob << ";" << r.nglob << ";" << r.oglob
                   << ";" << std::setprecision(17) << r.minTime
                   << ";" << std::setprecision(17) << r.medianTime
                   << ";" << std::setprecision(17) << r.avgTime
                   << ";" << std::setprecision(4) << r.medianAbsolutePercentError << "%"
                   << std::endl;
            }
            ss << "***DATAEND***" << std::endl;
            std::string output = ss.str();
            std::replace(output.begin(), output.end(), '.', ',');
            std::cout << output;
        }
        MPI_Barrier(mpi_comm);
    }

    inline void printCsvFormat(std::vector<ResultGhostUpdateMpi> results, MPI_Comm mpi_comm, int mpi_rank)
    {
        // print min times
        MPI_Barrier(mpi_comm);
        if (mpi_rank == 0)
        {
            std::stringstream ss;
            ss << std::endl;
            ss << "***DATASTART***" << std::endl;
            // ss << "gpus;spi;iters;name;mloc;nloc;oloc;mglob;nglob;oglob;minTimeInMs;medianTimeInMs;avgTimeInMs;err%" << std::endl;
            ss << "ghosts;mloc;nloc;oloc;mglob;nglob;oglob;minTimeInMs;medianTimeInMs;avgTimeInMs;err%" << std::endl;
            for (auto r : results)
            {
                ss << r.ghosts << ";"
                   /* << r.name << ";" */
                   << r.mloc << ";" << r.nloc << ";" << r.oloc << ";"
                   << r.mglob << ";" << r.nglob << ";" << r.oglob
                   << ";" << std::setprecision(17) << r.minTime
                   << ";" << std::setprecision(17) << r.medianTime
                   << ";" << std::setprecision(17) << r.avgTime
                   << ";" << std::setprecision(4) << r.medianAbsolutePercentError << "%"
                   << std::endl;
            }
            ss << "***DATAEND***" << std::endl;
            std::string output = ss.str();
            std::replace(output.begin(), output.end(), '.', ',');
            std::cout << output;
        }
        MPI_Barrier(mpi_comm);
    }

}