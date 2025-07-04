#pragma once

#include <chrono>

#ifdef MGCL_PRINT_TRACE
#include <cpptrace/cpptrace.hpp>
#endif

namespace mgcl
{
// define mgcl_debug for debugging output. Compile with -D MGCL_DEBUG to enable, e.g. run: make CPPFLAGS="-D MGCL_DEBUG"
// taken from https://stackoverflow.com/questions/1644868/define-macro-for-debug-printing-in-c
#ifdef MGCL_DEBUG
#define mgcl_debug(fmt, ...) printf("%s:%d:%s(): " fmt, __FILE__, __LINE__, __func__, ##__VA_ARGS__)
#else
#define mgcl_debug(fmt, ...) \
    do                       \
    {                        \
    } while (0)
#endif

// Define the function mgcl::error(msg) for throwing an exception in case of an error. If stack tracing is
// enabled, the stack trace is printed.
#ifdef MGCL_PRINT_TRACE
#define error(msg)                                        \
    do                                                    \
    {                                                     \
        cpptrace::generate_trace().print_with_snippets(); \
        throw msg;                                        \
    } while (0)
#else
#define error(msg) throw msg
#endif

#define warning(msg) printf("mgcl warning: %s\n", msg);

    template <
        class result_t = std::chrono::milliseconds,
        class clock_t = std::chrono::steady_clock,
        class duration_t = std::chrono::milliseconds>
    auto mgcl_since(std::chrono::time_point<clock_t, duration_t> const& start)
    {
        return std::chrono::duration_cast<result_t>(clock_t::now() - start);
    }

    typedef enum
    {
        MGCL_L2,
        MGCL_INF
    } MGCL_RESIDUAL_NORM;

    typedef enum
    {
        MGCL_LAPLACE_7POINT,
        MGCL_LAPLACE_19POINT,
        MGCL_LAPLACE_27POINT,
        MGCL_VARYING,
        MGCL_FIXED,
        MGCL_BLOCKSTENCIL
    } MGCL_STENCIL;

    typedef enum
    {
        MGCL_JACOBI_SCALAR,
        MGCL_JACOBI_BLOCK
    } MGCL_SMOOTHER;

    enum class BC
    {
        // TODO maybe add Hybrid antyime in the future, need do adjust ghosts_in logic then.
        PERIODIC,
        DIRICHLET
    };

    enum class OCL_DEVICE_STRATEGY
    {
        /**
         * Always use the first device that matches device type and name.
         */
        ALWAYS_FIRST,

        /**
         * If multiple devices are available and mgcl is run with MPI, distribute devices evenly accross the processes.
         * rank 0 gets device 0, rank 1 device 1 etc. If #ranks > #devices, rank #devices gets device 0 again,
         * rank #devices+1 gets device 1 etc. This is the default.
         *
         */
        DISTRIBUTE_EVENLY
    };
}
