#pragma once

#include <chrono>

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

    template <
        class result_t = std::chrono::milliseconds,
        class clock_t = std::chrono::steady_clock,
        class duration_t = std::chrono::milliseconds>
    auto mgcl_since(std::chrono::time_point<clock_t, duration_t> const &start)
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
        MGCL_VARYING
    } MGCL_STENCIL;

    enum class BC
    {
        // TODO maybe add Hybrid antyime in the future, need do adjust ghosts_in logic then.
        PERIODIC,
        DIRICHLET
    };
}
