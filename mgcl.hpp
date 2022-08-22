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
        MGCL_7POINT,
        MGCL_19POINT,
        MGCL_27POINT, // fixed laplacian stencils
        MGCL_7POINT_VARSYM,
        MGCL_19POINT_VARSYM,
        MGCL_27POINT_VARSYM // varying symmetric stencils
    } MGCL_STENCIL;
}
