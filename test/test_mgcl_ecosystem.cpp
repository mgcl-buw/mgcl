#include <catch2/catch_test_macros.hpp>

#include "../mgcl.hpp"

TEST_CASE("mgcl_generate_config")
{
    mgcl::mgcl_config *conf;
    mgcl::mgcl_generate_config(&conf);

    SECTION("default values")
    {
        REQUIRE(conf->v == NULL);
        REQUIRE(conf->f == NULL);
        REQUIRE(conf->m == -1);
        REQUIRE(conf->n == -1);
        REQUIRE(conf->o == -1);
        REQUIRE(conf->ghosts == 1);
        REQUIRE(conf->ghosts_in == 0);
        REQUIRE(conf->nu1 == 2);
        REQUIRE(conf->nu2 == 2);
        REQUIRE(conf->omega == 0.8);
        REQUIRE(conf->maxiter_vcycles == 5);
        REQUIRE(conf->tol == 1e-7);
        REQUIRE(conf->maxlevel == -1);
        REQUIRE(conf->residual_norm == mgcl::MGCL_L2);
        REQUIRE(conf->stencil == mgcl::MGCL_7POINT);

        REQUIRE(conf->device_type == CL_DEVICE_TYPE_DEFAULT);
        REQUIRE(conf->kernel_dir == "./");
        REQUIRE(conf->device_name == "");
        REQUIRE(conf->device_id == NULL);
        REQUIRE(conf->commands == NULL);
        REQUIRE(conf->context == NULL);
        REQUIRE(conf->d_v == NULL);
        REQUIRE(conf->d_f == NULL);
        REQUIRE(conf->use_opencl == 0);
        REQUIRE(conf->reuse_opencl_buffers == 0);
        REQUIRE(conf->copy_buffer_data == 0);
        REQUIRE(conf->read_results == 0);
        REQUIRE(conf->use_local_memory == 0);

        REQUIRE(conf->jacobi_wg_size_x == 16);
        REQUIRE(conf->jacobi_wg_size_y == 16);
        REQUIRE(conf->jacobi_iterations_per_kernel == 3);

        REQUIRE(conf->stencil_size_multiplier == 1);
        REQUIRE(conf->stencil_values == NULL);
        REQUIRE(conf->d_stencil_values == NULL);
        REQUIRE(conf->restrict_prolongate_stencil == 1);

        free(conf);
    }
}

TEST_CASE("mgcl_init")
{
    mgcl::mgcl_config *conf;
    mgcl::mgcl_generate_config(&conf);
    conf->m = 3;
    conf->n = 3;
    conf->o = 3;

    mgcl::mgcl_level_data *level_data;
    int err = EXIT_SUCCESS;

    SECTION("invalid conf.m")
    {
        conf->m = 0;
        err = mgcl::mgcl_init(conf, &level_data);
        REQUIRE(err == EXIT_FAILURE);
        free(level_data);
        free(conf);
    }

    SECTION("invalid conf.n")
    {
        conf->n = 0;
        err = mgcl::mgcl_init(conf, &level_data);
        REQUIRE(err == EXIT_FAILURE);
        free(level_data);
        free(conf);
    }

    SECTION("invalid conf.o")
    {
        conf->o = 0;
        err = mgcl::mgcl_init(conf, &level_data);
        REQUIRE(err == EXIT_FAILURE);
        free(level_data);
        free(conf);
    }

    SECTION("invalid conf.f")
    {
        conf->m = 0;
        err = mgcl::mgcl_init(conf, &level_data);
        REQUIRE(err == EXIT_FAILURE);
        free(level_data);
        free(conf);
    }
}