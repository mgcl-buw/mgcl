#!/bin/bash

# This script runs all tests in the MPI test folder.
# Run from build/test directory.
# New tests must be added manually.

# Cancel script on first failure
set -e

# Function that runs a test, captures the output of it and prints it only when
# the exit code is not 0. Arguments are the test command, so call it e.g. like so:
# run_test mpiexec --oversubscribe -n 4 "$exe" "MPI Problem::calculateAndSetMaxLevel (4 processes)"
run_test() {
  echo "Running mpiexec $* ..."
  cmdout=$(mpiexec "$@")
  ret=$?
  if ((ret)); then
    echo >&2 "Test returned code $ret:" 
    echo >&2 "$cmdout"
  fi
}

print_help() {
  echo "Available options:"
  echo "-p,--problem: Run Problem specific tests"
  echo "-l,--level: Run Level specific tests"
  echo "-g,--ghosts: Run Ghost-Update specific tests (Cuboid only)"
  echo "-j,--jacobi: Run Jacobi specific tests"
  echo "-r,--residual: Run Residual specific tests"
  echo "-u,--util: Run Utility specific tests (e.g. gather and scatter)"
  echo "-v,--vcycle: Run Vcycle specific tests (i.e. solving a problem)"
  echo "-s,--stencil: Run Stencil specific tests (mult, ghost update, etc.)"
  echo "If no option is given, all tests will be run."
}

TEST_ALL=true
TEST_PROBLEM=false
TEST_LEVEL=false
TEST_GHOSTS=false
TEST_JACOBI=false
TEST_RESIDUAL=false
TEST_UTIL=false
TEST_VCYCLE=false
TEST_STENCIL=false

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      print_help
      exit 0
      ;;
    -p|--problem)
      TEST_PROBLEM=true
      TEST_ALL=false
      shift # past argument
      ;;
    -l|--level)
      TEST_LEVEL=true
      TEST_ALL=false
      shift # past argument
      ;;
    -g|--ghosts)
      TEST_GHOSTS=true
      TEST_ALL=false
      shift # past argument
      ;;
    -j|--jacobi)
      TEST_JACOBI=true
      TEST_ALL=false
      shift # past argument
      ;;
    -r|--residual)
      TEST_RESIDUAL=true
      TEST_ALL=false
      shift # past argument
      ;;
    -u|--util)
      TEST_UTIL=true
      TEST_ALL=false
      shift # past argument
      ;;
    -v|--vcycle)
      TEST_VCYCLE=true
      TEST_ALL=false
      shift # past argument
      ;;
    -s|--stencil)
      TEST_STENCIL=true
      TEST_ALL=false
      shift # past argument
      ;;
  esac
done

if [ "$TEST_ALL" = true ] ; then
    echo "Running all tests..."
    echo ""
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
exe="$SCRIPT_DIR/tests_mpi"

if [ "$TEST_PROBLEM" = true ] || [ "$TEST_ALL" = true ] ; then
    echo "#######################"
    echo "Run Tests from test_mpi_problem.cpp ..."
    echo "#######################"

    run_test -n 1 "$exe" "MPI_Problem::setMpiComm"
    run_test -n 1 "$exe" "MPI_Problem::calculateAndSetMaxLevel_1proc"
    run_test --oversubscribe -n 4 "$exe" "MPI_Problem::calculateAndSetMaxLevel_4procs"
    run_test -n 2 "$exe" "MPI_Problem::calculateAndSetMpiLevelThreshold_valid_2procs"
    run_test -n 2 "$exe" "MPI_Problem::calculateAndSetMpiLevelThreshold_throwing_2procs"
    run_test -n 1 "$exe" "MPI_Problem::init"
    run_test -n 4 "$exe" "MPI_Problem::init"
fi

if [ "$TEST_LEVEL" = true ] || [ "$TEST_ALL" = true ] ; then
    echo "#######################"
    echo "Run Tests from test_mpi_level.cpp ..."
    echo "#######################"

    run_test -n 1 "$exe" "Level::initMpiData (1 process)"
    run_test --oversubscribe -n 2 "$exe" "Level::initMpiData (2 processes)"
    run_test --oversubscribe -n 8 "$exe" "Level::initMpiData (8 processes)"
    run_test --oversubscribe -n 24 "$exe" "Level::initMpiData (24 processes)"
fi

if [ "$TEST_GHOSTS" = true ] || [ "$TEST_ALL" = true ] ; then
    echo "#######################"
    echo "Run Tests from test_mpi_ghosts.cpp ..."
    echo "#######################"

    run_test -n 1 "$exe" "MPI updateGhostsSeq (1 process)"
    run_test --oversubscribe -n 8 "$exe" "MPI updateGhostsSeq (n processes)"
    run_test --oversubscribe -n 8 "$exe" "MPI updateGhosts ocl (n processes)"
fi

if [ "$TEST_JACOBI" = true ] || [ "$TEST_ALL" = true ] ; then
    echo "#######################"
    echo "Run Tests from test_mpi_jacobi.cpp ..."
    echo "#######################"

    run_test --oversubscribe -n 1 "$exe" "MPI_jacobiSeq_Laplace_n_processes"
    run_test --oversubscribe -n 8 "$exe" "MPI_jacobiSeq_Laplace_n_processes"
    run_test --oversubscribe -n 1 "$exe" "MPI_jacobiSeq_VaryingStencil_n_processes"
    run_test --oversubscribe -n 8 "$exe" "MPI_jacobiSeq_VaryingStencil_n_processes"
    run_test --oversubscribe -n 1 "$exe" "MPI_jacobi_ocl_Laplace_n_processes"
    run_test --oversubscribe -n 8 "$exe" "MPI_jacobi_ocl_Laplace_n_processes"
    run_test --oversubscribe -n 1 "$exe" "MPI_jacobi_ocl_VaryingStencil_n_processes"
    run_test --oversubscribe -n 8 "$exe" "MPI_jacobi_ocl_VaryingStencil_n_processes"
fi

if [ "$TEST_RESIDUAL" = true ] || [ "$TEST_ALL" = true ] ; then
    echo "#######################"
    echo "Run Tests from test_mpi_residual.cpp ..."
    echo "#######################"

    run_test --oversubscribe -n 1 "$exe" "MPI_residual_seq_VaryingStencil_n_processes"
    run_test --oversubscribe -n 8 "$exe" "MPI_residual_seq_VaryingStencil_n_processes"
    run_test --oversubscribe -n 1 "$exe" "MPI_residual_ocl_VaryingStencil_n_processes"
    run_test --oversubscribe -n 8 "$exe" "MPI_residual_ocl_VaryingStencil_n_processes"
fi

if [ "$TEST_UTIL" = true ] || [ "$TEST_ALL" = true ] ; then
    echo "#######################"
    echo "Run Tests from test_mpi_util.cpp ..."
    echo "#######################"

    run_test --oversubscribe -n 17 "$exe" "mpi_util::gather-src-dest-different"
    run_test --oversubscribe -n 17 "$exe" "mpi_util::gather-src-dest-same"
    run_test --oversubscribe -n 17 "$exe" "mpi_util::gather-src-dest-same-different-gh"
    run_test --oversubscribe -n 5 "$exe" "mpi_util::gather-GPU-src-dest-same-different-gh"
    run_test --oversubscribe -n 17 "$exe" "mpi_util::scatter-src-dest-same"
    run_test --oversubscribe -n 17 "$exe" "mpi_util::scatter-src-dest-different"
    run_test --oversubscribe -n 17 "$exe" "mpi_util::scatter-src-dest-same-with-ghosts"
    run_test --oversubscribe -n 17 "$exe" "mpi_util::scatter-GPU-src-dest-same-with-ghosts"
    run_test --oversubscribe -n 17 "$exe" "mpi_util::gather-src-dest-same-stencil"
    run_test --oversubscribe -n 5 "$exe" "mpi_util::gather-GPU-src-dest-same-stencil"
fi

if [ "$TEST_VCYCLE" = true ] || [ "$TEST_ALL" = true ] ; then
    echo "#######################"
    echo "Run Tests from test_mpi_vcycle.cpp ..."
    echo "#######################"

    run_test --oversubscribe -n 1 "$exe" "MPI_vcycle_immediate_gather_scatter_Laplace7p"
    run_test --oversubscribe -n 4 "$exe" "MPI_vcycle_immediate_gather_scatter_Laplace7p"
    run_test --oversubscribe -n 4 "$exe" "MPI_vcycle_threshold_gt_0_Laplace7p"
    run_test --oversubscribe -n 1 "$exe" "MPI_vcycle_immediate_gather_scatter_Varying27p"
    run_test --oversubscribe -n 4 "$exe" "MPI_vcycle_immediate_gather_scatter_Varying27p"
    run_test --oversubscribe -n 1 "$exe" "MPI_vcycle_threshold_eq_1_Varying27p"
    run_test --oversubscribe -n 4 "$exe" "MPI_vcycle_threshold_eq_1_Varying27p"
    run_test --oversubscribe -n 1 "$exe" "MPI_vcycle_threshold_eq_2_Varying27p"
    run_test --oversubscribe -n 4 "$exe" "MPI_vcycle_threshold_eq_2_Varying27p"
    run_test --oversubscribe -n 1 "$exe" "MPI_vcycle_GPU_immediate_gather_scatter_Laplace7p"
    run_test --oversubscribe -n 4 "$exe" "MPI_vcycle_GPU_immediate_gather_scatter_Laplace7p"
    run_test --oversubscribe -n 1 "$exe" "MPI_vcycle_GPU_threshold_gt_0_Laplace7p"
    run_test --oversubscribe -n 4 "$exe" "MPI_vcycle_GPU_threshold_gt_0_Laplace7p"
    run_test --oversubscribe -n 1 "$exe" "MPI_vcycle_GPU_threshold_eq_1_Varying27p"
    run_test --oversubscribe -n 4 "$exe" "MPI_vcycle_GPU_threshold_eq_1_Varying27p"
    run_test --oversubscribe -n 1 "$exe" "MPI_vcycle_GPU_threshold_eq_2_Varying27p"
    run_test --oversubscribe -n 4 "$exe" "MPI_vcycle_GPU_threshold_eq_2_Varying27p"
    run_test --oversubscribe -n 4 "$exe" "MPI_vcycle_GPU_threshold_eq_2_Varying27p_multiple_jacobi_iters"
    run_test --oversubscribe -n 4 "$exe" "MPI_vcycle_multiple_solve_calls"
    run_test --oversubscribe -n 4 "$exe" "MPI_vcycle_different_relres"
fi

if [ "$TEST_STENCIL" = true ] || [ "$TEST_ALL" = true ] ; then
    echo "#######################"
    echo "Run Tests from test_mpi_stencil.cpp ..."
    echo "#######################"

    run_test --oversubscribe -n 1 "$exe" "MPI-stencil-updateGhostsSeq-1proc"
    run_test --oversubscribe -n 2 "$exe" "MPI-stencil-updateGhostsSeq-nprocs"
    run_test --oversubscribe -n 4 "$exe" "MPI-stencil-updateGhostsSeq-nprocs"
    run_test --oversubscribe -n 2 "$exe" "MPI-updateGhostsStencilOclMpi-nprocs"
    run_test --oversubscribe -n 4 "$exe" "MPI-updateGhostsStencilOclMpi-nprocs"
    run_test --oversubscribe -n 1 "$exe" MPI_galerkin_different_thresholds
    run_test --oversubscribe -n 4 "$exe" MPI_galerkin_different_thresholds
fi

echo "Done. All good!"

