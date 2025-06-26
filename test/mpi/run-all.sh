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
  echo "-g,--ghosts: Run Ghost-Update specific tests (Cuboid and CuboidBS only)"
  echo "-j,--jacobi: Run Jacobi specific tests"
  echo "-r,--residual: Run Residual specific tests"
  echo "-u,--util: Run Utility specific tests (e.g. gather and scatter)"
  echo "-v,--vcycle: Run Vcycle specific tests (i.e. solving a problem)"
  echo "-s,--stencil: Run Stencil specific tests (mult, ghost update, etc.)"
  echo "-a,--galerkin: Run Galerkin specific tests (galerkinOptimized, Problem::init)"
  echo "--no-ocl: Skip OpenCL tests"
  echo "--ocl-device-types: Specify OpenCL device types to use, separated by comma. Allowed are: gpu,cpu. Default: gpu."
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
TEST_GALERKIN=false
NO_OCL=false
OCL_DEVICE_TYPES="gpu"

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
    -a|--galerkin)
      TEST_GALERKIN=true
      TEST_ALL=false
      shift # past argument
      ;;
    --no-ocl)
      NO_OCL=true
      shift # past argument
      ;;
    --ocl-device-types)
      shift # past argument
      if [[ "$1" =~ .*gpu.* ]] && [[ "$1" =~ .*cpu.* ]]; then
        echo "Running OpenCL tests on GPU and CPU"
        OCL_DEVICE_TYPES="gpu,cpu"
      elif [[ "$1" =~ .*cpu.* ]] ; then
        echo "Running OpenCL tests only on CPU"
        OCL_DEVICE_TYPES="cpu"
      fi
      shift # past argument
  esac
done

if [ "$TEST_ALL" = true ] ; then
    echo "Running all tests..."
fi

if [ "$NO_OCL" = true ] ; then
    echo "but skipping GPU tests..."
fi

echo ""

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
    run_test --oversubscribe -n 8 "$exe" "MPI-updateGhostsSeq-CuboidBS_(n_processes)"
    run_test --oversubscribe -n 8 "$exe" "MPI-updateGhostsSeq-Blockstencil_(n_processes)"
    if [ "$NO_OCL" = false ] ; then
        run_test --oversubscribe -n 8 "$exe" "MPI updateGhosts ocl (n processes)" --deviceTypes "$OCL_DEVICE_TYPES"
        run_test --oversubscribe -n 8 "$exe" "MPI_updateGhosts_ocl_CuboidBS_(n_processes)" --deviceTypes "$OCL_DEVICE_TYPES"
        run_test --oversubscribe -n 8 "$exe" "MPI_updateGhosts_ocl_Blockstencil_(n_processes)" --deviceTypes "$OCL_DEVICE_TYPES"
    fi
fi

if [ "$TEST_JACOBI" = true ] || [ "$TEST_ALL" = true ] ; then
    echo "#######################"
    echo "Run Tests from test_mpi_jacobi.cpp ..."
    echo "#######################"

    run_test --oversubscribe -n 1 "$exe" "MPI_jacobiSeq_Laplace_n_processes"
    run_test --oversubscribe -n 8 "$exe" "MPI_jacobiSeq_Laplace_n_processes"
    run_test --oversubscribe -n 1 "$exe" "MPI_jacobiSeq_VaryingStencil_n_processes"
    run_test --oversubscribe -n 8 "$exe" "MPI_jacobiSeq_VaryingStencil_n_processes"
    if [ "$NO_OCL" = false ] ; then
      run_test --oversubscribe -n 1 "$exe" "MPI_jacobi_ocl_Laplace_n_processes" --deviceTypes "$OCL_DEVICE_TYPES"
      run_test --oversubscribe -n 8 "$exe" "MPI_jacobi_ocl_Laplace_n_processes" --deviceTypes "$OCL_DEVICE_TYPES"
      run_test --oversubscribe -n 1 "$exe" "MPI_jacobi_ocl_VaryingStencil_n_processes" --deviceTypes "$OCL_DEVICE_TYPES"
      run_test --oversubscribe -n 8 "$exe" "MPI_jacobi_ocl_VaryingStencil_n_processes" --deviceTypes "$OCL_DEVICE_TYPES"
    fi
fi

if [ "$TEST_RESIDUAL" = true ] || [ "$TEST_ALL" = true ] ; then
    echo "#######################"
    echo "Run Tests from test_mpi_residual.cpp ..."
    echo "#######################"

    run_test --oversubscribe -n 1 "$exe" "MPI_residual_seq_VaryingStencil_n_processes"
    run_test --oversubscribe -n 8 "$exe" "MPI_residual_seq_VaryingStencil_n_processes"
    if [ "$NO_OCL" = false ] ; then
      run_test --oversubscribe -n 1 "$exe" "MPI_residual_ocl_VaryingStencil_n_processes" --deviceTypes "$OCL_DEVICE_TYPES"
      run_test --oversubscribe -n 8 "$exe" "MPI_residual_ocl_VaryingStencil_n_processes" --deviceTypes "$OCL_DEVICE_TYPES"
    fi
fi

if [ "$TEST_UTIL" = true ] || [ "$TEST_ALL" = true ] ; then
    echo "#######################"
    echo "Run Tests from test_mpi_util.cpp ..."
    echo "#######################"

    run_test --oversubscribe -n 17 "$exe" "mpi_util::gather-src-dest-different"
    run_test --oversubscribe -n 17 "$exe" "mpi_util::gather-src-dest-same"
    run_test --oversubscribe -n 17 "$exe" "mpi_util::gather-src-dest-same-different-gh"
    run_test --oversubscribe -n 17 "$exe" "mpi_util::scatter-src-dest-same"
    run_test --oversubscribe -n 17 "$exe" "mpi_util::scatter-src-dest-different"
    run_test --oversubscribe -n 17 "$exe" "mpi_util::scatter-src-dest-same-with-ghosts"
    run_test --oversubscribe -n 17 "$exe" "mpi_util::gather-src-dest-same-stencil"
    run_test --oversubscribe -n 17 "$exe" "mpi_util::gather-cuboidbs-src-dest-same"
    run_test --oversubscribe -n 17 "$exe" "mpi_util::gather-blockstencil-src-dest-same"
    run_test --oversubscribe -n 17 "$exe" "mpi_util::scatter-cuboidbs-src-dest-same-with-ghosts"

    run_test --oversubscribe -n 2 "$exe" "mpi_util::sendBorderPlanes_cuboid"
    run_test --oversubscribe -n 8 "$exe" "mpi_util::sendBorderPlanes_cuboid"
    run_test --oversubscribe -n 17 "$exe" "mpi_util::sendBorderPlanes_cuboid"
    run_test --oversubscribe -n 2 "$exe" "mpi_util::sendBorderPlanes_stencil"
    run_test --oversubscribe -n 8 "$exe" "mpi_util::sendBorderPlanes_stencil"
    run_test --oversubscribe -n 17 "$exe" "mpi_util::sendBorderPlanes_stencil"
    run_test --oversubscribe -n 2 "$exe" "mpi_util::sendBorderPlanes_cuboidbs"
    run_test --oversubscribe -n 8 "$exe" "mpi_util::sendBorderPlanes_cuboidbs"
    run_test --oversubscribe -n 17 "$exe" "mpi_util::sendBorderPlanes_cuboidbs"
    run_test --oversubscribe -n 2 "$exe" "mpi_util::sendBorderPlanes_blockstencil"
    run_test --oversubscribe -n 8 "$exe" "mpi_util::sendBorderPlanes_blockstencil"
    run_test --oversubscribe -n 17 "$exe" "mpi_util::sendBorderPlanes_blockstencil"

    if [ "$NO_OCL" = false ] ; then
      run_test --oversubscribe -n 5 "$exe" "mpi_util::gather-GPU-src-dest-same-different-gh" --deviceTypes "$OCL_DEVICE_TYPES"
      run_test --oversubscribe -n 17 "$exe" "mpi_util::scatter-GPU-src-dest-same-with-ghosts" --deviceTypes "$OCL_DEVICE_TYPES"
      run_test --oversubscribe -n 17 "$exe" "mpi_util::scatter-cuboidbs-GPU-src-dest-same-with-ghosts" --deviceTypes "$OCL_DEVICE_TYPES"
      run_test --oversubscribe -n 5 "$exe" "mpi_util::gather-GPU-src-dest-same-stencil" --deviceTypes "$OCL_DEVICE_TYPES"
      run_test --oversubscribe -n 5 "$exe" "mpi_util::gather-cuboidbs-src-dest-same-ocl" --deviceTypes "$OCL_DEVICE_TYPES"
      run_test --oversubscribe -n 5 "$exe" "mpi_util::gather-blockstencil-src-dest-same-ocl" --deviceTypes "$OCL_DEVICE_TYPES"
    fi
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
    run_test --oversubscribe -n 2 "$exe" "MPI_vcycle_seq_immediate_gather_scatter_blockstencil_size1"
    run_test --oversubscribe -n 2 "$exe" "MPI_vcycle_seq_treshold_gt_0_blockstencil_size1"
    if [ "$NO_OCL" = false ] ; then
      run_test --oversubscribe -n 1 "$exe" "MPI_vcycle_GPU_immediate_gather_scatter_Laplace7p" --deviceTypes "$OCL_DEVICE_TYPES"
      run_test --oversubscribe -n 4 "$exe" "MPI_vcycle_GPU_immediate_gather_scatter_Laplace7p" --deviceTypes "$OCL_DEVICE_TYPES"
      run_test --oversubscribe -n 1 "$exe" "MPI_vcycle_GPU_threshold_gt_0_Laplace7p" --deviceTypes "$OCL_DEVICE_TYPES"
      run_test --oversubscribe -n 4 "$exe" "MPI_vcycle_GPU_threshold_gt_0_Laplace7p" --deviceTypes "$OCL_DEVICE_TYPES"
      run_test --oversubscribe -n 1 "$exe" "MPI_vcycle_GPU_threshold_eq_1_Varying27p" --deviceTypes "$OCL_DEVICE_TYPES"
      run_test --oversubscribe -n 4 "$exe" "MPI_vcycle_GPU_threshold_eq_1_Varying27p" --deviceTypes "$OCL_DEVICE_TYPES"
      run_test --oversubscribe -n 1 "$exe" "MPI_vcycle_GPU_threshold_eq_2_Varying27p" --deviceTypes "$OCL_DEVICE_TYPES"
      run_test --oversubscribe -n 4 "$exe" "MPI_vcycle_GPU_threshold_eq_2_Varying27p" --deviceTypes "$OCL_DEVICE_TYPES"
      run_test --oversubscribe -n 4 "$exe" "MPI_vcycle_GPU_threshold_eq_2_Varying27p_multiple_jacobi_iters" --deviceTypes "$OCL_DEVICE_TYPES"
      run_test --oversubscribe -n 4 "$exe" "MPI_vcycle_GPU_FixedStencil_multiple_jacobi_iters" --deviceTypes "$OCL_DEVICE_TYPES"
      run_test --oversubscribe -n 2 "$exe" "MPI_vcycle_GPU_immediate_gather_scatter_blockstencil_size1" --deviceTypes "$OCL_DEVICE_TYPES"
      run_test --oversubscribe -n 2 "$exe" "MPI_vcycle_GPU_treshold_gt_0_blockstencil_size1" --deviceTypes "$OCL_DEVICE_TYPES"
    fi
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

    if [ "$NO_OCL" = false ] ; then
      run_test --oversubscribe -n 2 "$exe" "MPI-updateGhostsStencilOclMpi-nprocs" --deviceTypes "$OCL_DEVICE_TYPES"
      run_test --oversubscribe -n 4 "$exe" "MPI-updateGhostsStencilOclMpi-nprocs" --deviceTypes "$OCL_DEVICE_TYPES"
    fi
fi

if [ "$TEST_GALERKIN" = true ] || [ "$TEST_ALL" = true ] ; then
    echo "#######################"
    echo "Run Tests from test_mpi_galerkin.cpp ..."
    echo "#######################"

    run_test --oversubscribe -n 1 "$exe" "MPI_seq_galerkinOptimized_nprocs"
    run_test --oversubscribe -n 4 "$exe" "MPI_seq_galerkinOptimized_nprocs"
    run_test --oversubscribe -n 1 "$exe" MPI_seq_galerkin_different_thresholds
    run_test --oversubscribe -n 4 "$exe" MPI_seq_galerkin_different_thresholds

    if [ "$NO_OCL" = false ] ; then
      run_test --oversubscribe -n 1 "$exe" "MPI_GPU_galerkinOptimized_nprocs" --deviceTypes "$OCL_DEVICE_TYPES"
      run_test --oversubscribe -n 4 "$exe" "MPI_GPU_galerkinOptimized_nprocs" --deviceTypes "$OCL_DEVICE_TYPES"
    fi
fi

echo "Done. All good!"

