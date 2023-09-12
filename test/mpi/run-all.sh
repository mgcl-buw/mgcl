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
  echo "-g,--ghosts: Run Ghost-Update specific tests"
  echo "-j,--jacobi: Run Jacobi specific tests"
  echo "-u,--util: Run Utility specific tests"
  echo "-v,--vcycle: Run Vcycle specific tests"
  echo "-s,--stencil: Run Stencil specific tests"
  echo "If no option is given, all tests will be run."
}

TEST_ALL=true
TEST_PROBLEM=false
TEST_LEVEL=false
TEST_GHOSTS=false
TEST_JACOBI=false
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

    run_test -n 1 "$exe" "MPI Problem::setMpiComm"
    run_test -n 1 "$exe" "MPI Problem::calculateAndSetMaxLevel (1 process)"
    run_test --oversubscribe -n 4 "$exe" "MPI Problem::calculateAndSetMaxLevel (4 processes)"
    run_test -n 1 "$exe" "MPI Problem::calculateAndSetMpiLevelThreshold valid (1 process)"
    run_test -n 1 "$exe" "MPI Problem::calculateAndSetMpiLevelThreshold throwing (1 process)"
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

    run_test --oversubscribe -n 8 "$exe" "MPI jacobiSeq (n processes)"
    run_test --oversubscribe -n 8 "$exe" "MPI jacobi ocl (n processes)"
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
fi

if [ "$TEST_VCYCLE" = true ] || [ "$TEST_ALL" = true ] ; then
    echo "#######################"
    echo "Run Tests from test_mpi_vcycle.cpp ..."
    echo "#######################"

    run_test --oversubscribe -n 1 "$exe" "MPI_vcycle_immediate_gather_scatter"
    run_test --oversubscribe -n 2 "$exe" "MPI_vcycle_immediate_gather_scatter"
    run_test --oversubscribe -n 4 "$exe" "MPI_vcycle_threshold_gt_0"
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
fi

echo "Done. All good!"

