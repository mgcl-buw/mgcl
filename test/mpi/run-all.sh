#!/bin/bash

# Cancel script on first failure
set -e

# This script runs all tests in the MPI test folder.
# Run from build/test directory.
# New tests must be added manually.

TEST_ALL=true
TEST_PROBLEM=false
TEST_LEVEL=false
TEST_GHOSTS=false
TEST_JACOBI=false
TEST_UTIL=false
TEST_VCYCLE=false

while [[ $# -gt 0 ]]; do
  case $1 in
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
  esac
done

if [ "$TEST_ALL" = true ] ; then
    echo "Running all tests..."
    echo ""
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
exe="$SCRIPT_DIR/tests_mpi"

if [ "$TEST_PROBLEM" = true ] ; then
    echo "#######################"
    echo "Run Tests from test_mpi_problem.cpp ..."
    echo "#######################"

    $exe "MPI Problem::setMpiComm"
    $exe "MPI Problem::calculateAndSetMaxLevel (1 process)"
    mpiexec --oversubscribe -n 4 "$exe" "MPI Problem::calculateAndSetMaxLevel (4 processes)"
    $exe "MPI Problem::calculateAndSetMpiLevelThreshold valid (1 process)"
    $exe "MPI Problem::calculateAndSetMpiLevelThreshold throwing (1 process)"
fi

if [ "$TEST_LEVEL" = true ] ; then
    echo "#######################"
    echo "Run Tests from test_mpi_level.cpp ..."
    echo "#######################"

    $exe "Level::initMpiData (1 process)"
    mpiexec --oversubscribe -n 2 "$exe" "Level::initMpiData (2 processes)"
    mpiexec --oversubscribe -n 8 "$exe" "Level::initMpiData (8 processes)"
    mpiexec --oversubscribe -n 24 "$exe" "Level::initMpiData (24 processes)"
fi

if [ "$TEST_GHOSTS" = true ] ; then
    echo "#######################"
    echo "Run Tests from test_mpi_ghosts.cpp ..."
    echo "#######################"

    $exe "MPI updateGhostsSeq (1 process)"
    mpiexec --oversubscribe -n 8 "$exe" "MPI updateGhostsSeq (n processes)"
    mpiexec --oversubscribe -n 8 "$exe" "MPI updateGhosts ocl (n processes)"
fi

if [ "$TEST_JACOBI" = true ] ; then
    echo "#######################"
    echo "Run Tests from test_mpi_jacobi.cpp ..."
    echo "#######################"

    mpiexec --oversubscribe -n 8 "$exe" "MPI jacobiSeq (n processes)"
    mpiexec --oversubscribe -n 8 "$exe" "MPI jacobi ocl (n processes)"
fi

if [ "$TEST_UTIL" = true ] ; then
    echo "#######################"
    echo "Run Tests from test_mpi_util.cpp ..."
    echo "#######################"

    mpiexec --oversubscribe -n 4 "$exe" "MPI_Gatherv src dest different"
    mpiexec --oversubscribe -n 4 "$exe" "MPI_Gatherv src dest same"
    mpiexec --oversubscribe -n 4 "$exe" "mpi_util::scatter src dest same"
fi

if [ "$TEST_VCYCLE" = true ] ; then
    echo "#######################"
    echo "Run Tests from test_mpi_vcycle.cpp ..."
    echo "#######################"

    mpiexec --oversubscribe -n 1 "$exe" "MPI_vcycle_immediate_gather_scatter"
    mpiexec --oversubscribe -n 2 "$exe" "MPI_vcycle_immediate_gather_scatter"
fi

