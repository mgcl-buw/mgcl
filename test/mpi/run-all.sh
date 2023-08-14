#!/bin/bash

# Cancel script on first failure
set -e

# This script runs all tests in the MPI test folder.
# Run from build/test directory.
# New tests must be added manually.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
exe="$SCRIPT_DIR/tests_mpi"

echo "#######################"
echo "Run Tests from test_mpi_problem.cpp ..."
echo "#######################"

$exe "MPI Problem::setMpiComm"
$exe "MPI Problem::calculateAndSetMaxLevel (1 process)"
mpiexec --oversubscribe -n 4 "$exe" "MPI Problem::calculateAndSetMaxLevel (4 processes)"
$exe "MPI Problem::calculateAndSetMpiLevelThreshold valid (1 process)"
$exe "MPI Problem::calculateAndSetMpiLevelThreshold throwing (1 process)"

echo "#######################"
echo "Run Tests from test_mpi_level.cpp ..."
echo "#######################"

$exe "Level::initMpiData (1 process)"
mpiexec --oversubscribe -n 2 "$exe" "Level::initMpiData (2 processes)"
mpiexec --oversubscribe -n 8 "$exe" "Level::initMpiData (8 processes)"
mpiexec --oversubscribe -n 24 "$exe" "Level::initMpiData (24 processes)"

echo "#######################"
echo "Run Tests from test_mpi_ghosts.cpp ..."
echo "#######################"

$exe "MPI updateGhostsSeq (1 process)"
mpiexec --oversubscribe -n 8 "$exe" "MPI updateGhostsSeq (n processes)"
mpiexec --oversubscribe -n 8 "$exe" "MPI updateGhosts ocl (n processes)"

echo "#######################"
echo "Run Tests from test_mpi_jacobi.cpp ..."
echo "#######################"

mpiexec --oversubscribe -n 8 "$exe" "MPI jacobiSeq (n processes)"
mpiexec --oversubscribe -n 8 "$exe" "MPI jacobi ocl (n processes)"
