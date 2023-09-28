#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=mgcl_bench
#SBATCH --partition=gpu
#SBATCH --account=imacm_gpu
#SBATCH -N1
#SBATCH -n64
#SBATCH --gpus 1
# #SBATCH --gpus-per-task 1
# #SBATCH --cpus-per-task 64

current_time=$(date "+%Y.%m.%d-%H.%M.%S")

cd "$HOME"/projects/mgcl/build/benchmarks || exit

# run e.g. with 
# sbatch --chdir /beegfs/shoffmann/projects/mgcl/build/ run_benchmarks.sh [solve][console]
mpiexec -n 64 "$HOME"/projects/mgcl/build/benchmarks/benchmarks "$@" > ~/output/"$current_time"-mgcl-bench-1-64.txt

cd "$HOME"/projects/mgcl || exit
