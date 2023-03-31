#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=mgcl_bench
#SBATCH --partition=gpu
#SBATCH --account=imacm_gpu
#SBATCH -N1
#SBATCH -n1
#SBATCH --gpus 1
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 1

current_time=$(date "+%Y.%m.%d-%H.%M.%S")

cd "$HOME"/projects/mgcl/build || exit

# run e.g. with 
# sbatch run_benchmarks.sh --chdir /beegfs/shoffmann/projects/mgcl/build/ [solve][console]
"$HOME"/projects/mgcl/build/benchmarks/benchmarks "$@" > ~/output/"$current_time"-mgcl-bench.txt

cd "$HOME"/projects/mgcl || exit
