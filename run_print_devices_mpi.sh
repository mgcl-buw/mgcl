#!/bin/bash
## #SBATCH --exclusive
#SBATCH --job-name=mgcl_print_devices
#SBATCH --partition=gpu
#SBATCH --account=imacm_gpu
#SBATCH --nodes=1
## #SBATCH --ntasks-per-node=4
## #SBATCH --gpus 4
#SBATCH --gres=gpu:4            # number of GPUs per node (gres=gpu:N)
## #SBATCH --gpus-per-task 1
## #SBATCH --cpus-per-task 4

current_time=$(date "+%Y.%m.%d-%H.%M.%S")

EX_DIR="$HOME"/projects/mgcl/build/examples

cd "$EX_DIR" || exit

# run e.g. with 
# sbatch --chdir /beegfs/shoffmann/projects/mgcl/build/ run_benchmarks.sh [solve][console]
mpiexec -n 4 "$EX_DIR"/example_print_devices "$@" > ~/output/"$current_time"-mgcl-ex-print-devices.txt 2>&1

cd "$HOME"/projects/mgcl || exit
