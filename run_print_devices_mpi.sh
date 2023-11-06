#!/bin/bash
## #SBATCH --exclusive
#SBATCH --job-name=mgcl_print_devices
#SBATCH --partition=gpu
#SBATCH --account=imacm_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus=2
#SBATCH --gpus-per-task=1

## srun -n 2 --partition=gpu --account=imacm_gpu --nodes=1 --gpus=2 --gpus-per-task=1 -w gpu21004  build/examples/example_print_devices

current_time=$(date "+%Y.%m.%d-%H.%M.%S")

EX_DIR="$HOME"/projects/mgcl/build/examples

cd "$EX_DIR" || exit

# run e.g. with 
# sbatch --chdir /beegfs/shoffmann/projects/mgcl/build/ run_benchmarks.sh [solve][console]
srun "$EX_DIR"/example_print_devices "$@" > ~/output/"$current_time"-mgcl-ex-print-devices.txt 2>&1

cd "$HOME"/projects/mgcl || exit
