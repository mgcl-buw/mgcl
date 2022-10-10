#!/bin/bash
#SBATCH --exclusive
#SBATCH --job-name=mgcl_build
#SBATCH --partition=gpu
#SBATCH --account=imacm_gpu
#SBATCH -N1
#SBATCH -n1
#SBATCH --gpus 1
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 1

# cd ~/output/ || exit

cmake -DCMAKE_BUILD_TYPE=Release "$HOME"/projects/mgcl/build
cmake --build "$HOME"/projects/mgcl/build --clean-first

# cd "$HOME"/projects/mgcl || exit
