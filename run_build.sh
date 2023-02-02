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

# create build dir if it doesn't exist
[ ! -d "$HOME"/projects/mgcl/build ] && mkdir -p "$HOME"/projects/mgcl/build

# init cmake
cd "$HOME"/projects/mgcl/build || exit
cmake "$HOME"/projects/mgcl

cd "$HOME"/projects/mgcl || exit

cmake -DCMAKE_BUILD_TYPE=Release "$HOME"/projects/mgcl/build
cmake --build "$HOME"/projects/mgcl/build

# cd "$HOME"/projects/mgcl || exit
