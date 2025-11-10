#!/bin/bash
## do not run exclusively SBATCH --exclusive
#SBATCH --job-name=mgcl_clean_build
#SBATCH --partition=gpushort
#SBATCH --account=imacm_gpu
#SBATCH -N1
#SBATCH -n1
#SBATCH --gpus 1
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 1

# remove build dir if it exists
[ -d "$HOME"/projects/mgcl/build ] && rm -rf "$HOME"/projects/mgcl/build

# create build dir
mkdir -p "$HOME"/projects/mgcl/build

# init cmake
cd "$HOME"/projects/mgcl/build || exit
cmake "$HOME"/projects/mgcl

cd "$HOME"/projects/mgcl || exit

cmake -DCMAKE_BUILD_TYPE=Release "$HOME"/projects/mgcl/build
cmake --build "$HOME"/projects/mgcl/build
