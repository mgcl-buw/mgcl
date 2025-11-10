#!/bin/bash
## not running exclusively #SBATCH --exclusive
#SBATCH --job-name=mgcl_build
#SBATCH --partition=gpushort
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
cmake -DCMAKE_BUILD_TYPE=Release "$HOME"/projects/mgcl
cd "$HOME"/projects/mgcl || exit

cmake --build "$HOME"/projects/mgcl/build
# cmake --install "$HOME"/projects/mgcl/build --prefix "$HOME"/mgcl 
# cd "$HOME"/projects/mgcl || exit
