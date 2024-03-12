#!/bin/bash
## not running exclusively #SBATCH --exclusive
#SBATCH --job-name=mgcl_build
#SBATCH --partition=gpu
#SBATCH --account=imacm_gpu
#SBATCH -N1
#SBATCH -n1
#SBATCH --gpus 1
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 1

# cd ~/output/ || exit
dir_build="$HOME/projects/mgcl/buildDebug"
dir_mgcl="$HOME/projects/mgcl"

# create build dir if it doesn't exist
[ ! -d "$dir_build" ] && mkdir -p "$dir_build"

# init cmake
cd "$dir_build" || exit
cmake -DCMAKE_BUILD_TYPE=Debug "$dir_mgcl"
cd "$dir_mgcl" || exit

cmake --build "$dir_build"
# cmake --install "$dir_build"/ --prefix "$HOME"/mgcl 
# cd "$dir_mgcl"/ || exit
