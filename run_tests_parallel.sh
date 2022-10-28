#!/bin/bash

# taken from bowie7070 at https://github.com/catchorg/Catch2/issues/399

# cd directory-for-testing

mkdir "tmp$$"

exe=build/tests

# don't halt on error
$exe --list-tests --order rand --rng-seed time "$@" |
    parallel --bar -j 20 $exe "{}" ">" "tmp$$/{}.out"

# halt on error
# $exe --list-tests --order rand --rng-seed time "$@" |
#     parallel --bar --halt now,fail=1 -j 20 $exe "{}" ">" "tmp$$/{}.out"
    
