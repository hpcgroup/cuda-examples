#!/bin/bash
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -o cuda-example-%A.out
#SBATCH -J cuda-examples
#SBATCH -t 00:03:00

source ./utilities.bash

module load cuda

echo "running job..."
START="$(timestamp)"

printf "\nRunning test-cuda\n"
./test-cuda/test-cuda

printf "\nRunning cuda-memory\n"
./cuda-memory/cuda-memory

printf "\nRunning saxpy-single-block\n"
./saxpy/saxpy-single-block

printf "\nRunning saxpy\n"
./saxpy/saxpy

printf "\nRunning matrix-add\n"
./matrix-add/matrix-add

printf "\nRunning matmul-simple\n"
./matmul/matmul-simple

printf "\nRunning matmul\n"
./matmul/matmul

END="$(timestamp)"
DURATION="$(diff_minutes ${START} ${END})"
echo "run took ${DURATION} minutes."
