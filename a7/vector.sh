#!/bin/bash
#SBATCH --job-name=exercise                     # the name of your job
#SBATCH --output=/scratch/gdv4/exercise.out     # this is the file your output and errors go to
#SBATCH --time=6:00                             # 20 min, shorter time, quicker start, max run time
#SBATCH --workdir=/scratch/gdv4                 # your work directory
#SBATCH --mem=1000                              # 2GB of memory

#SBATCH --gres=gpu:tesla:1
#SBATCH --qos=gpu_class

#SBATCH --constraint=p100

# load a module, for example
module load c

srun nvcc -arch=compute_60 -code=sm_60 -lcuda -Xcompiler -fopenmp  vector_add_starter.cu -o vector_add
