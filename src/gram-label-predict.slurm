#!/bin/bash

#SBATCH --job-name=gram-predict-class_ae-03
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=55G
##SBATCH --gres=gpu:a100:1
##SBATCH --partition=gpu

module purge
module load GCC/12.3.0  OpenMPI/4.1.5
module load matplotlib/3.7.2 
module load TensorFlow/2.13.0

jobstats &

srun -N 1 -n 1 --cpus-per-task=$SLURM_CPUS_PER_TASK python -u class-predict-label.py >> clslabel-predict_ae-03.out

jobstats
