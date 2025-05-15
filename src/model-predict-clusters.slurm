#!/bin/bash

#SBATCH --job-name=gram-autoencoder-clusters
#SBATCH --time=00:35:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=100G

module purge
module load GCC/12.3.0 OpenMPI/4.1.5
module load scikit-learn/1.3.1
module load matplotlib/3.7.2
module load numba/0.58.1

jobstats &

srun -N 1 -n 1 --cpus-per-task=$SLURM_CPUS_PER_TASK python -u cluster-models-lc01.py >> gram-model-clusters.out

jobstats
