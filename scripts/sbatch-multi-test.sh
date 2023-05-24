#!/bin/bash
#
#SBATCH -p a40
#SBATCH --gres=gpu:1
#SBATCH -c 8                   # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=2     
#SBATCH -a 2-5
#SBATCH -t 2-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=100GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/sbatch_logs/%x_%j.out      # File to which STDOUT will be written
#SBATCH -e /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/sbatch_logs/%x_%j.err      # File to which STDERR will be written

ALL_VALUES=(1005 0.05 0.5)

srun --jobid $SLURM_JOBID bash -c 'echo ${SLURM_ARRAY_TASK_ID}; echo ${ALL_VALUES[0]}'
