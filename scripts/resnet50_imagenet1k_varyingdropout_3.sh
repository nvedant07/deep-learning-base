#!/bin/bash
#
#SBATCH --partition=a40        # Use GPU partition "a100"
#SBATCH --gres=gpu:2            # set 2 GPUs per job
#SBATCH -c 16                   # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 4-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=100GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/sbatch_logs/%x_%j.out      # File to which STDOUT will be written
#SBATCH -e /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/sbatch_logs/%x_%j.err      # File to which STDERR will be written


export EPOCHS=50
export MODEL="resnet50"
export BATCH_SIZE=256

# step lr schedule; steps are epochs
srun --jobid $SLURM_JOBID bash -c 'python -m deep-learning-base.supervised_training \
--dataset imagenet \
--transform_dataset imagenet \
--save_every 0 \
--model $MODEL \
--batch_size $BATCH_SIZE \
--wandb_name imagenet-training-scratch \
--max_epochs $EPOCHS \
--optimizer sgd \
--lr 0.01 \
--step_lr 5 \
--drop_rate 0.1'

srun --jobid $SLURM_JOBID bash -c 'python -m deep-learning-base.supervised_training \
--dataset imagenet \
--transform_dataset imagenet \
--save_every 0 \
--model $MODEL \
--batch_size $BATCH_SIZE \
--wandb_name imagenet-training-scratch \
--max_epochs $EPOCHS \
--optimizer sgd \
--lr 0.01 \
--step_lr 5 \
--drop_rate 0.2'

srun --jobid $SLURM_JOBID bash -c 'python -m deep-learning-base.supervised_training \
--dataset imagenet \
--transform_dataset imagenet \
--save_every 0 \
--model $MODEL \
--batch_size $BATCH_SIZE \
--wandb_name imagenet-training-scratch \
--max_epochs $EPOCHS \
--optimizer sgd \
--lr 0.01 \
--step_lr 5 \
--drop_rate 0.3'

srun --jobid $SLURM_JOBID bash -c 'python -m deep-learning-base.supervised_training \
--dataset imagenet \
--transform_dataset imagenet \
--save_every 0 \
--model $MODEL \
--batch_size $BATCH_SIZE \
--wandb_name imagenet-training-scratch \
--max_epochs $EPOCHS \
--optimizer sgd \
--lr 0.01 \
--step_lr 5 \
--drop_rate 0.4'

