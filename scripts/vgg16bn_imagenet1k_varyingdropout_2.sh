#!/bin/bash
#
#SBATCH --gres=gpu:2            # set 2 GPUs per job
#SBATCH -c 16                   # Number of cores
#SBATCH -N 1                    # Ensure that all cores are on one machine
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per dist per node!
#SBATCH -t 2-00:00              # Maximum run-time in D-HH:MM
#SBATCH --mem=100GB               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/sbatch_logs/%x_%j.out      # File to which STDOUT will be written
#SBATCH -e /NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/sbatch_logs/%x_%j.err      # File to which STDERR will be written


EPOCHS=50
MODEL="vgg16_bn"
BATCH_SIZE=256

# cosine lr schedule with longer warmup; bigger learning rates
# python -m deep-learning-base.supervised_training \
# --dataset imagenet \
# --transform_dataset imagenet \
# --save_every 0 \
# --model $MODEL \
# --batch_size $BATCH_SIZE \
# --wandb_name imagenet-training-scratch \
# --max_epochs $EPOCHS \
# --optimizer sgd \
# --lr 0.01 \
# --step_lr 500 \
# --warmup_steps 1000 \
# --gradient_clipping 1.0 \
# --drop_rate 0.5

python -m deep-learning-base.supervised_training \
--dataset imagenet \
--transform_dataset imagenet \
--save_every 0 \
--model $MODEL \
--batch_size $BATCH_SIZE \
--wandb_name imagenet-training-scratch \
--max_epochs $EPOCHS \
--optimizer sgd \
--lr 0.01 \
--step_lr 500 \
--warmup_steps 1000 \
--gradient_clipping 1.0 \
--drop_rate 0.6

python -m deep-learning-base.supervised_training \
--dataset imagenet \
--transform_dataset imagenet \
--save_every 0 \
--model $MODEL \
--batch_size $BATCH_SIZE \
--wandb_name imagenet-training-scratch \
--max_epochs $EPOCHS \
--optimizer sgd \
--lr 0.01 \
--step_lr 500 \
--warmup_steps 1000 \
--gradient_clipping 1.0 \
--drop_rate 0.7

python -m deep-learning-base.supervised_training \
--dataset imagenet \
--transform_dataset imagenet \
--save_every 0 \
--model $MODEL \
--batch_size $BATCH_SIZE \
--wandb_name imagenet-training-scratch \
--max_epochs $EPOCHS \
--optimizer sgd \
--lr 0.01 \
--step_lr 500 \
--warmup_steps 1000 \
--gradient_clipping 1.0 \
--drop_rate 0.8