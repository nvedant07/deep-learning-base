EPOCHS=50
MODEL="vit_small_patch32_224"
BATCH_SIZE=256

# step lr schedule; steps are epochs
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
--step_lr 5 \
--drop_rate 0.1

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
--step_lr 5 \
--drop_rate 0.2

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
--step_lr 5 \
--drop_rate 0.3

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
--step_lr 5 \
--drop_rate 0.4

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
--step_lr 5 \
--drop_rate 0.5

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
--step_lr 5 \
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
--step_lr 5 \
--drop_rate 0.7

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
# --step_lr 5 \
# --drop_rate 0.8
