EPOCHS=50
MODEL="resnet50"
BATCH_SIZE=64


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
--loss decov

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
# --loss decov \
# --decov_alpha 0.01