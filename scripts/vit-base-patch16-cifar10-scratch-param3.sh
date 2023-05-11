EPOCHS=100
MODEL="vit_base_patch16_224"

for bs in {128,256,512}
do

# cosine lr schedule with longer warmup; bigger learning rates
python -m deep-learning-base.supervised_training \
--dataset cifar10 \
--transform_dataset imagenet \
--save_every 0 \
--model $MODEL \
--use_timm_for_cifar \
--batch_size $bs \
--wandb_name vits-scratch \
--max_epochs $EPOCHS \
--optimizer sgd \
--lr 0.01 \
--step_lr 500 \
--warmup_steps 1000 \
--gradient_clipping 1.0

python -m deep-learning-base.supervised_training \
--dataset cifar10 \
--transform_dataset imagenet \
--save_every 0 \
--model $MODEL \
--use_timm_for_cifar \
--batch_size $bs \
--wandb_name vits-scratch \
--max_epochs $EPOCHS \
--optimizer sgd \
--lr 0.01 \
--step_lr 500 \
--warmup_steps 2000 \
--gradient_clipping 1.0

python -m deep-learning-base.supervised_training \
--dataset cifar10 \
--transform_dataset imagenet \
--save_every 0 \
--model $MODEL \
--use_timm_for_cifar \
--batch_size $bs \
--wandb_name vits-scratch \
--max_epochs $EPOCHS \
--optimizer sgd \
--lr 0.01 \
--step_lr 500 \
--warmup_steps 3000 \
--gradient_clipping 1.0

done