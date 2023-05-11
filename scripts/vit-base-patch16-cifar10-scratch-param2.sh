EPOCHS=100
MODEL="vit_base_patch16_224"

for bs in {128,256,512}
do

# step lr schedule; steps are epochs
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
--lr 0.001 \
--step_lr 10

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
--step_lr 10

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
--lr 0.1 \
--step_lr 10

done