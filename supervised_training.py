import torch
import math
from pytorch_lightning import utilities as pl_utils
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.plugins import DDPPlugin
from training import NicerModelCheckpointing, LitProgressBar
from . import architectures as arch
from architectures.callbacks import LightningWrapper
from attack.losses import DeCovLoss
from data_modules import DATA_MODULES
from training import finetuning as ft
from dataset_metadata import DATASET_PARAMS, \
    SIMCLR_TRAIN_TRANSFORMS
    # AUTOAUGMENT_TRAIN_TRANSFORMS
from functools import partial
import argparse
from pytorch_lightning.loggers import WandbLogger
import wandb

"""
run as:

python -m deep-learning-base.supervised_training \
--dataset imagenet \
--model vgg16_bn \
--seed 1 \
--batch_size 256 \
--max_epochs 100 \
--save_every 0 \
--optimizer sgd \
--lr 0.1 \
--step_lr 1000 \
--warmup_steps 200 \
--gradient_clipping 1.0
"""


parser = argparse.ArgumentParser(description='PyTorch Visual Explanation')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--transform_dataset', type=str, default=None)
parser.add_argument('--model', type=str, default='resnet18')
parser.add_argument('--seed', type=int, default=420)
parser.add_argument('--batch_size', type=int, default=None)
parser.add_argument('--max_epochs', type=int, default=None)
parser.add_argument('--save_every', type=int, default=0)
parser.add_argument('--simclr_augs', dest='simclr_augs', 
                    default=False, action='store_true')
parser.add_argument('--autoaugment', dest='autoaugment', 
                    default=False, action='store_true')
parser.add_argument('--optimizer', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--step_lr', type=float, default=None)
parser.add_argument('--warmup_steps', type=int, default=None)
parser.add_argument('--gradient_clipping', type=float, default=0.)
parser.add_argument('--use_timm_for_cifar', dest='use_timm_for_cifar', 
                    action='store_true', default=False)
parser.add_argument('--wandb_name', type=str, default=None)
parser.add_argument('--drop_rate', type=float, default=None)
parser.add_argument('--loss', type=str, default=None)
parser.add_argument('--decov_alpha', type=float, default=1.)
args = parser.parse_args()


if args.wandb_name is not None:
    # Initialize WandB
    wandb_logger = WandbLogger(project=args.wandb_name, 
                               config=wandb.helper.parse_config(args, 
                                      exclude=('max_epochs', 'save_every', 'wandb_name')))

if args.loss is not None:
    assert args.loss == 'decov', 'Only DeCov Loss supported!'
    args.decov_alpha = 10**(-args.decov_alpha)
    loss_function = DeCovLoss(alpha=args.decov_alpha)
    inference_kwargs = {'with_latent': True}
    args.loss += f'_{args.decov_alpha:.4f}'
else:
    loss_function, inference_kwargs = None, {}

def lightningmodule_callback(args):
    if args.warmup_steps is not None:
        return ft.CosineLRWrapper
    else:
        return LightningWrapper

def choose_transforms(args):
    if bool(args.simclr_augs):
        return SIMCLR_TRAIN_TRANSFORMS(DATASET_PARAMS[args.transform_dataset]['input_size'])
    if args.autoaugment:
        return AUTOAUGMENT_TRAIN_TRANSFORMS(args.dataset, 
                                            DATASET_PARAMS[args.transform_dataset]['input_size'])
    return DATASET_PARAMS[args.transform_dataset]['transform_train']


DATA_PATH_IMAGENET = '/NS/twitter_archive/work/vnanda/data'
DATA_PATH = '/NS/robustness_4/work/vnanda/data'

model = args.model
dataset = args.dataset
transform_dataset = dataset if args.transform_dataset is None else args.transform_dataset
seed = args.seed
max_epochs = DATASET_PARAMS[dataset]['epochs'] if args.max_epochs is None else args.max_epochs
devices = torch.cuda.device_count()
batch_size = args.batch_size
model_kwargs = {'drop_rate': args.drop_rate} if args.drop_rate is not None else {}


dm = DATA_MODULES[dataset](
    data_dir=DATA_PATH_IMAGENET if 'imagenet' in args.dataset else DATA_PATH,
    transform_train=choose_transforms(args),
    transform_test=DATASET_PARAMS[transform_dataset]['transform_test'],
    batch_size=batch_size)

steps_per_epoch = math.ceil(len(dm.train_dataloader())/devices)
total_steps = steps_per_epoch * max_epochs
print (f'Total Steps: {total_steps} ({steps_per_epoch} per epoch)')
m1 = arch.create_model(model, dataset, pretrained=False,
                       seed=seed, num_classes=DATASET_PARAMS[dataset]['num_classes'],
                       use_timm_for_cifar=args.use_timm_for_cifar,
                       callback=partial(lightningmodule_callback(args),
                                        dataset_name=dataset,
                                        optimizer=args.optimizer,
                                        step_lr=args.step_lr,
                                        lr=args.lr,
                                        warmup_steps=args.warmup_steps,
                                        total_steps=total_steps,
                                        training_params_dataset=dataset,
                                        loss=loss_function,
                                        inference_kwargs=inference_kwargs),
                       model_kwargs=model_kwargs)

pl_utils.seed.seed_everything(seed, workers=True)

model_name = f'{model}' + '_'.join([f'{k}_{v}' for k,v in model_kwargs.items()])
base_dirpath = f'/NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/{dataset}'
model_path = f'{model_name}_bs_{batch_size}_seed_{seed}_lr_{args.lr}_opt_{args.optimizer}_step_{args.step_lr}'\
             f'_warmup_{args.warmup_steps}_gradclip_{args.gradient_clipping}'
if args.simclr_augs:
    model_path += '_simclr_augs'
if args.autoaugment:
    model_path += '_autoaugment'
if args.loss is not None:
    model_path += f'_{args.loss}'
checkpointer = NicerModelCheckpointing(
    dirpath=f'{base_dirpath}/{model_path}', 
    filename='{epoch}', 
    every_n_epochs=args.save_every, 
    save_top_k=1, 
    save_last=False,
    verbose=True,
    mode='min', 
    monitor='val_loss')
## always use ddp for multi-GPU training -- works much faster, does not split batches
## can pass any quantity to LitProgressBar to be
## monitored during training, must be logged by the LightningModule 
## in `train_step_end` for it to be displayed during training
trainer = Trainer(accelerator='gpu', devices=devices,
                  strategy=DDPPlugin(find_unused_parameters=False) if devices > 1 else None, 
                  auto_select_gpus=True, deterministic=True,
                  logger=wandb_logger if args.wandb_name is not None else True,
                  max_epochs=max_epochs,
                  num_sanity_val_steps=0,
                  log_every_n_steps=1,
                  sync_batchnorm=True,
                  check_val_every_n_epoch=1,
                  callbacks=[LitProgressBar(['loss', 'running_train_acc', 'running_val_acc']), 
                             checkpointer])
output = trainer.fit(m1, datamodule=dm)
if trainer.is_global_zero:
    if args.wandb_name is not None:
        wandb_logger.log_metrics({'final_val_acc': m1.final_val_acc, 
                          'final_test_acc': m1.final_test_acc},
                          trainer.global_step)
