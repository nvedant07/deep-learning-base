# quick testing script
from pytorch_lightning import utilities as pl_utils
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.plugins import DDPPlugin
from torchvision.transforms import transforms
from training import NicerModelCheckpointing, LitProgressBar
import torch
import architectures as arch
from attack.callbacks import AdvCallback
from architectures.callbacks import LightningWrapper, AdvAttackWrapper, LinearEvalWrapper
from architectures.inference import inference_with_features
from datasets.data_modules import DATA_MODULES
from datasets.dataset_metadata import DATASET_PARAMS
from ssl.simclr_datamodule import simclr_dm
from ssl.simclr_callback import SimCLRWrapper
from functools import partial


dataset = 'cifar10'
# dataset = 'imagenet'

# model = 'vgg16'
model = 'resnet18'
# model = 'resnet50'

# checkpoint_path = '/NS/robustness_1/work/vnanda/adv-trades/checkpoints'\
#                   '/resnet18/beta_1.0_eps_1.000/checkpoint.pt.best'
# checkpoint_path = './tests/weights/resnet18_cifar10.pt'
# checkpoint_path = './tests/weights/vgg16_cifar10.pt'
# checkpoint_path = './tests/weights/resnet18_l2eps3_imagenet.pt'
checkpoint_path = ''

pretrained = False
seed = 2
devices = 1
num_nodes = 1
strategy = DDPPlugin(find_unused_parameters=True) if devices > 1 else None
max_epochs = 100
# max_epochs = DATASET_PARAMS[dataset]['epochs']

imagenet_path = '/NS/twitter_archive/work/vnanda/data'
data_path = '/NS/twitter_archive2/work/vnanda/data'

dm = DATA_MODULES[dataset](
    data_dir=imagenet_path if dataset == 'imagenet' else data_path,
    val_frac=0.)

# send SimCLRWrapper to model
m1 = arch.create_model(model, dataset, pretrained=pretrained,
                       checkpoint_path=checkpoint_path, seed=seed,
                       callback=partial(LinearEvalWrapper,
                                        DATASET_PARAMS[dataset]['epochs'],
                                        dataset_name=dataset))

pl_utils.seed.seed_everything(seed, workers=True)

checkpointer = NicerModelCheckpointing(
                               save_partial=[f'model.{x[0]}' for x in list(m1.model.named_parameters())[-2:]],
                               dirpath=f'/NS/robustness_2/work/vnanda/deep_learning_base/checkpoints/{dataset}/{model}/simclr_bs_1024', 
                               filename='rand_init_linear_eval' + f'_{seed}', 
                               save_top_k=1,
                               every_n_epochs=0,
                               save_last=False,
                               verbose=True,
                               mode='max', # change to max if accuracy is being monitored
                               monitor='val_acc1')
# adv_callback = AdvCallback(constraint_train='2',
#                            eps_train=1.,
#                            step_size=1.,
#                            iterations_train=1,
#                            iterations_val=10,
#                            iterations_test=10,
#                            random_start_train=False,
#                            random_restarts_train=0,
#                            return_image=True)
trainer = Trainer(accelerator='gpu', devices=devices,
                  num_nodes=num_nodes,
                  strategy=strategy, 
                  log_every_n_steps=1,
                  auto_select_gpus=True, deterministic=True,
                  max_epochs=max_epochs,
                  check_val_every_n_epoch=1,
                  limit_val_batches=.5,
                  num_sanity_val_steps=0,
                  callbacks=[LitProgressBar(['loss', 'train_acc1']),
                             checkpointer])
# ## always use ddp -- works much faster, does not split batches
trainer.fit(m1, datamodule=dm)
# acc = trainer.test(m1, datamodule=dm)
# print (f'Accuracy: {acc[0]["test_acc1"]}')
