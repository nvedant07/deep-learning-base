# quick testing script
from pytorch_lightning import utilities as pl_utils
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.plugins import DDPPlugin
from training import NicerModelCheckpointing, LitProgressBar
import sys
import architectures as arch
from attack.callbacks import AdvCallback
from architectures.callbacks import LightningWrapper, AdvAttackWrapper
from datasets.data_modules import DATA_MODULES
from datasets.dataset_metadata import DATASET_PARAMS
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
seed = 420
devices = 1

imagenet_path = '/NS/twitter_archive/work/vnanda/data'
data_path = '/NS/twitter_archive2/work/vnanda/data'

m1 = arch.create_model(model, dataset, pretrained=pretrained, 
                       checkpoint_path=checkpoint_path, seed=seed, 
                       callback=partial(AdvAttackWrapper, dataset_name=dataset))
dm = DATA_MODULES[dataset](
    data_dir=imagenet_path if dataset == 'imagenet' else data_path)

pl_utils.seed.seed_everything(seed, workers=True)

checkpointer = NicerModelCheckpointing(dirpath=f'checkpoints/{dataset}/{model}', 
                               filename='{epoch}', 
                               every_n_epochs=5, 
                               save_top_k=5, 
                               save_last=False,
                               verbose=True,
                               mode='min', # change to max if accuracy is being monitored
                               monitor='val_loss_adv')
adv_callback = AdvCallback(constraint_train='2',
                           eps_train=1.,
                           step_size=1.,
                           iterations_train=1,
                           iterations_val=10,
                           iterations_test=10,
                           random_start_train=False,
                           random_restarts_train=0,
                           return_image=True)
trainer = Trainer(accelerator='gpu', devices=devices,
                  strategy=DDPPlugin(find_unused_parameters=False) if devices > 1 else None, 
                  auto_select_gpus=True, deterministic=True,
                  max_epochs=DATASET_PARAMS[dataset]['epochs'],
                  check_val_every_n_epoch=1,
                  num_sanity_val_steps=0,
                  callbacks=[LitProgressBar(['running_acc_clean', 'running_acc_adv']),
                             checkpointer,
                             adv_callback])
## always use ddp -- works much faster, does not split batches
trainer.validate(m1, datamodule=dm)
acc = trainer.test(m1, datamodule=dm)
print (f'Accuracy: {acc[0]["test_acc1"]}')
