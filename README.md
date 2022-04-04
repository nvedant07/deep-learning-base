This is a directory for common DL code including models, adversarial attacks, supervised training. This is provides easy ways to extend models and training paradigms. This provides a smooth interface between many SOTA libraries like ``timm``, ``robustness``, ``vissl``. All the code relies on PyTorch and PyTorch-Lightning.

## Models

ImageNet (and similarly large) models are taken from [timm](https://github.com/rwightman/pytorch-image-models). 
CIFAR10/100 model schemas differ a little from ImageNet models.

Example of initializing a (random) model (in case dataset isn't specified in ``datasets.dataset_metadata``, initializes an ImageNet model):

```python
import architectures as arch

dataset = 'cifar10'
model = 'resnet18'

m = arch.create_model(model, dataset)
# m has randomly initialized weights with default seed
```

Example of initializing a (random) model with a different seed:

```python
import architectures as arch

dataset = 'cifar10'
model = 'resnet18'
seed = 42

m = arch.create_model(model, dataset, seed=seed)
# m has randomly initialized weights with seed=42
```

Initializing with user-defined pre-trained weights:

```python
import architectures as arch

dataset = 'imagenet'
model = 'resnet18'
pretrained = True
checkpoint_path = './checkpoint.best.pt'

m = arch.create_model(model, dataset, pretrained=pretrained, checkpoint_path=checkpoint_path)
# m takes weights from specified path
```

Initializing with pre-trained weights in ``timm`` (ImageNet only):

```python
import architectures as arch

dataset = 'cifar10'
model = 'resnet18'
pretrained = True

m = arch.create_model(model, dataset, pretrained=pretrained)
# m has pretrained weights as defined in timm
```

To perform inference (using PyTorch Lightning):

```python

from datasets.data_modules import DATA_MODULES
from architectures.callbacks import LightningWrapper
from functools import partial

dataset = 'imagenet'
model = 'resnet50'
devices = 2

m = arch.create_model(model, dataset, pretrained=pretrained, 
                      callback=partial(LightningWrapper, dataset_name=dataset))
dm = DATA_MODULES[dataset](data_dir='./data')

trainer = Trainer(accelerator='gpu', devices=devices,
                  strategy=DDPPlugin(find_unused_parameters=False) if devices > 1 else None, 
                  auto_select_gpus=True, deterministic=True)
acc = trainer.test(m1, datamodule=dm)
```


To get intermediate layer representations:

```python
# show an example of obtaining feature maps of ResNet
```


## Datasets

Standard dataloaders from torchvision + support for custom datasets. Many dataset classes taken from [robustness](https://github.com/MadryLab/robustness) and [robust-models-transfer](https://github.com/Microsoft/robust-models-transfer)

Initializing datasets:

```python

from datasets.data_modules import DATA_MODULES

dataset = 'imagenet'

dm = DATA_MODULES[dataset](data_dir='./data')
# dm is an instance of pl.LightningDataModule

# to access training data
for x, y in dm.train_dataloader():
    ...

# to access validation data
for x, y in dm.val_dataloader():
    ...

# to access test data
for x, y in dm.test_dataloader():
    ...

```


Datasets with custom data augmentations:

```python
# show the use of data augmentation callback
```


Model dependent data augmentations:

```python
# example of model dependent data aug, eg: worst-of-K spatial augmentations
```


## Training

Supports different losses (eg: adversarial training), different optimizers (all included in ``timm``)

Supervised: Uses [PyTorch-Lightning](https://github.com/facebookresearch/vissl) for easy, scalable training. 

Self-supervised: Uses [vissl](https://github.com/facebookresearch/vissl) to track SOTA models and weights.

Uses [robustness](https://github.com/MadryLab/robustness) for attack module used in adversarial training.

Example of supervised training (standard):

```python
from pytorch_lightning import utilities as pl_utils
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.plugins import DDPPlugin
from training import NicerModelCheckpointing, LitProgressBar
import architectures as arch
from architectures.callbacks import LightningWrapper
from datasets.data_modules import DATA_MODULES
from datasets.dataset_metadata import DATASET_PARAMS
from functools import partial

model = 'resnet50'
seed = 420
devices = 1

m1 = arch.create_model(model, dataset, pretrained=False, 
                       checkpoint_path='', seed=seed, 
                       callback=partial(LightningWrapper, dataset_name='cifar10'))
dm = DATA_MODULES['cifar10'](data_dir='./data')

pl_utils.seed.seed_everything(seed, workers=True)

checkpointer = NicerModelCheckpointing(dirpath=f'checkpoints/{dataset}/{model}', 
                               filename='{epoch}', 
                               every_n_epochs=5, 
                               save_top_k=5, 
                               save_last=False,
                               verbose=True,
                               mode='min', 
                               monitor='val_loss')
## always use ddp for multi-GPU training -- works much faster, does not split batches
trainer = Trainer(accelerator='gpu', devices=devices,
                  strategy=DDPPlugin(find_unused_parameters=False) if devices > 1 else None, 
                  auto_select_gpus=True, deterministic=True,
                  max_epochs=DATASET_PARAMS[dataset]['epochs'],
                  check_val_every_n_epoch=1,
                  callbacks=[LitProgressBar(['running_acc']), # can pass any quantity to be monitored during training, must be logged by the LightningModule in `train_step_end` for it to be displayed during training
                             checkpointer])
trainer.fit(m1, datamodule=dm)

```


Example of supervised training (adversarial):

```python
# show an example of adversarial training
```

Example of self-supervised learning (SimCLR):

```python
# show an example of SimCLR from vissl
```

Example of self-supervised learning (BYOL):

```python
# show an example of BYOL from vissl
```


## Adversarial Attacks

Attack module in [robustness](https://github.com/MadryLab/robustness). Also includes spatial attacks, taken from [adversarial_spatial](https://github.com/MadryLab/adversarial_spatial).

To do adversarial attack on models, wrap them in the ``Attack`` module. Here's an example:

```python
# Show how to do L_inf PGD attack on an ImageNet ResNet
```


To do spatial attacks:

```python
# Show an example of worst-of-K and first order spatial attacks
```

## Unit Tests

To run unit tests:

```bash
cd tests
pytest -c conftest.py \
--data_path="../../data" \
--imagenet_path="/NS/twitter_archive/work/vnanda/data"
```

Replace the paths with the actual paths