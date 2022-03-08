import pytest
import torch
from torch.nn import Linear, Conv2d
from functools import partial
import architectures as arch
from datasets import dataset_metadata as ds
from datasets.data_modules import DATA_MODULES
from architectures.callbacks import LightningWrapper
from pytorch_lightning.trainer.trainer import Trainer

@pytest.mark.parametrize('dataset_name', ['cifar10', 'imagenet'])
def test_random_inits(dataset_name):
    seed1, seed2 = 0, 1
    for model_name in arch.list_models(dataset_name):
        models, diff = [], []
        for s in [seed1, seed2]:
            models.append(arch.create_model(model_name, dataset_name, seed=s))
        for (m1name, m1), (m2name, m2) in zip(models[0].named_modules(), models[1].named_modules()):
            if hasattr(m1, 'weight') and (isinstance(m1, Linear) or isinstance(m1, Conv2d)):
                diff.append(not torch.all(m1.weight == m2.weight))
        ## some attention heads have same weights regardless of init, 
        ## however some layers should have different weights as a result of different seeds
        assert torch.sum(diff) > 0, f'For {model_name}, all modules had same weights!'

        # models initialized with same seed should have exact same initial weights
        models = []
        for s in [seed1, seed1]:
            models.append(arch.create_model(model_name, dataset_name, seed=s))
        for (m1name, m1), (m2name, m2) in zip(models[0].named_modules(), models[1].named_modules()):
            if hasattr(m1, 'weight') and (isinstance(m1, Linear) or isinstance(m1, Conv2d)):
                assert torch.all(m1.weight == m2.weight), \
                    f'{model_name}\n\n{m1name}: {m1}\n{m2name}: {m2}\n{m1.weight == m2.weight}'


@pytest.mark.parametrize('dataset_name,model_name,weights_path', 
                        [('cifar10', 'resnet18', './weights/resnet18_cifar.pt'), 
                         ('imagenet', 'resnet18', './weights/resnet18_l2eps3_imagenet.pt')])
def test_load_weights(dataset_name, model_name, weights_path):
    m1 = arch.create_model(model_name, dataset_name, checkpoint_path=weights_path, 
        callback=partial(LightningWrapper, dataset_name=dataset_name))
    trainer = Trainer()
    acc = trainer.test(m1, datamodule=)['acc']