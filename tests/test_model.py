import pytest
import torch
from torch.nn import Linear, Conv2d
import architectures as arch
from datasets import dataset_metadata as ds

@pytest.mark.parametrize('dataset_name', ['cifar10', 'imagenet'])
def test_random_inits(dataset_name):
    seed1, seed2 = 0, 1
    for model_name in arch.list_models(dataset_name):
        models = []
        for s in [seed1, seed2]:
            models.append(arch.create_model(model_name, dataset_name, seed=s))
        for (m1name, m1), (m2name, m2) in zip(models[0].named_modules(), models[1].named_modules()):
            if hasattr(m1, 'weight') and (isinstance(m1, Linear) or isinstance(m1, Conv2d)):
                assert not torch.all(m1.weight == m2.weight), \
                    f'{model_name}\n\n{m1name}: {m1}\n{m2name}: {m2}\n{m1.weight == m2.weight}'
        models = []
        for s in [seed1, seed1]:
            models.append(arch.create_model(model_name, dataset_name, seed=s))
        for (m1name, m1), (m2name, m2) in zip(models[0].named_modules(), models[1].named_modules()):
            if hasattr(m1, 'weight') and (isinstance(m1, Linear) or isinstance(m1, Conv2d)):
                assert torch.all(m1.weight == m2.weight), \
                    f'{model_name}\n\n{m1name}: {m1}\n{m2name}: {m2}\n{m1.weight == m2.weight}'
