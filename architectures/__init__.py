import torch.nn as nn
from typing import Callable

from . import cifar_models
### all paths are added in __init__.py of the top level dir which allows for the following import
from datasets import dataset_metadata as ds
import importlib
timm = importlib.import_module('pytorch-image-models.timm')


def identity(x):
    return x

def load_pretrained_weights(model, pretrained, checkpoint_path):
    ## Function to load pretrained weights for CIFAR-like classifiers
    if pretrained:
        assert checkpoint_path, 'Must pass checkpoint_path for pretrained CIFAR models'
        timm.models.helpers.load_checkpoint(model, checkpoint_path)


def create_model(model_name: str, 
                dataset_name: str, 
                pretrained: bool = False,
                checkpoint_path: str = '',
                callback: Callable = identity) -> nn.Module:
    """
    callback: function that can be used to alter the model after creation 
            (eg: adding more classification layers)
    """
    if 'cifar' in dataset_name:
        assert model_name in cifar_models.__dict__, f'{model_name} not available for {dataset_name}'
        model = cifar_models.__dict__[model_name](num_classes=ds.DATASET_PARAMS[dataset_name]['num_classes'])
        load_pretrained_weights(model, pretrained, checkpoint_path)
    else:
        # Use timm for ImageNet and other big dataset models 
        model = timm.create_model(model_name, 
                                  num_classes=ds.DATASET_PARAMS[dataset_name]['num_classes'],
                                  pretrained=pretrained, 
                                  checkpoint_path=checkpoint_path)

    return callback(model)