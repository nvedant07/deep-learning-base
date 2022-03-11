import torch.nn as nn
from pytorch_lightning import utilities as pl_utils
from typing import Callable

from . import cifar_models
### all paths are added in __init__.py of the top level dir which allows for the following import
from datasets import dataset_metadata as ds
import timm
import timm.models as models


def list_models(dataset_name):
    if 'cifar' in dataset_name:
        return cifar_models.model_names
    else:
        return models.list_models()

def identity(x):
    return x

def load_pretrained_weights(model, pretrained, checkpoint_path):
    if pretrained:
        assert checkpoint_path, 'Must pass checkpoint_path for pretrained CIFAR models'
        timm.models.helpers.load_checkpoint(model, checkpoint_path)

def create_model(model_name: str, 
                dataset_name: str, 
                pretrained: bool = False,
                checkpoint_path: str = '',
                seed: int = 0, 
                loading_function: Callable = load_pretrained_weights, 
                callback: Callable = identity) -> nn.Module:
    """
    callback: function that can be used to alter the model after creation 
            (eg: adding more classification layers)
    loading_function: user-defined function that defines the logic for loading weights
            from a user-defined checkpoint file. Must take 3 args: model, pretrained, checkpoint_path
    """
    pl_utils.seed.seed_everything(seed)
    
    if dataset_name not in ds.DATASET_PARAMS:    dataset_name = 'imagenet'

    if 'cifar' in dataset_name:
        assert model_name in cifar_models.model_names, f'{model_name} not available for {dataset_name}'
        model = cifar_models.create_model_fn(model_name)(num_classes=ds.DATASET_PARAMS[dataset_name]['num_classes'])
        loading_function(model, pretrained, checkpoint_path)
    else:
        # Use timm for ImageNet and other big dataset models 
        should_custom_load = loading_function != load_pretrained_weights and pretrained and checkpoint_path
        model = timm.create_model(model_name, 
                                  num_classes=ds.DATASET_PARAMS[dataset_name]['num_classes'],
                                  pretrained=False if should_custom_load else pretrained, 
                                  checkpoint_path='' if should_custom_load else checkpoint_path)
        if should_custom_load:    loading_function(model, pretrained, checkpoint_path)

    return callback(model)