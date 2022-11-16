import torch
import torch.nn as nn
from pytorch_lightning import utilities as pl_utils
from typing import Callable, Optional, Dict
from collections import OrderedDict
import logging, os

import timm
import timm.models as models
from . import timm_addons
from . import cifar_models
from . import utils
import clip
### all paths are added in __init__.py of the top level dir which allows for the following import
import dataset_metadata as ds

_logger = logging.getLogger(__name__)

CLIP_MODEL_PATHS = {
    'vit_base_patch32_224': '/NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/clip/ViT-B-32.pt',
    'resnet50': '/NS/robustness_2/work/vnanda/invariances_in_reps/deep-learning-base/checkpoints/clip/RN50.pt'
}

def list_models(dataset_name):
    if 'cifar' in dataset_name:
        return cifar_models.model_names
    else:
        return models.list_models()

def identity(x):
    return x

def load_state_dict(checkpoint_path, use_ema=False):

    def strip_module(state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # strip `module.` prefix
            name = k[7:] if k.startswith('module') else k
            new_state_dict[name] = v
        return new_state_dict
    def strip_model(state_dict):
        if sum([k.startswith('model') for k in state_dict.keys()]) > 0:
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # strip `model.` prefix
                if k.startswith('model'):
                    new_state_dict[k[6:]] = v
            return new_state_dict
        return state_dict

    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = ''
        if isinstance(checkpoint, dict):
            if use_ema and checkpoint.get('state_dict_ema', None) is not None:
                state_dict_key = 'state_dict_ema'
            elif use_ema and checkpoint.get('model_ema', None) is not None:
                state_dict_key = 'model_ema'
            elif 'state_dict' in checkpoint:
                state_dict_key = 'state_dict'
            elif 'model' in checkpoint:
                state_dict_key = 'model'
        if state_dict_key:
            state_dict = checkpoint[state_dict_key]
        else:
            state_dict = checkpoint
        
        state_dict = strip_module(strip_model(strip_module(state_dict)))
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, use_ema=False, strict=True):
    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return
    state_dict = load_state_dict(checkpoint_path, use_ema)
    model.load_state_dict(state_dict, strict=strict)


def load_pretrained_weights(model, pretrained, checkpoint_path, **kwargs):
    if pretrained:
        assert checkpoint_path, 'Must pass checkpoint_path for pretrained CIFAR models'
        load_checkpoint(model, checkpoint_path, **kwargs) # from timm

def make_symbolic(model: nn.Module):
    if model.__class__.__name__ == 'VisionTransformer':
        new_model = utils.SymbolicClipVisionTransformer(
            input_resolution = model.input_resolution,
            patch_size=model.conv1.weight.shape[-1],
            width=model.conv1.weight.shape[0],
            layers=model.transformer.layers, 
            heads=model.conv1.weight.shape[0] * 32 // 64, 
            output_dim=model.proj.shape[-1]
        )
        new_model.load_state_dict(model.state_dict())
        return new_model
    return model

def create_model(model_name: str, 
                dataset_name: str, 
                pretrained: bool = False,
                checkpoint_path: str = '',
                seed: int = 0, 
                loading_function: Callable = load_pretrained_weights, 
                callback: Callable = identity,
                num_classes: Optional[int] = None,
                loading_function_kwargs: Dict = {}) -> nn.Module:
    """
    callback: function that can be used to alter the model after creation 
            (eg: adding more classification layers)
    loading_function: user-defined function that defines the logic for loading weights
            from a user-defined checkpoint file. Must take 3 args: model, pretrained, checkpoint_path
    """
    pl_utils.seed.seed_everything(seed)

    if dataset_name not in ds.DATASET_PARAMS:    dataset_name = 'imagenet'
    if num_classes is None:    num_classes = ds.DATASET_PARAMS[dataset_name]['num_classes']

    if dataset_name.endswith('cifar10'):
        assert model_name in cifar_models.model_names, f'{model_name} not available for {dataset_name}'
        model = cifar_models.create_model_fn(model_name)(num_classes=num_classes)
        loading_function(model, pretrained, checkpoint_path, **loading_function_kwargs)
    elif 'clip' in dataset_name:
        assert pretrained and checkpoint_path, f'For CLIP models, pretrained must be True along with a checkpoint_path to a CLIP model'
        ## make sure checkpoints are saved as needed by CLIP (https://github.com/openai/CLIP/blob/main/clip/clip.py)
        model = clip.load(CLIP_MODEL_PATHS[model_name], device='cpu')[0].visual ## only take the visual component
        ## CLIP VIT models are not symbolically traceable; wrap
        model = make_symbolic(model)
        ## add a placeholder fc layer to be replaced later
        # in_fts = list(model.named_parameters())[-1][1].shape[-1]
        in_fts = model(torch.rand((1,3,224,224))).shape[1]
        model = nn.Sequential(OrderedDict(
            [('backbone', list(model.named_modules())[0][1]), ('fc', nn.Linear(in_fts, num_classes))]))
        if checkpoint_path.split('/')[-1] != CLIP_MODEL_PATHS[model_name].split('/')[-1]:
            print ('Loading finetuned CLIP weights')
            loading_function(model, pretrained, checkpoint_path, **loading_function_kwargs)
    else:
        # Use timm for ImageNet and other big dataset models 
        should_custom_load = pretrained and checkpoint_path != ''
        model = timm.create_model(model_name, 
                                  num_classes=num_classes,
                                  pretrained= ~should_custom_load and pretrained, ## 
                                  checkpoint_path='') ## default loading happens via loading_function
        if should_custom_load:    loading_function(model, pretrained, 
                                                   checkpoint_path, **loading_function_kwargs)

    return callback(model)