from time import pthread_getcpuclockid
import warnings
import torch
import torch.nn as nn
from timm.models import layers
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from .cifar_models.custom_modules import FakeReLUM
from timm.models.fx_features import GraphExtractNet
from .utils import intermediate_layer_names, unroll_dataparallel, reroll_dataparallel


def check_fake_and_no_relu(model, args, kwargs):
    if model.__class__.__name__ == 'ResNet':
        if ('fake_relu' in kwargs and kwargs['fake_relu']) or (len(args) > 1 and args[1]):
            for block in model.layer4:
                block.__setattr__('act3', FakeReLUM())
        elif ('no_relu' in kwargs and kwargs['no_relu']) in kwargs or (len(args) > 2 and args[2]):
            for block in model.layer4:
                block.__setattr__('act3', nn.Identity())
    elif model.__class__.__name__ == 'DenseNet':
        if ('fake_relu' in kwargs and kwargs['fake_relu']) or (len(args) > 1 and args[1]):
            model.features.norm5.__setattr__('act', FakeReLUM())
        elif ('no_relu' in kwargs and kwargs['no_relu']) or (len(args) > 2 and args[2]):
            warnings.warn('no_relu not applicable for DenseNet, this will have no effect')
    elif model.__class__.__name__ == 'VGG':
        if ('fake_relu' in kwargs and kwargs['fake_relu']) or (len(args) > 1 and args[1]):
            model.pre_logits.__setattr__('act2', FakeReLUM())
        elif ('no_relu' in kwargs and kwargs['no_relu']) or (len(args) > 2 and args[2]):
            model.pre_logits.__setattr__('act2', nn.Identity())


def get_prediction_from_latent(model, X, out):
    possible_names = ['fc', 'head']
    for name in possible_names:
        if hasattr(model, name):
            classifer = model.__getattr__(name)
            break
    try:
        pred = classifer(out)
    except:
        warnings.warn(f'Model classifier not in {possible_names}, '
                        'doing another forward pass, '
                        'might lead to unintended behaviors')
        pred = model(X)
    return pred


class LatentModel(nn.Module):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, X):
        out = self.model.forward_features(X)
        if self.model.__class__.__name__ == 'VisionTransformer':
            if self.model.global_pool:
                out = out[:, self.model.num_prefix_tokens:].mean(dim=1) if self.model.global_pool == 'avg' else out[:, 0]
            out = self.model.fc_norm(out)
            x_latent = out
        elif self.model.__class__.__name__ == 'VGG': 
            # special case; pooling is done in the head
            out = self.model.forward_head(out, pre_logits=True)
            x_latent = self.model.head(out, pre_logits=True)
        else:
            # these features need to be pooled
            if hasattr(self.model, 'global_pool'):
                out = self.model.global_pool(out)
                # check if global_pool contains Flatten
                if not isinstance(self.model.global_pool.flatten, nn.Flatten):
                    out = nn.Flatten(1)(out)
            else:
                pooling = layers.SelectAdaptivePool2d(pool_type='avg', 
                                                        flatten=True)
                out = pooling(out)
            x_latent = out
        return x_latent, out


def inference_with_features(model: nn.Module, 
                            X: torch.Tensor, *args, **kwargs):
    ### NOTE: if model is wrapped in a nn.DataParallel, then
    ### this function unwraps it and wraps it again; 
    ### this is okay for inference, but use with caution
    ### during training, since it might lead to unwanted behavior.
    ### Additionally for DataParallel models this can be suboptimal since
    ### get_prediction_from_latent takes in the unrolled model and 
    ### does not exploit parallelism. However, the overhead from this
    ### should be negligible since it's only one layer
    model, metadata = unroll_dataparallel(model)
    if ('with_latent' in kwargs and kwargs['with_latent']):
            check_fake_and_no_relu(model, args, kwargs)
            x_latent, out = reroll_dataparallel(LatentModel(model), metadata)(X)
            return get_prediction_from_latent(model, X, out), x_latent
    elif 'layer_num' in kwargs and kwargs['layer_num'] is not None:
        filtered_nodes = intermediate_layer_names(model)
        feature_model = GraphExtractNet(model, filtered_nodes)
        feature_model = reroll_dataparallel(feature_model, metadata)
        all_fts = feature_model(X)
        latent = all_fts[kwargs['layer_num']]
        if len(latent.shape) > 2:
            latent = latent.flatten(1)
        return get_prediction_from_latent(model, X, all_fts[-1].flatten(1)), latent
    model = reroll_dataparallel(model, metadata)
    return model(X)
