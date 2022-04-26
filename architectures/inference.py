import torch
import torch.nn as nn
from timm.models import layers

def inference_with_features(model: nn.Module, 
                            X: torch.Tensor, *args, **kwargs):
    if (('with_latent' in kwargs and kwargs['with_latent']) or \
        (len(args) and args[0])) and \
            hasattr(model, 'foward_features'):
            
            out = model.forward_features(X)
            # these features need to be pooled
            if hasattr(model, 'global_pool'):
                out = model.global_pool(out) 
                # check if global_pool contains Flatten
                if not isinstance(model.global_pool.flatten, nn.Flatten):
                    out = nn.Flatten(1)(out)
            else:
                pooling = layers.SelectAdaptivePool2d(pool_type='avg', 
                                                      flatten=True)
                out = pooling(out)
            
            possible_names = ['fc', 'head']
            for name in possible_names:
                if hasattr(model, name):
                    classifer = model.__getattr__(name)
                    break
            try:
                pred = classifer(out)
            except:
                raise ValueError(f'Model classifier not in {possible_names}')

            return pred, out
    
    return model(X, *args, **kwargs)
