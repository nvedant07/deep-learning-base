### extra models that can be obtained via simple modifications of timm functions

from functools import partial
import numpy as np
import torch as ch
import torch.nn as nn
import torch.nn.functional as F
import timm.models as models
## TODO: do something about this import
import sys
sys.path.append('../training')
sys.path.append('deep-learning-base/training')
try:
    from partial_inference_layer import PartialLinear
except:
    from training.partial_inference_layer import PartialLinear

models.registry._model_pretrained_cfgs['wide_resnet50_4'] = models.resnet._cfg('')
def wide_resnet50_4(pretrained=False, **kwargs):
    model_args = dict(block=models.Bottleneck, layers=[3, 4, 6, 3], base_width=256, **kwargs)
    return models.resnet._create_resnet('wide_resnet50_4', pretrained, **model_args)
models.register_model(wide_resnet50_4)


## For models taken from Matryoshka representation learning (https://github.com/RAIVNLab/MRL)
class BlurPoolConv2d(ch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = ch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)

def apply_blurpool(mod: ch.nn.Module):
        for (name, child) in mod.named_children():
            if isinstance(child, ch.nn.Conv2d) and (np.max(child.stride) > 1 and child.in_channels >= 16): 
                setattr(mod, name, BlurPoolConv2d(child))
            else: apply_blurpool(child)

models.registry._model_pretrained_cfgs['resnet50_mrl'] = models.resnet._cfg('')
def resnet50_mrl(pretrained=False, **kwargs):
    model_args = dict(block=models.Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    m = models.resnet._create_resnet('resnet50_mrl', pretrained, **model_args)
    apply_blurpool(m)
    return m
models.register_model(resnet50_mrl)


### fixed feature resnets -- replaces the usual linear layer for model.fc with PartialLinear layer
### taken from MRL repo so they also require BlurPool
def resnet50_ff(ft_size, pretrained=False, **kwargs):
    model_args = dict(block=models.Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    m = models.resnet._create_resnet(f'resnet50_ff{ft_size}', pretrained, **model_args)
    apply_blurpool(m)
    old_linear = m.__getattr__('fc')
    m.__setattr__('fc', PartialLinear(ch.arange(ft_size), 
                                      nn.Linear(ft_size, old_linear.out_features)))
    return m
for ft_size in [8,16,32,64,128,256,512,1024,2048]:
    models.registry._model_pretrained_cfgs[f'resnet50_ff{ft_size}'] = models.resnet._cfg('')
    pfn = partial(resnet50_ff, ft_size)
    pfn.__setattr__('__name__', f'resnet50_ff{ft_size}')
    pfn.__setattr__('__module__', resnet50_mrl.__module__)
    models.register_model(pfn)
