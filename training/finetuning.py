import sched
from typing import Optional
import torch as ch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import lr_scheduler
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.core.lightning import LightningModule
from training.partial_inference_layer import PartialLinear, PCALinear
from architectures.callbacks import LightningWrapper
from training.utils import OPTIMIZERS, _construct_opt_params, get_cosine_schedule_with_warmup


def setup_model_for_finetuning(model: nn.Module, 
                               num_classes: int, 
                               mode: Optional[str] = None, 
                               fraction: Optional[float] = None, 
                               seed: Optional[int] = None, 
                               inplace: bool = True,
                               infer_features: bool = False,
                               num_neurons: Optional[int] = None, 
                               return_metadata: bool = False, 
                               layer_kwargs: dict = {}):
    """
    model (nn.Module): a PyTorch model
    num_classes: classes for the downstream dataset
    inplace: if model should be modified here
    """

    ### if not doing on penultimate layer then set ``infer_features`` to True
    if not infer_features:
        for name, param in list(model.named_modules())[::-1]:
            if isinstance(param, nn.Linear) or isinstance(param, PartialLinear):
                break
        in_fts = param.in_features
    else:
        name = 'fc'
        in_fts = model(ch.rand((1,3,224,224))).shape[1] # TODO: remove this hard coded input dim

    if mode is not None:
        assert (fraction is not None and num_neurons is None) or \
            (fraction is None and num_neurons is not None)
        if num_neurons is None:
            num_neurons = int(fraction * in_fts)
        else:
            fraction = num_neurons / in_fts
        linear = nn.Linear(num_neurons, num_classes)
        if mode == 'random':
            assert seed is not None, 'must pass seed for random mode'
            ch.manual_seed(seed)
            ## masking operation fails on GPU; fix requires making a copy on CPU 
            ## (https://github.com/pytorch/pytorch/issues/61032); better to just do 
            ## masking on CPU and then use the (slower) .to(device) call here
            chosen_neurons = ch.randperm(in_fts)[:num_neurons]
            new_layer = PartialLinear(chosen_neurons, linear)
        elif mode == 'first':
            chosen_neurons = ch.arange(num_neurons)
            new_layer = PartialLinear(chosen_neurons, linear)
        elif 'pca' in mode:
            assert 'projection_matrix' in layer_kwargs, \
                'layer_kwargs must have projection_matrix for mode == pca'
            if mode == 'pca-least':
                layer_kwargs['which'] = 'least'
            new_layer = PCALinear(num_neurons, linear, **layer_kwargs)
        elif mode == 'randproj':
            assert 'generator' in layer_kwargs, \
                'Must pass a generator for random projections'
            layer_kwargs['projection_matrix'] = F.normalize(
                ch.rand((in_fts, in_fts), generator=layer_kwargs.pop('generator')), 
                dim=0)
            new_layer = PCALinear(num_neurons, linear, **layer_kwargs)
        else:
            raise ValueError(f'Mode {mode} not recognized!')
    else:
        new_layer = nn.Linear(in_fts, num_classes)
    if inplace:
        if '.' in name:
            for modname in name.split('.')[:-1]:
                mod = model.__getattr__(modname)
            mod.__setattr__(name.split('.')[-1], new_layer)
        else:
            model.__setattr__(name, new_layer)
    
    if not return_metadata:
        return new_layer
    return new_layer, in_fts, num_neurons, fraction


def get_param_names(model: nn.Module, mode: str):
    if mode == 'linear':
        return [f'model.{x[0]}' for x in list(model.named_parameters())[-2:]]
    return []


## add cosine decay on LR -- useful for full finetuning
class CosineLRWrapper(LightningWrapper):
    """
    Callback on the model itself.
    CAUTION: this changes the LR at every step and 
    hence step_lr should be adjusted accordingly
    """
    def __init__(self, *args, **kwds) -> None:
        super().__init__(*args, **kwds)
    
    def configure_optimizers(self):
        optimizer = OPTIMIZERS[self.optimizer](self.prepare_model_params(), 
            **_construct_opt_params(
                self.optimizer, self.lr, self.weight_decay, self.momentum))
        schedule = get_cosine_schedule_with_warmup(optimizer, 
                                                   self.warmup_steps, 
                                                   self.total_steps)
        return {'optimizer': optimizer, 
                'lr_scheduler': {'scheduler': schedule, 
                                 'interval': 'step', 
                                 'frequency': 1}}


class FinetuningCallback(Callback):
    """
    Callback passed to trainer, 
    """

    def __init__(self, mode: str) -> None:
        """
        mode (str): "linear", "full"
            if "linear" only last two parameters of model have requires_grad set to true
            additionally all parameters before the final layer are set to .eval() mode to 
            ensure layers that are updated during forward pass (eg: batchnorm) are not
            changed while finetuning.
        """
        super().__init__()
        self.mode = mode

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.is_global_zero:
            print (f'Learning rate at start of training epoch: {[o.param_groups[0]["lr"] for o in trainer.optimizers]}')
        if self.mode == 'linear':
            named_mods = list(pl_module.model.named_children())
            for _, mod in named_mods[:-1]:
                mod.eval()
