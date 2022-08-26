from typing import Optional
import torch as ch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.core.lightning import LightningModule
from training.partial_inference_layer import PartialLinear


def setup_model_for_finetuning(model: nn.Module, num_classes: int, 
                               mode: Optional[str] = None, 
                               fraction: Optional[float] = None, 
                               seed: Optional[int] = None, 
                               inplace: bool = True):
    """
    model (nn.Module): a PyTorch model
    num_classes: classes for the downstream dataset
    inplace: if model should be modified here
    """
    name, param = list(model.named_modules())[-1]
    in_fts = param.in_features

    if mode is not None:
        assert fraction is not None and seed is not None
        num_neurons = int(fraction * in_fts)
        linear = nn.Linear(num_neurons, num_classes)
        if mode == 'random':
            ch.manual_seed(seed)
            ## masking operation fails on GPU; fix requires making a copy on CPU 
            ## (https://github.com/pytorch/pytorch/issues/61032); better to just do 
            ## masking on CPU and then use the (slower) .to(device) call here
            chosen_neurons = ch.randperm(in_fts)[:num_neurons]
        new_layer = PartialLinear(chosen_neurons, linear)
    else:
        new_layer = nn.Linear(in_fts, num_classes)
    if inplace:
        model.__setattr__(name, new_layer)
    return new_layer

def get_param_names(model: nn.Module, mode: str):
    if mode == 'linear':
        return [f'model.{x[0]}' for x in list(model.named_parameters())[-2:]]
    return []

class FinetuningCallback(Callback):

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
        if self.mode == 'linear':
            for _, mod in list(pl_module.model.named_modules())[:-2]:
                mod.eval()
