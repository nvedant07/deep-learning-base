from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.core.lightning import LightningModule
import torch.nn as nn
import torch
from torch.optim import SGD, lr_scheduler
from collections.abc import Callable
from typing import Optional
from .utils import InputNormalize
from datasets import dataset_metadata as ds


class LightningWrapper(LightningModule):
    """
    Wraps a pytorch model (from timm or otherwise) in a PyTorch-Lightning like
    model that can then be trained using a Trainer. 

    Can be inherited/overridden by any code using this

    Input Normalization is performed here
    """
    def __init__(self, 
                 model: nn.Module, 
                 mean: Optional[torch.Tensor[float]] = None, 
                 std: Optional[torch.Tensor[float]] = None, 
                 loss: Optional[Callable] = None, 
                 lr: Optional[float] = None, 
                 weight_decay: Optional[float] = None, 
                 step_lr: Optional[float] = None, 
                 step_lr_gamma: Optional[float] = None, 
                 momentum: Optional[float] = None, 
                 dataset_name: Optional[str] = None):
        self.model = model
        assert dataset_name or (mean and std), \
            'Both dataset_name and (mean, std) cannot be None'
        if not (mean and std):
            mean = ds.DATASET_PARAMS[dataset_name]['mean']
            std = ds.DATASET_PARAMS[dataset_name]['std']
        self.normalizer = InputNormalize(mean, std)
        self.loss = ds.DATASET_PARAMS[dataset_name]['loss'] \
            if loss is None else loss
        self.lr = ds.DATASET_PARAMS[dataset_name]['lr'] \
            if lr is None else lr
        self.momentum = ds.DATASET_PARAMS[dataset_name]['momentum'] \
            if momentum is None else momentum
        self.step_lr = ds.DATASET_PARAMS[dataset_name]['step_lr'] \
            if step_lr is None else step_lr
        self.step_lr_gamma = ds.DATASET_PARAMS[dataset_name]['step_lr_gamma'] \
            if step_lr_gamma is None else step_lr_gamma
        self.weight_decay = ds.DATASET_PARAMS[dataset_name]['weight_decay'] \
            if weight_decay is None else weight_decay
    
    def forward(self, x, *args, **kwargs):
        return self.model(self.normalizer(x), *args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        ## use this for compatibility with ddp2 and dp
        x, y = batch
        op = self.forward(x)
        return {'pred': op, 'gt': y}
    
    def training_step_end(self, training_step_outputs):
        ## use this for compatibility with ddp2 and dp
        pred, true = [], []
        for op in training_step_outputs:
            pred.append(op['pred'])
            true.append(op['gt'])
        return self.loss(torch.cat(pred), torch.cat(true))
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        op = self.forward(x)
        return {'pred': op, 'gt': y}
    
    def validation_step_end(self, validation_step_outputs):
        ## use this for compatibility with ddp2 and dp
        pred, true = [], []
        for op in validation_step_outputs:
            pred.append(op['pred'])
            true.append(op['gt'])
        return self.loss(torch.cat(pred), torch.cat(true))
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        op = self.forward(x)
        return {'pred': op, 'gt': y}
    
    def test_step_end(self, test_step_outputs):
        ## use this for compatibility with ddp2 and dp
        pred, true = [], []
        for op in test_step_outputs:
            pred.append(op['pred'])
            true.append(op['gt'])
        return {'loss': self.loss(torch.cat(pred), torch.cat(true)),
                'acc': }

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), self.lr, momentum=self.momentum,
                        weight_decay=self.weight_decay)
        schedule = lr_scheduler.StepLR(optimizer, 
                                       step_size=self.step_lr, 
                                       gamma=self.step_lr_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': schedule}

