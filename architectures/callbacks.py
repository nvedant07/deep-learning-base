from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.core.lightning import LightningModule
import torch.nn as nn
import torch
from torch.optim import SGD, lr_scheduler
import torchmetrics
from typing import Dict, Optional, Callable, Union
from .utils import InputNormalize
from datasets import dataset_metadata as ds
from attack.attack_module import Attacker
from architectures.utils import AverageMeter

class LightningWrapper(LightningModule):
    """
    Wraps a pytorch model (from timm or otherwise) in a PyTorch-Lightning like
    model that can then be trained using a PyTorchLightning Trainer. 

    Can be inherited/overridden by any code using this

    Input Normalization is performed here, before input is fed to the model
    """
    def __init__(self, 
                 model: nn.Module, 
                 mean: Optional[torch.Tensor] = None, 
                 std: Optional[torch.Tensor] = None, 
                 loss: Optional[Callable] = None, 
                 lr: Optional[float] = None, 
                 weight_decay: Optional[float] = None, 
                 step_lr: Optional[float] = None, 
                 step_lr_gamma: Optional[float] = None, 
                 momentum: Optional[float] = None, 
                 dataset_name: Optional[str] = None):
        super().__init__()
        self.model = model
        self.accuracy_top1 = torchmetrics.Accuracy(top_k=1)
        self.accuracy_top5 = torchmetrics.Accuracy(top_k=5)
        self.accuracy1_meter = AverageMeter('Acc@1')
        self.accuracy5_meter = AverageMeter('Acc@5')
        self.loss_meter = AverageMeter('Loss')

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
        pred, true = training_step_outputs['pred'], training_step_outputs['gt']
        train_loss = self.loss(pred, true)
        self.log('running_acc', self.accuracy_top1(pred, true))
        return {'loss': train_loss, 
                'acc5': self.accuracy_top5(pred, true),
                'acc1': self.accuracy_top1(pred, true),
                'num_samples': len(pred)}

    def training_epoch_end(self, training_outputs):
        accs1, accs5, losses, samples = [], [], [], []
        for op in training_outputs:
            accs1.append(op['acc1'] * op['num_samples'])
            accs5.append(op['acc5'] * op['num_samples'])
            losses.append(op['loss'] * op['num_samples'])
            samples.append(op['num_samples'])
        self.log(f'train_acc1', sum(accs1)/sum(samples))
        self.log(f'train_acc5', sum(accs5)/sum(samples))
        self.log(f'train_loss', sum(losses)/sum(samples))

    def validation_step(self, batch, batch_idx):
        x, y = batch
        op = self.forward(x)
        return {'pred': op, 'gt': y}

    def validation_step_end(self, validation_step_outputs):
        ## use this for compatibility with ddp2 and dp
        pred, true = validation_step_outputs['pred'], validation_step_outputs['gt']
        val_loss = self.loss(pred, true)
        self.log('running_acc', self.accuracy_top1(pred, true))
        return {'loss': val_loss,
                'acc5': self.accuracy_top5(pred, true), 
                'acc1': self.accuracy_top1(pred, true),
                'num_samples': len(pred)}

    def validation_epoch_end(self, validation_outputs):
        accs1, accs5, losses, samples = [], [], [], []
        for op in validation_outputs:
            accs1.append(op['acc1'] * op['num_samples'])
            accs5.append(op['acc5'] * op['num_samples'])
            losses.append(op['loss'] * op['num_samples'])
            samples.append(op['num_samples'])
        self.log(f'val_acc1', sum(accs1)/sum(samples))
        self.log(f'val_acc5', sum(accs5)/sum(samples))
        self.log(f'val_loss', sum(losses)/sum(samples))
        print (f'Validation Acc: {sum(accs1)/sum(samples):.3f} (top1), '
               f'{sum(accs5)/sum(samples):.3f} (top5)')

    def test_step(self, batch, batch_idx):
        x, y = batch
        op = self.forward(x)
        return {'pred': op, 'gt': y}
    
    def test_step_end(self, test_step_outputs):
        ## use this for compatibility with ddp2 and dp
        ## test_step_outputs here has all the splits aggregated
        pred, true = test_step_outputs['pred'], test_step_outputs['gt']
        return {'loss': self.loss(pred, true),
                'acc5': self.accuracy_top5(pred, true), 
                'acc1': self.accuracy_top1(pred, true),
                'num_samples': len(true)}
    
    def test_epoch_end(self, test_outputs):
        accs1, accs5, samples = [], [], []
        for op in test_outputs:
            accs1.append(op['acc1'] * op['num_samples'])
            accs5.append(op['acc5'] * op['num_samples'])
            samples.append(op['num_samples'])
        self.log(f'test_acc1', sum(accs1)/sum(samples))
        self.log(f'test_acc5', sum(accs5)/sum(samples))

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), self.lr, momentum=self.momentum,
                        weight_decay=self.weight_decay)
        schedule = lr_scheduler.StepLR(optimizer, 
                                       step_size=self.step_lr, 
                                       gamma=self.step_lr_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': schedule}


class AdvAttackWrapper(LightningWrapper):
    """
    Useful for adversarial training and evaluation
    NOTE: use pl.callbacks to set params for adv attack. 
          Example given in attack.callbacks
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attacker = Attacker(self.model, self.normalizer)
        self.clean_accuracy1_meter = AverageMeter('clean_acc@1')
        self.clean_accuracy5_meter = AverageMeter('clean_acc@5')
        self.clean_loss_meter = AverageMeter('clean_loss')
        self.adv_accuracy1_meter = AverageMeter('adv_acc@1')
        self.adv_accuracy5_meter = AverageMeter('adv_acc@5')
        self.adv_loss_meter = AverageMeter('adv_loss')

    def forward(self, x, targ=None, adv=False, *args, **kwargs):
        """
        kwargs will be sent to attacker
        args will be sent to the forward function of the model
        """
        if adv:
            x, _ = self.attacker(x, targ, **kwargs)
        return self.model(self.normalizer(x), *args)

    def step(self, batch, batch_idx):
        assert hasattr(self, 'attack_kwargs'), \
            'Must pass a callback that initializes attack_kwargs'
        x, y = batch
        adv_pred = self.forward(x, y, adv=True, **self.attack_kwargs)
        clean_pred = self.forward(x, y, adv=False)
        return {'adv_pred': adv_pred, 'clean_pred': clean_pred, 'gt': y}

    def step_end(self, step_outputs, split):
        adv_pred, clean_pred, true = step_outputs['adv_pred'], \
                                     step_outputs['clean_pred'], \
                                     step_outputs['gt']                             

        loss_clean = self.loss(clean_pred, true).detach().item()
        running_clean_acc1 = self.accuracy_top1(clean_pred, true).detach().item()
        running_clean_acc5 = self.accuracy_top5(clean_pred, true).detach().item()
        self.clean_accuracy1_meter.update(running_clean_acc1, len(true))
        self.clean_accuracy5_meter.update(running_clean_acc5, len(true))
        self.clean_loss_meter.update(loss_clean, len(true))

        loss_adv = self.loss(adv_pred, true)
        if split != 'train':
            loss_adv = loss_adv.detach()
        running_adv_acc1 = self.accuracy_top1(adv_pred, true).detach().item()
        running_adv_acc5 = self.accuracy_top5(adv_pred, true).detach().item()
        self.adv_accuracy1_meter.update(running_adv_acc1, len(true))
        self.adv_accuracy5_meter.update(running_adv_acc5, len(true))
        self.adv_loss_meter.update(loss_adv.item(), len(true))

        self.log('running_acc_clean', self.clean_accuracy1_meter.avg)
        self.log('running_acc_adv', self.adv_accuracy1_meter.avg)

        return {'loss': loss_adv}
    
    def epoch_end(self, outputs, split):
        self.log(f'clean_{split}_acc1', self.clean_accuracy1_meter.avg)
        self.clean_accuracy1_meter.reset()
        self.log(f'adv_{split}_acc1', self.adv_accuracy1_meter.avg)
        self.adv_accuracy1_meter.reset()
        self.log(f'clean_{split}_acc5', self.clean_accuracy5_meter.avg)
        self.clean_accuracy5_meter.reset()
        self.log(f'adv_{split}_acc5', self.adv_accuracy5_meter.avg)
        self.adv_accuracy5_meter.reset()
        self.log(f'{split}_loss_adv', self.adv_loss_meter.avg)
        self.adv_loss_meter.reset()
        self.log(f'{split}_loss_clean', self.clean_loss_meter.avg)
        self.clean_loss_meter.reset()

    def training_step(self, batch, batch_idx):
        ## use this for compatibility with ddp2 and dp
        assert self.training
        return self.step(batch, batch_idx)

    def training_step_end(self, training_step_outputs):
        ## use this for compatibility with ddp2 and dp
        assert self.training
        return self.step_end(training_step_outputs, 'train')

    def training_epoch_end(self, training_outputs):
        assert self.training
        return self.epoch_end(training_outputs, 'train')

    def validation_step(self, batch, batch_idx):
        ## use this for compatibility with ddp2 and dp
        assert not self.training
        return self.step(batch, batch_idx)

    def validation_step_end(self, validation_step_outputs):
        ## use this for compatibility with ddp2 and dp
        assert not self.training
        return self.step_end(validation_step_outputs, 'val')

    def validation_epoch_end(self, validation_outputs):
        assert not self.training
        return self.epoch_end(validation_outputs, 'val')

    def test_step(self, batch, batch_idx):
        ## use this for compatibility with ddp2 and dp
        assert not self.training
        return self.step(batch, batch_idx)

    def test_step_end(self, test_step_outputs):
        ## use this for compatibility with ddp2 and dp
        assert not self.training
        return self.step_end(test_step_outputs, 'test')

    def test_epoch_end(self, test_outputs):
        assert not self.training
        return self.epoch_end(test_outputs, 'test')
