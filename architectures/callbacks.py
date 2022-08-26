from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.core.lightning import LightningModule
import torch.nn as nn
import torch
from torch.optim import SGD, lr_scheduler, Adam
import torchmetrics
from typing import Dict, Optional, Callable, Union
from .utils import InputNormalize, OPTIMIZERS, _construct_opt_params
from datasets import dataset_metadata as ds
from attack.attack_module import Attacker
from architectures.utils import AverageMeter
from architectures.inference import inference_with_features

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
                 momentum: Optional[float] = None, 
                 step_lr: Optional[float] = None, 
                 step_lr_gamma: Optional[float] = None, 
                 dataset_name: Optional[str] = None,
                 inference_kwargs: Dict = {},
                 optimizer: Optional[str] = 'sgd'):
        """
        inference_kwargs are passed onto the ``inference_with_features``
        function in architectures.inference
        """
        super().__init__()
        self.model = model
        self.accuracy_top1 = torchmetrics.Accuracy(top_k=1)
        self.accuracy_top5 = torchmetrics.Accuracy(top_k=5)
        self.accuracy1_meter = AverageMeter('Acc@1')
        self.accuracy5_meter = AverageMeter('Acc@5')
        self.loss_meter = AverageMeter('Loss')

        assert dataset_name or (mean and std), \
            'Both dataset_name and (mean, std) cannot be None'
        if mean is None:
            mean = ds.DATASET_PARAMS[dataset_name]['mean']
        if std is None:
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
        self.inference_kwargs = inference_kwargs
        self.optimizer = optimizer

    def forward(self, x, *args, **kwargs):
        return inference_with_features(self.model, 
                                       self.normalizer(x), 
                                       *args, **kwargs)

    def predict_step(self, batch, batch_idx, dataloader_idx: Optional[int] = None):
        x, y = batch
        out, latent = self(x, **self.inference_kwargs) ## not safe; TODO: add checks
        out, latent = out.detach(), latent.detach()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            all_y = self.all_gather(y)
            all_y = torch.cat([all_y[i] for i in range(len(all_y))])
            all_x = self.all_gather(x)
            all_x = torch.cat([all_x[i] for i in range(len(all_x))])
            all_out = self.all_gather(out)
            all_out = torch.cat([all_out[i] for i in range(len(all_out))])
            all_latent = self.all_gather(latent)
            all_latent = torch.cat([all_latent[i] for i in range(len(all_latent))])
            return all_out, all_latent, all_y

        return out, latent, y

    def on_predict_epoch_end(self, results):
        for i in range(len(results)):
            cat_out, cat_latent, cat_y = None, None, None
            for batch_res in results[i]:
                out, latent, y = batch_res
                cat_out = out if cat_out is None else torch.cat((cat_out, out))
                cat_latent = latent if cat_latent is None else torch.cat((cat_latent, latent))
                cat_y = y if cat_y is None else torch.cat((cat_y, y))
            results[i] = (cat_out, cat_latent, cat_y)
        return results

    def step(self, batch, batch_idx):
        ## use this for compatibility with ddp2 and dp
        x, y = batch
        op = self(x)
        return {'pred': op, 'gt': y}
    
    def step_end(self, step_ops, stage):
        ## use this for compatibility with ddp2 and dp
        pred, true = step_ops['pred'], step_ops['gt']
        loss = self.loss(pred, true)
        self.accuracy1_meter.update(self.accuracy_top1(pred, true))
        self.accuracy5_meter.update(self.accuracy_top5(pred, true))
        self.loss_meter.update(loss)
        self.log(f'running_{stage}_acc', self.accuracy_top1(pred, true))
        return {'loss': loss}
    
    def epoch_end(self, all_ops, stage):
        self.log(f'{stage}_acc1', self.accuracy1_meter.avg)
        self.accuracy1_meter.reset()
        self.log(f'{stage}_acc5', self.accuracy5_meter.avg)
        self.accuracy5_meter.reset()
        self.log(f'{stage}_loss', self.loss_meter.avg)
        self.loss_meter.reset()

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def training_step_end(self, training_step_outputs):
        return self.step_end(training_step_outputs, 'train')

    def training_epoch_end(self, training_outputs):
        self.epoch_end(training_outputs, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step_end(self, validation_step_outputs):
        return self.step_end(validation_step_outputs, 'val')

    def validation_epoch_end(self, validation_outputs):
        return self.epoch_end(validation_outputs, 'val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)
    
    def test_step_end(self, test_step_outputs):
        self.step_end(test_step_outputs, 'test')
    
    def test_epoch_end(self, test_outputs):
        self.epoch_end(test_outputs, 'test')

    def prepare_model_params(self):
        # disable requires_grad for all but last layer
        return list(self.model.parameters())

    def configure_optimizers(self):
        optimizer = OPTIMIZERS[self.optimizer](self.prepare_model_params(), 
            **_construct_opt_params(
                self.optimizer, self.lr, self.weight_decay, self.momentum))
        schedule = lr_scheduler.StepLR(optimizer, 
                                       step_size=self.step_lr, 
                                       gamma=self.step_lr_gamma)
        return {'optimizer': optimizer, 'lr_scheduler': schedule}


class AdvAttackWrapper(LightningWrapper):
    """
    Useful for adversarial training and evaluation
    NOTE: use pl.callbacks to dynamically set params for adv attack. 
          Example given in attack.callbacks
    """
    def __init__(self, model: nn.Module, return_adv_samples=False, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.return_adv_samples = return_adv_samples
        self.attacker = Attacker(self.model, self.normalizer)
        self.clean_accuracy1_meter = AverageMeter('clean_acc@1')
        self.clean_accuracy5_meter = AverageMeter('clean_acc@5')
        self.clean_loss_meter = AverageMeter('clean_loss')
        self.adv_accuracy1_meter = AverageMeter('adv_acc@1')
        self.adv_accuracy5_meter = AverageMeter('adv_acc@5')
        self.adv_loss_meter = AverageMeter('adv_loss')

    def forward(self, x, targ=None, adv=False, return_adv=False, 
                *args, **kwargs):
        """
        kwargs will be sent to attacker
        args will be sent to the forward function of the model
        """
        if adv:
            x, _ = self.attacker(x, targ, **self.attack_kwargs)
            if return_adv:
                return (x, 
                        inference_with_features(
                            self.model, self.normalizer(x), *args, **kwargs
                            )
                        )
        return inference_with_features(self.model, self.normalizer(x), *args, **kwargs)

    def predict_step(self, batch, batch_idx, dataloader_idx: Optional[int] = None):
        x, y = batch
        adv_x, pred_adv_x = self(x, y, adv=True, return_adv=self.return_adv_samples, 
                                **self.inference_kwargs)
        adv_x, pred_adv_x = adv_x.detach(), pred_adv_x.detach()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            all_x = self.all_gather(x)
            all_x = torch.cat([all_x[i] for i in range(len(all_x))])
            all_adv_x = self.all_gather(adv_x)
            all_adv_x = torch.cat([all_adv_x[i] for i in range(len(all_adv_x))])
            all_pred_adv_x = self.all_gather(pred_adv_x)
            all_pred_adv_x = torch.cat([all_pred_adv_x[i] for i in range(len(all_pred_adv_x))])
            
            return all_x, (all_adv_x, all_pred_adv_x)

        return x, (adv_x, pred_adv_x)
    
    def on_predict_epoch_end(self, results):
        for i in range(len(results)):
            cat_x, cat_xadv, cat_pred_xadv = None, None, None
            for batch_res in results[i]:
                x, (x_adv, pred_x_adv) = batch_res
                cat_x = x if cat_x is None else torch.cat((cat_x, x))
                cat_xadv = x_adv if cat_xadv is None else torch.cat((cat_xadv, x_adv))
                cat_pred_xadv = pred_x_adv if cat_pred_xadv is None else \
                    torch.cat((cat_pred_xadv, pred_x_adv))
            results[i] = (cat_x, (cat_xadv, cat_pred_xadv))
        return results

    def step(self, batch, batch_idx):
        # done only during validate, test or train loops and cannot pass 
        # multi-dim tensors -- so this will never return_adv_samples
        # use trainer.predict() instead to get adversarial samples
        assert hasattr(self, 'attack_kwargs'), \
            'Must pass a callback that initializes attack_kwargs'
        x, y = batch
        return_dict = {'gt': y}

        adv_pred = self.forward(x, y, adv=True, return_adv=False, 
                                **self.attack_kwargs)
        clean_pred = self.forward(x, y, adv=False)
        return_dict['adv_pred'] = adv_pred
        return_dict['clean_pred'] = clean_pred
        return return_dict

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
        return outputs

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


class LinearEvalWrapper(LightningWrapper):
    """
    Wraps a pytorch model (from timm or otherwise) in a PyTorch-Lightning like
    model that can then be trained using a PyTorchLightning Trainer. 

    Input Normalization is performed here, before input is fed to the model

    This takes in a trained model, freezes all params apart from the final layer

    ## CAUTION: if model uses batchnorm, be sure to set layers with batchnorm to .eval() mode
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def prepare_model_params(self):
        # disable requires_grad for all but last layer
        parameters = list(self.model.parameters())
        for p in parameters[:-2]:
            p.requires_grad = False
        return parameters[-2:]
