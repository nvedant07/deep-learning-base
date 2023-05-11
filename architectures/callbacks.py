from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.core.lightning import LightningModule
import torch.nn as nn
import torch
from torch.optim import SGD, lr_scheduler, Adam
import torchmetrics
from typing import Dict, Optional, Callable, Union, List
import clip
from .utils import InputNormalize
from training.utils import OPTIMIZERS, _construct_opt_params
import dataset_metadata as ds
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
                 optimizer: Optional[str] = 'sgd',
                 warmup_steps: Optional[int] = None,
                 total_steps: Optional[int] = None,
                 training_params_dataset: Optional[str] = None):
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

        assert dataset_name or (mean is not None and std is not None), \
            'Both dataset_name and (mean, std) cannot be None'
        if mean is None:
            mean = ds.DATASET_PARAMS[dataset_name]['mean']
        if std is None:
            std = ds.DATASET_PARAMS[dataset_name]['std']
        self.normalizer = InputNormalize(mean, std)
        
        ####### training_params ######
        training_params_dataset = dataset_name if training_params_dataset is None \
            else training_params_dataset
        self.loss = ds.DATASET_PARAMS[training_params_dataset]['loss'] \
            if loss is None else loss
        self.lr = ds.DATASET_PARAMS[training_params_dataset]['lr'] \
            if lr is None else lr
        self.momentum = ds.DATASET_PARAMS[training_params_dataset]['momentum'] \
            if momentum is None else momentum
        ## note that step_lr and step_lr_gamma assume scheduler.step() is called after every epoch
        self.step_lr = ds.DATASET_PARAMS[training_params_dataset]['step_lr'] \
            if step_lr is None else step_lr
        self.step_lr_gamma = ds.DATASET_PARAMS[training_params_dataset]['step_lr_gamma'] \
            if step_lr_gamma is None else step_lr_gamma

        self.weight_decay = ds.DATASET_PARAMS[training_params_dataset]['weight_decay'] \
            if weight_decay is None else weight_decay
        ## this is used for get_cosine_schedule_with_warmup in training.utils and 
        ## assumes scheduler.step() is called every step 
        self.warmup_steps = ds.DATASET_PARAMS[training_params_dataset]['warmup_steps'] \
            if warmup_steps is None else warmup_steps
        self.total_steps = total_steps
        self.inference_kwargs = inference_kwargs
        self.optimizer = optimizer

    def forward(self, x, *args, **kwargs):
        return inference_with_features(self.model, 
                                       self.normalizer(x), 
                                       *args, **kwargs)

    def _has_latent(self):
        if 'with_latent' in self.inference_kwargs and self.inference_kwargs['with_latent']:
            return True
        if 'layer_num' in self.inference_kwargs and self.inference_kwargs['layer_num'] is not None:
            return True
        return False

    def _return_x(self):
        return 'with_x' in self.inference_kwargs and self.inference_kwargs['with_x']

    def predict_step(self, batch, batch_idx, dataloader_idx: Optional[int] = None):
        x, y = batch
        out = self(x, **self.inference_kwargs) ## not safe; TODO: add checks
        if self._has_latent():
            out, latent = out
            out, latent = out.detach(), latent.detach()
        else:
            out = out.detach()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            all_y = self.all_gather(y)
            all_y = torch.cat([all_y[i] for i in range(len(all_y))])
            all_x = self.all_gather(x)
            all_x = torch.cat([all_x[i] for i in range(len(all_x))])
            all_out = self.all_gather(out)
            all_out = torch.cat([all_out[i] for i in range(len(all_out))])
            if self._has_latent():
                all_latent = self.all_gather(latent)
                all_latent = torch.cat([all_latent[i] for i in range(len(all_latent))])
                op_tup = [all_out, all_latent, all_y]
            else:
                op_tup = [all_out, all_y]
            return op_tup + [all_x] if self._return_x() else op_tup

        op_tup = [out, latent, y] if self._has_latent() else [out, y]
        return op_tup + [x] if self._return_x() else op_tup

    def on_predict_epoch_end(self, results):
        for i in range(len(results)):
            cat_x, cat_out, cat_latent, cat_y = None, None, None, None
            for batch_res in results[i]:
                if self._has_latent() and self._return_x():
                    out, latent, y, x = batch_res
                elif self._has_latent():
                    out, latent, y = batch_res
                elif self._return_x():
                    out, y, x = batch_res
                else:
                    out, y = batch_res
                cat_out = out if cat_out is None else torch.cat((cat_out, out))
                cat_y = y if cat_y is None else torch.cat((cat_y, y))
                if self._has_latent():
                    cat_latent = latent if cat_latent is None else torch.cat((cat_latent, latent))
                if self._return_x():
                    cat_x = x if cat_x is None else torch.cat((cat_x, x))
            result_op = [cat_out, cat_latent, cat_y] if self._has_latent() else [cat_out, cat_y]
            results[i] = result_op + [cat_x] if self._return_x() else result_op
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
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', current_lr)
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
    
    def on_train_end(self) -> None:
        if self.trainer.is_global_zero:
            # run evaluation on test and val sets and log accuracy
            t = Trainer(accelerator='gpu', devices=1)
            out = t.predict(self, 
                dataloaders=[self.trainer.datamodule.val_dataloader(),
                             self.trainer.datamodule.test_dataloader()])
            print (f'out: {[o.shape for o in out]}')
            val_acc = torch.sum(torch.argmax(out[0][0], 1) == out[0][1]) / len(out[0][0])
            test_acc = torch.sum(torch.argmax(out[1][0], 1) == out[1][1]) / len(out[1][0])
            self.log('final_val_acc', val_acc)
            self.log('final_test_acc', test_acc)

    def prepare_model_params(self):
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
        clean_pred = inference_with_features(
            self.model, self.normalizer(x), *args, **kwargs
            )
        if adv:
            x, _ = self.attacker(x, targ, **self.attack_kwargs)
            if return_adv:
                return (x, 
                        clean_pred,
                        inference_with_features(
                            self.model, self.normalizer(x), *args, **kwargs
                            )
                        )
        return clean_pred, inference_with_features(self.model, self.normalizer(x), *args, **kwargs)

    def predict_step(self, batch, batch_idx, dataloader_idx: Optional[int] = None):
        x, y = batch
        adv_x, pred_clean_x, pred_adv_x = self(x, y, adv=True, return_adv=self.return_adv_samples, 
                                **self.inference_kwargs)
        adv_x, pred_clean_x, pred_adv_x = adv_x.detach(), pred_clean_x.detach(), pred_adv_x.detach()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            all_x = self.all_gather(x)
            all_x = torch.cat([all_x[i] for i in range(len(all_x))])
            all_y = self.all_gather(y)
            all_y = torch.cat([all_y[i] for i in range(len(all_y))])
            all_adv_x = self.all_gather(adv_x)
            all_adv_x = torch.cat([all_adv_x[i] for i in range(len(all_adv_x))])
            all_pred_adv_x = self.all_gather(pred_adv_x)
            all_pred_adv_x = torch.cat([all_pred_adv_x[i] for i in range(len(all_pred_adv_x))])
            all_pred_clean_x = self.all_gather(pred_clean_x)
            all_pred_clean_x = torch.cat([all_pred_clean_x[i] for i in range(len(all_pred_clean_x))])
            
            return (all_x, all_pred_clean_x), (all_adv_x, all_pred_adv_x), all_y

        return (x, pred_clean_x), (adv_x, pred_adv_x), y
    
    def on_predict_epoch_end(self, results):
        for i in range(len(results)):
            cat_x, cat_y, cat_xadv, cat_pred_x, cat_pred_xadv = None, None, None, None, None
            for batch_res in results[i]:
                (x, pred_x), (x_adv, pred_x_adv), y = batch_res
                cat_x = x if cat_x is None else torch.cat((cat_x, x))
                cat_y = y if cat_y is None else torch.cat((cat_y, y))
                cat_xadv = x_adv if cat_xadv is None else torch.cat((cat_xadv, x_adv))
                cat_pred_xadv = pred_x_adv if cat_pred_xadv is None else \
                    torch.cat((cat_pred_xadv, pred_x_adv))
                cat_pred_x = pred_x if cat_pred_x is None else \
                    torch.cat((cat_pred_x, pred_x))
            results[i] = ((cat_x, cat_pred_x), (cat_xadv, cat_pred_xadv), cat_y)
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


class MultimodalEvalWrapper(LightningWrapper):
    """
    Wrapper used for one-shot evaluation of multi-modal models like CLIP. 
    This uses the default prompt of 'a photo of a <class_name>' for each class
    and converts it into text representations, that are cached for easy access.
    Such models must have two methods: 
    encode_text and encode_image, just like OpenAI's 
    CLIP implementation (https://github.com/openai/CLIP/blob/main/clip/clip.py)
    """
    def __init__(self, tokenizer: Callable = clip.tokenize, 
                 scale: float = 100., 
                 class_prompt: str = 'a photo of a ',
                 **kwargs) -> None:
        """
        This wrapper allows zero-shot classification using CLIP-like 
        multi-modal models. 
        class_prompt (str): string after which class_name is added.
        """
        super().__init__(**kwargs)
        self.tokenizer = tokenizer
        self.scale = scale
        # information about classes (eg: prompt / class_names) can be changed 
        # dynamically through the use of callbacks
        self.class_prompt = class_prompt
        self.classes = None
        # We keep track of encoded text corresponding to each class.
        # This way we avoid an expensive forward pass for every batch.
        self.cached_class_embeddings = None

    def _set_class_prompt(self, prompt: str) -> None:
        self.class_prompt = prompt

    def _set_classes(self, classes) -> None:
        self.classes = classes
    
    def _reset_classes(self) -> None:
        self.classes = None
        self.cached_class_embeddings = None

    def _construct_class_text(self) -> List[str]:
        return [f'{self.class_prompt}{class_name}' for class_name in self.classes]

    def forward(self, x_image: torch.Tensor, x_text: List[str] = None, *args, **kwargs):
        if x_test is None:
            x_test = self._construct_class_text()
        text = self.forward_text(x_text, *args, **kwargs)
        image = self.forward_img(x_image, *args, **kwargs)
        return text, image

    def forward_img(self, x_image, *args, **kwargs) -> torch.Tensor:
        return self.model.encode_image(self.normalizer(x_image))

    def forward_text(self, x_text, *args, **kwargs) -> torch.Tensor:
        return self.model.encode_text(self.tokenizer(x_text))

    def predictions(self, text_encoding: torch.Tensor, 
                          image_encoding: torch.Tensor,
                          topk: int=1) -> torch.Tensor:
        # see OpenAI's implementation: 
        # https://github.com/openai/CLIP/blob/main/clip/model.py
        assert text_encoding.shape[-1] == image_encoding.shape[-1], \
            'Dimension of text and image encoders must be same!'
        image_encoding /= image_encoding.norm(dim=-1, keepdim=True)
        text_encoding /= text_encoding.norm(dim=-1, keepdim=True)
        similarity = (self.scale * image_encoding @ text_encoding.T).softmax(dim=-1)
        values, indices = similarity.topk(topk)
        return indices.detach(), values.detach() # shapes will be [batch_size, topk]

    def predict_step(self, batch, batch_idx, dataloader_idx: Optional[int] = None):
        image_x, y_gt = batch
        if self.cached_class_embeddings is None:
            self.cached_class_embeddings = self.forward_text(self._construct_class_text())
        predictions, scores = self.predictions(self.cached_class_embeddings, 
                                               self.forward_image(image_x))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            y_gt = torch.cat([y_ for y_ in self.all_gather(y_gt)])
            predictions = torch.cat([p_ for p_ in self.all_gather(predictions)])
            scores = torch.cat([s_ for s_ in self.all_gather(scores)])        
        return predictions, scores, y_gt

    def on_predict_epoch_end(self, results):
        for i in range(len(results)):
            cat_preds, cat_scores, cat_gt = None, None, None
            for batch_res in results[i]:
                preds, scores, gt = batch_res
                cat_preds = preds if cat_preds is None else torch.cat((cat_preds, preds))
                cat_gt = gt if cat_gt is None else torch.cat((cat_gt, gt))
                cat_scores = scores if cat_scores is None else torch.cat((cat_scores, scores))
            results[i] = [cat_preds, cat_scores, cat_gt]
        return results
