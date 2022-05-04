import numpy as np
import torch as ch
import torch.nn as nn

import lpips

from .base_rep_loss import BaseLoss

ROBUST_MODEL_WEIGHTS = {
    'resnet18_madry_imagenet': '/NS/twitter_archive2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-18-l2-eps3.ckpt',
    'resnet50_madry_imagenet': '/NS/twitter_archive2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/resnet-50-l2-eps3.ckpt',
    'vgg16_bn_madry_imagenet': '/NS/twitter_archive2/work/vnanda/adv-robustness/logs/robust_imagenet/eps3/vgg16_bn_l2_eps3.ckpt'
}


def forward_models(model, inp):
    if isinstance(model, nn.DataParallel):
        model_name = model.module.__class__.__name__
    else:
        model_name = model.__class__.__name__
    fake_relu = True
    if model_name == 'VGG' or model_name == 'InceptionV3' or 'sparse' in model_name.lower():
        fake_relu = False
    return model(inp, with_latent=True, fake_relu=fake_relu)


class LPNormLossSingleModel(BaseLoss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, model1, model2, inp, targ1, targ2):
        _, rep1 = forward_models(model1, inp)
        self.model1_loss_normed = ch.div(ch.norm(rep1 - targ1, p=self.lpnorm_type, dim=1), 
                                            ch.norm(targ1, p=self.lpnorm_type, dim=1))
        self.model1_loss = ch.norm(rep1 - targ1, p=self.lpnorm_type, dim=1)
        loss = self.model1_loss_normed

        rep1 = None
        ch.cuda.empty_cache()

        return loss

    def __repr__(self):
        return f'Model1 Loss: {ch.mean(self.model1_loss)} ({ch.mean(self.model1_loss_normed)})'


class LpNormLossSingleModelPerceptual(BaseLoss):
    """
    Uses perceptual loss (LPIPS) to simulate human perception, goal of this loss 
    is to generate images which are perceptually different from the target.
    """
    def __init__(self, *args, **kwargs):
        """
        args = first argument should be the 
        args must contain the model name to be used for LPIPS. Valid model names:
        args.lpips_model: model architecture
            alex (for finetuned weights, use 
                   /NS/twitter_archive2/work/vnanda/PerceptualSimilarity/lpips/weights/v0.1/alex.pth)
            vgg16 (for finetuned weights, use
                   /NS/twitter_archive2/work/vnanda/PerceptualSimilarity/lpips/weights/v0.1/vgg.pth)
            squeeze (for finetuned weights, use 
                   /NS/twitter_archive2/work/vnanda/PerceptualSimilarity/lpips/weights/v0.1/squeeze.pth)
            resnet18
            resnet18_madry_imagenet
            resnet50_madry_imagenet
            vgg16_bn_madry_imagenet

        args.lpips_model_path: if set to None, then standard ImageNet weights are used. 
            For models of the form *_madry_*, there must be a path, if None is specified, 
            then the default ones are used (defined in `ROBUST_MODEL_WEIGHTS`)
        """
        super().__init__(*args, **kwargs)
        if kwargs['lpips_model'] in ROBUST_MODEL_WEIGHTS and kwargs['lpips_model_path'] is None:
            model_path = ROBUST_MODEL_WEIGHTS[kwargs['lpips_model']]
        else:
            model_path = args.lpips_model_path
        self.lpips = lpips.LPIPS(net=args.lpips_model, 
                                 model_path=model_path, 
                                 pretrained=model_path is not None, 
                                 device=ch.device(f'cuda:{kwargs["devices"][0]}'))
        self.target_inp = None
        self.scaling_factor = None

    def set_target_inps(self, target_inp) -> None:
        self.target_inp = target_inp

    def __call__(self, model1, model2, inp, targ1, targ2):
        assert self.target_inp is not None, 'Must call `set_target_inps` first!'

        _, rep1 = forward_models(model1, inp)
        self.model1_loss_normed = ch.div(ch.norm(rep1 - targ1, p=self.args.lpnorm_type, dim=1), 
                                            ch.norm(targ1, p=self.args.lpnorm_type, dim=1))
        self.model1_loss = ch.norm(rep1 - targ1, p=self.args.lpnorm_type, dim=1)
        # LPIPS expects images in [-1, 1] so pass normalize = True
        self.lpips_distance = self.lpips(inp, self.target_inp, normalize=True).squeeze()

        if self.scaling_factor is None:
            # set it initially, otherwise this will never converge
            self.scaling_factor = ch.mean(self.model1_loss_normed).item()/ch.mean(self.lpips_distance).item()
        loss = self.model1_loss - self.scaling_factor * self.lpips_distance

        rep1 = None
        ch.cuda.empty_cache()

        return loss

    def __repr__(self) -> str:
        return f'Model1 Loss: {ch.mean(self.model1_loss)} ({ch.mean(self.model1_loss_normed)})'\
               f'LPIPS Dist: {ch.mean(self.lpips_distance)}'
