"""
Contains all metadata for supported datasets inlcuding training params such 
as LR, weight decay,  etc.
"""
import torch
import torch.nn as nn
from torchvision import transforms

from .gaussian_blur import GaussianBlur

## Default Data Augs taken from: 
# https://github.com/MadryLab/robustness/blob/master/robustness/data_augmentation.py

# lighting transform
# https://git.io/fhBOc
IMAGENET_PCA = {
    'eigval':torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec':torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}
class Lighting(object):
    """
    Lighting noise (see https://git.io/fhBOc)
    """
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))



TRAIN_TRANSFORMS_IMAGENET = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1
        ),
        transforms.ToTensor(),
        Lighting(0.05, IMAGENET_PCA['eigval'], 
                      IMAGENET_PCA['eigvec'])
    ])

"""
Standard training data augmentation for ImageNet-scale datasets: Random crop,
Random flip, Color Jitter, and Lighting Transform (see https://git.io/fhBOc)
"""

TEST_TRANSFORMS_IMAGENET = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
"""
Standard test data processing (no augmentation) for ImageNet-scale datasets,
Resized to 256x256 then center cropped to 224x224.
"""

# Special transforms for smaller sized ImageNet(s)
TRAIN_TRANSFORMS_IMAGENET_CUSTOM = lambda size: transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1
        ),
        transforms.ToTensor(),
        Lighting(0.05, IMAGENET_PCA['eigval'], 
                      IMAGENET_PCA['eigvec'])
    ])
"""
Standard training data augmentation for ImageNet-scale datasets: Random crop,
Random flip, Color Jitter, and Lighting Transform (see https://git.io/fhBOc)
"""

TEST_TRANSFORMS_IMAGENET_CUSTOM = lambda size: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Resize(size),
        transforms.ToTensor(),
    ])
"""
Standard test data processing (no augmentation) for ImageNet-scale datasets,
Resized to 256x256 then center cropped to 224x224.
"""

# Data Augmentation: supervised learning defaults
TRAIN_TRANSFORMS_DEFAULT = lambda size: transforms.Compose([
            transforms.Resize(size),
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(.25,.25,.25),
            transforms.RandomRotation(2),
            transforms.ToTensor(),
        ])
"""
Generic training data transform, given image side length does random cropping,
flipping, color jitter, and rotation. Called as, for example,
:meth:`robustness.data_augmentation.TRAIN_TRANSFORMS_DEFAULT(32)` for CIFAR-10.
"""

TEST_TRANSFORMS_DEFAULT = lambda size:transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])
"""
Generic test data transform (no augmentation) to complement
:meth:`robustness.data_augmentation.TEST_TRANSFORMS_DEFAULT`, takes in an image
side length.
"""


SIMCLR_TRAIN_TRANSFORMS = lambda size, s=1: transforms.Compose(
    [
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.8*s, contrast=0.8*s, 
                                    saturation=0.8*s, hue=0.2*s)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        GaussianBlur(size//2, 0.5)
    ]
) # same as original simclr implementation as well as https://github.com/AndrewAtanov/simclr-pytorch and https://github.com/sthalles/SimCLR

SIMCLR_TRAIN_TRANSFORMS_NOCOLOR = lambda size, s=1: transforms.Compose(
    [
        transforms.RandomResizedCrop(size=size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        GaussianBlur(size//2, 0.5)
    ]
) # same as original simclr implementation as well as https://github.com/AndrewAtanov/simclr-pytorch and https://github.com/sthalles/SimCLR

DATASET_PARAMS = {
    'imagenet': {
        'num_classes': 1000,
        'input_size': 224, 
        'mean': torch.tensor([0.485, 0.456, 0.406]), 
        'std': torch.tensor([0.229, 0.224, 0.225]), 
        'transform_train': TRAIN_TRANSFORMS_IMAGENET,
        'transform_test': TEST_TRANSFORMS_IMAGENET,
        'loss': nn.CrossEntropyLoss(), 
        'epochs': 200,
        'batch_size':256,
        'weight_decay':1e-4,
        'step_lr': 50, 
        'step_lr_gamma': 0.1, 
        'lr': 0.1, 
        'momentum': 0.9
    },
    'cifar10': {
        'num_classes': 10,
        'input_size': 32, 
        'mean': torch.tensor([0.4914, 0.4822, 0.4465]),
        'std': torch.tensor([0.2023, 0.1994, 0.2010]), 
        'transform_train': TRAIN_TRANSFORMS_DEFAULT(32),
        'transform_test': TEST_TRANSFORMS_DEFAULT(32),
        'loss': nn.CrossEntropyLoss(), 
        'epochs': 150,
        'batch_size': 128,
        'weight_decay':5e-4,
        'step_lr': 50, 
        'step_lr_gamma': 0.1, 
        'lr': 0.1, 
        'momentum': 0.9
    },
    'cifar100': {
        'num_classes': 100,
        'input_size': 32, 
        'mean': torch.tensor([0.5071, 0.4865, 0.4409]),
        'std': torch.tensor([0.2673, 0.2564, 0.2762]),
        'transform_train': TRAIN_TRANSFORMS_DEFAULT(32),
        'transform_test': TEST_TRANSFORMS_DEFAULT(32),
        'loss': nn.CrossEntropyLoss(), 
        'epochs': 150,
        'batch_size': 128,
        'weight_decay':5e-4,
        'step_lr': 50, 
        'step_lr_gamma': 0.1, 
        'lr': 0.1, 
        'momentum': 0.9
    },
    'stl10': {
        'num_classes': 10,
        'input_size': 96, 
        'mean': torch.tensor([0.4467, 0.4398, 0.4066]),
        'std': torch.tensor([0.2603, 0.2566, 0.2713]),
        'transform_train': TRAIN_TRANSFORMS_DEFAULT(96),
        'transform_test': TEST_TRANSFORMS_DEFAULT(96),
        'loss': nn.CrossEntropyLoss(), 
        'epochs': 150,
        'batch_size': 128,
        'weight_decay':5e-4,
        'step_lr': 50, 
        'step_lr_gamma': 0.1, 
        'lr': 0.1, 
        'momentum': 0.9
    }
}