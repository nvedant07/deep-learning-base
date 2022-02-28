This is a directory for common DL code including models, adversarial attacks, supervised training. This is provides easy ways to extend models and training paradigms.

> * Models

ImageNet (and similarly large) models are taken from [timm](https://github.com/rwightman/pytorch-image-models). 
CIFAR10/100 model schemas differ a little from ImageNet models.

Example of initializing a model (in case dataset isn't specified in ``dataset``, initializes an ImageNet model):

```
# show an example here
```

Initializing with user-defined pre-trained weights:

```
# show an example where pretrained weights are user specified
```

Initializing with pre-trained weights in ``timm``:

```
# show some resnets taken from timm
```

To perform inference:

```
# show an example of inference
```

To get intermediate layer representations:

```
# show an example of obtaining feature maps of ResNet
```


> * Datasets

Standard dataloaders from torchvision + support for custom datasets. Many dataset classes taken from [robustness](https://github.com/MadryLab/robustness) and [robust-models-transfer](https://github.com/Microsoft/robust-models-transfer)

Initializing datasets:

```
# show an example of initializing datasets
```

Datasets with custom data augmentations:

```
# show the use of data augmentation callback
```

Model dependent data augmentations:

```
# example of model dependent data aug, eg: worst-of-K spatial augmentations
```


> * Training

Supports different losses (eg: adversarial training), different optimizers (all included in ``timm``)

Supervised: Uses [PyTorch-Lightning](https://github.com/facebookresearch/vissl) for easy, scalable training. 

Self-supervised: Uses [vissl](https://github.com/facebookresearch/vissl) to track SOTA models and weights.

Uses [robustness](https://github.com/MadryLab/robustness) for attack module used in adversarial training.

Example of supervised training (standard):

```
# show an example os using PyTorch-Lightning here
```

Example of supervised training (adversarial):

```
# show an example of adversarial training
```

> * Adversarial Attacks

Attack module in [robustness](https://github.com/MadryLab/robustness). Also includes spatial attacks, taken from [adversarial_spatial](https://github.com/MadryLab/adversarial_spatial).

To do adversarial attack on models, wrap them in the ``Attack`` module. Here's an example:

```
# Show how to do L_inf PGD attack on an ImageNet ResNet
```

To do spatial attacks:

```
# Show an example of worst-of-K and first order spatial attacks
```


