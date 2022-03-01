This is a directory for common DL code including models, adversarial attacks, supervised training. This is provides easy ways to extend models and training paradigms. This provides a smooth interface between many SOTA libraries like ``timm``, ``robustness``, ``vissl``. All the code relies on PyTorch and PyTorch-Lightning.

## Models

ImageNet (and similarly large) models are taken from [timm](https://github.com/rwightman/pytorch-image-models). 
CIFAR10/100 model schemas differ a little from ImageNet models.

Example of initializing a (random) model (in case dataset isn't specified in ``dataset``, initializes an ImageNet model):

.. code-block:: python
    import architectures as arch

    dataset = 'cifar10'
    model = 'resnet18'

    m = arch.create_model(model, dataset)
    # m has randomly initialized weights


Initializing with user-defined pre-trained weights:

.. code-block:: python
    import architectures as arch

    dataset = 'imagenet'
    model = 'resnet18'
    pretrained = True
    checkpoint_path = './checkpoint.best.pt'

    m = arch.create_model(model, dataset, pretrained=pretrained, checkpoint_path=checkpoint_path)
    # m takes weights from specified path


Initializing with pre-trained weights in ``timm``:

.. code-block:: python
    import architectures as arch

    dataset = 'cifar10'
    model = 'resnet18'
    pretrained = True

    m = arch.create_model(model, dataset, pretrained=pretrained)
    # m has pretrained weights as defined in timm


To perform inference:

.. code-block:: python
    # show an example of inference


To get intermediate layer representations:

.. code-block:: python
    # show an example of obtaining feature maps of ResNet


## Datasets

Standard dataloaders from torchvision + support for custom datasets. Many dataset classes taken from [robustness](https://github.com/MadryLab/robustness) and [robust-models-transfer](https://github.com/Microsoft/robust-models-transfer)

Initializing datasets:

.. code-block:: python
# show an example of initializing datasets


Datasets with custom data augmentations:

.. code-block:: python
# show the use of data augmentation callback


Model dependent data augmentations:

.. code-block:: python
# example of model dependent data aug, eg: worst-of-K spatial augmentations



## Training

Supports different losses (eg: adversarial training), different optimizers (all included in ``timm``)

Supervised: Uses [PyTorch-Lightning](https://github.com/facebookresearch/vissl) for easy, scalable training. 

Self-supervised: Uses [vissl](https://github.com/facebookresearch/vissl) to track SOTA models and weights.

Uses [robustness](https://github.com/MadryLab/robustness) for attack module used in adversarial training.

Example of supervised training (standard):

.. code-block:: python
    # show an example os using PyTorch-Lightning here


Example of supervised training (adversarial):

.. code-block:: python
    # show an example of adversarial training


## Adversarial Attacks

Attack module in [robustness](https://github.com/MadryLab/robustness). Also includes spatial attacks, taken from [adversarial_spatial](https://github.com/MadryLab/adversarial_spatial).

To do adversarial attack on models, wrap them in the ``Attack`` module. Here's an example:

.. code-block:: python
    # Show how to do L_inf PGD attack on an ImageNet ResNet


To do spatial attacks:

.. code-block:: python
    # Show an example of worst-of-K and first order spatial attacks



