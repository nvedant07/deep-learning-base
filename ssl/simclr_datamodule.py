from datasets.dataset_metadata import SIMCLR_TRAIN_TRANSFORMS
from .contrastive_data import ContrastiveTransformations


def simclr_dm(datamodule, s=1, views=2):
    datamodule.__setattr__('transform_train', ContrastiveTransformations(
            SIMCLR_TRAIN_TRANSFORMS(datamodule.input_size, s), views))
    datamodule.setup() ## need to re-initialize train_ds

