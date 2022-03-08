import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Dataset, Sampler
import pytorch_lightning as pl
from pytorch_lightning import utilities as pl_utils
from typing import Iterable, Optional, Sequence, Union
from collections.abc import Callable
from .dataset_metadata import DATASET_PARAMS


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, 
                 dataset_class: Callable[[], Dataset],
                 data_dir: str = '/tmp/', 
                 shuffle_train: bool = True, 
                 shuffle_val: bool = False, 
                 shuffle_test: bool = False, 
                 pin_memory: bool = True, 
                 workers: int = 30,
                 random_split: int = 0, 
                 batch_size: Optional[int] = None, 
                 batch_sampler: Optional[Union[Sampler[Sequence], Iterable[Sequence]]] = None, 
                 transform_train: Optional[Callable] = None, 
                 transform_test: Optional[Callable] = None, 
                 **dataset_kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        # if passing a custom dataset class, the params should be compatible with 
        # the setup() and prepare_data() functions in downstream data class
        self.dataset_class = dataset_class
        self.dataset_kwargs = dataset_kwargs
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.shuffle_test = shuffle_test
        self.pin_memory = pin_memory
        self.batch_sampler = batch_sampler
        self.workers = workers
        self.transform_train = transform_train
        self.transform_test = transform_test
        self.random_split = random_split

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, 
            batch_sampler=self.batch_sampler, shuffle=self.shuffle_train, 
            num_workers=self.workers, pin_memory=self.pin_memory)
    
    def val_loader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, 
            batch_sampler=self.batch_sampler, shuffle=self.shuffle_val, 
            num_workers=self.workers, pin_memory=self.pin_memory)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,             
            batch_sampler=self.batch_sampler, shuffle=self.shuffle_test, 
            num_workers=self.workers, pin_memory=self.pin_memory)

class CIFAR10DataModule(BaseDataModule):
    def __init__(self, dataset_class=datasets.CIFAR10, *args, **kwargs):
        super(CIFAR10DataModule, self).__init__(dataset_class, *args, **kwargs)
        if self.transform_test is None:
            self.transform_test = DATASET_PARAMS['cifar10']['transform_test']
        if self.transform_train is None:
            self.transform_train = DATASET_PARAMS['cifar10']['transform_train']
        if self.batch_size is None:
            self.batch_size = DATASET_PARAMS['cifar10']['batch_size']
    
    def prepare_data(self):
        ## only needed when data needs to be downloaded
        self.dataset_class(self.data_dir, train=True, download=True)
        self.dataset_class(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        pl_utils.seed.seed_everything(self.random_split)
        if stage in (None, 'fit'):
            full_ds = self.dataset_class(root=self.data_dir, train=True, 
                transform=self.transform_train, **self.dataset_kwargs)
            train_size = int(0.9*len(full_ds))
            self.train_ds, self.val_ds = random_split(full_ds, [train_size, len(full_ds) - train_size])
        if stage in (None, 'test'):
            self.test_ds = self.dataset_class(root=self.data_dir, train=False, transform=self.transform_test, **self.dataset_kwargs)

class CIFAR100DataModule(BaseDataModule):
    def __init__(self, dataset_class=datasets.CIFAR100, *args, **kwargs):
        super(CIFAR100DataModule, self).__init__(dataset_class, *args, **kwargs)
        if self.transform_test is None:
            self.transform_test = DATASET_PARAMS['cifar100']['transform_test']
        if self.transform_train is None:
            self.transform_train = DATASET_PARAMS['cifar100']['transform_train']
        if self.batch_size is None:
            self.batch_size = DATASET_PARAMS['cifar100']['batch_size']
    
    def prepare_data(self):
        ## only needed when data needs to be downloaded
        self.dataset_class(self.data_dir, train=True, download=True)
        self.dataset_class(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        pl_utils.seed.seed_everything(self.random_split)
        if stage in (None, 'fit'):
            full_ds = self.dataset_class(root=self.data_dir, train=True, 
                transform=self.transform_train, **self.dataset_kwargs)
            train_size = int(0.9*len(full_ds))
            self.train_ds, self.val_ds = random_split(full_ds, [train_size, len(full_ds) - train_size])
            self.val_ds.__setattr__('transform', self.transform_test)
        if stage in (None, 'test'):
            self.test_ds = self.dataset_class(root=self.data_dir, train=False, transform=self.transform_test, **self.dataset_kwargs)

class STL10DataModule(BaseDataModule):
    def __init__(self, dataset_class=datasets.STL10, *args, **kwargs):
        super(STL10DataModule, self).__init__(dataset_class, *args, **kwargs)
        if self.transform_test is None:
            self.transform_test = DATASET_PARAMS['stl10']['transform_test']
        if self.transform_train is None:
            self.transform_train = DATASET_PARAMS['stl10']['transform_train']
        if self.batch_size is None:
            self.batch_size = DATASET_PARAMS['stl10']['batch_size']
    
    def prepare_data(self):
        ## only needed when data needs to be downloaded
        self.dataset_class(self.data_dir, train=True, download=True)
        self.dataset_class(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        pl_utils.seed.seed_everything(self.random_split)
        if stage in (None, 'fit'):
            self.train_ds = self.dataset_class(root=self.data_dir, 
                split='train+unlabeled', transform=self.transform_train, 
                **self.dataset_kwargs)
            self.val_ds = self.dataset_class(root=self.data_dir, split='train', 
                transform=self.transform_test, **self.dataset_kwargs)
        if stage in (None, 'test'):
            self.test_ds = self.dataset_class(root=self.data_dir, split='test',
                transform=self.transform_test, **self.dataset_kwargs)


class ImageNetDataModule(BaseDataModule):
    def __init__(self, dataset_class=datasets.ImageNet, *args, **kwargs):
        super(ImageNetDataModule, self).__init__(dataset_class, *args, **kwargs)
        if self.transform_test is None:
            self.transform_test = DATASET_PARAMS['imagenet']['transform_test']
        if self.transform_train is None:
            self.transform_train = DATASET_PARAMS['imagenet']['transform_train']
        if self.batch_size is None:
            self.batch_size = DATASET_PARAMS['imagenet']['batch_size']
    
    def prepare_data(self):
        # ImageNet needs to be pre-downloaded; this step will unzip the directory
        self.dataset_class(root=self.data_dir, split='train')
        self.dataset_class(root=self.data_dir, split='val')

    def setup(self, stage: Optional[str] = None):
        pl_utils.seed.seed_everything(self.random_split)
        if stage in (None, 'fit'):
            full_ds = self.dataset_class(root=self.data_dir, split='train', 
                transform=self.transform_train, **self.dataset_kwargs)
            train_size = int(0.9*len(full_ds))
            self.train_ds, self.val_ds = random_split(full_ds, [train_size, len(full_ds) - train_size])
            self.val_ds.__setattr__('transform', self.transform_test)
        if stage in (None, 'test'):
            self.test_ds = self.dataset_class(root=self.data_dir, split='val', 
                transform=self.transform_test, **self.dataset_kwargs)


DATA_MODULES = {
    'imagenet': ImageNetDataModule, 
    'cifar10': CIFAR10DataModule, 
    'cifar100': CIFAR100DataModule, 
    'stl10': STL10DataModule
}