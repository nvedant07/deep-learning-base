'''
torchvision's ImageFolder does not allow for a custom class_to_idx to be passed to constructor
Here we extend torchvision's ImageFolder to include this functionality.
'''

from torch.utils.data import Dataset
from torchvision import datasets
from typing import Callable, Any, Optional, Tuple, List, Dict
import os

class CustomImageFolder(datasets.DatasetFolder):

    def __init__(self, root: str, 
                 loader: Callable[[str], Any], 
                 transform: Optional[Callable] = None, 
                 target_transform: Optional[Callable] = None, 
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 find_classes_fun: Optional[Callable] = None) -> None:
        '''
        find_classes_fun: takes in a list of sorted directories and assigns 
            name of directory to an integer index and return it as a dictionary
        '''
        self.find_classes_fun = find_classes_fun
        super().__init__(root, loader,
            datasets.folder.IMG_EXTENSIONS if is_valid_file is None else None, 
            transform, target_transform, is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        class_to_idx = self.find_classes_fun(classes)
        return classes, class_to_idx
