###############
# callbacks.py offers callbacks for the arch.create_model function
# and hence are differnt from callbacks that are given to the trainer.
# this file contains callbacks that are given to the trainer to set
# params dynamicalle during training, similar to attack.callbacks.
###############

from typing import List
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.core.lightning import LightningModule

class ZeroShotCallback(Callback):
    """
    Used with arch.callbacks.MultimodalEvalWrapper
    to set class names at runtime.
    """
    def __init__(self, classes: List[str]) -> None:
        self.classes = classes

    def update_classes(self, new_classes: List[str]) -> None:
        self.classes = new_classes

    def on_predict_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module._set_classes(self.classes)
    
    def on_predict_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module._reset_classes()

