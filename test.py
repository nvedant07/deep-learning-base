# quick testing script
from lightning_lite import utilities as ll_utils
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import BasePredictionWriter
from torchvision.transforms import transforms
from pytorch_lightning.core.lightning import LightningModule
from training import NicerModelCheckpointing, LitProgressBar
import torch
import architectures as arch
from attack.callbacks import AdvCallback
from attack.attack_module import Attacker
from architectures.callbacks import LightningWrapper, AdvAttackWrapper, LinearEvalWrapper
from architectures.inference import inference_with_features
from datasets.data_modules import DATA_MODULES
from datasets.dataset_metadata import DATASET_PARAMS
from self_supervised.simclr_datamodule import simclr_dm
from self_supervised.simclr_callback import SimCLRWrapper
from functools import partial
from typing import List, Any

dataset = 'cifar10'
# dataset = 'imagenet'

# model = 'vgg16'
model = 'resnet18'
# model = 'resnet50'
# model = 'convit_base'

checkpoint_path = '/NS/robustness_2/work/vnanda/adv-trades/checkpoints'\
                  '/resnet18/beta_1.0_eps_1.000/checkpoint.pt.best'
# checkpoint_path = './tests/weights/resnet18_cifar10.pt'
# checkpoint_path = './tests/weights/vgg16_cifar10.pt'
# checkpoint_path = './tests/weights/resnet18_l2eps3_imagenet.pt'
# checkpoint_path = ''

pretrained = True
seed = 2
devices = 1
num_nodes = 1
strategy = DDPPlugin(find_unused_parameters=True) if devices > 1 else None
max_epochs = 100
# max_epochs = DATASET_PARAMS[dataset]['epochs']

imagenet_path = '/NS/twitter_archive/work/vnanda/data'
data_path = '/NS/twitter_archive2/work/vnanda/data'

dm = DATA_MODULES[dataset](
    data_dir=imagenet_path if dataset == 'imagenet' else data_path,
    val_frac=0.,
    subset=100,
    batch_size=32)

m1 = arch.create_model(model, dataset, pretrained=pretrained,
                       checkpoint_path=checkpoint_path, seed=seed, 
                       callback=partial(AdvAttackWrapper, 
                                        return_adv_samples=True,
                                        dataset_name=dataset))

ll_utils.seed.seed_everything(seed, workers=True)

adv_callback = AdvCallback(constraint_train='2',
                           eps_train=1.,
                           step_size=1.,
                           iterations_train=1,
                           iterations_val=10,
                           iterations_test=10,
                           random_start_train=False,
                           random_restarts_train=0,
                           return_image=True)
# attack_kwargs = adv_callback.get_eval_kwargs()
# attacker = Attacker(m1.model, m1.normalizer)
# for x, y in dm.test_dataloader():
#     print (x.device)
#     attacker(x, y **attack_kwargs)

class RecordPredictions(BasePredictionWriter):
    def __init__(self, write_interval: str):
        super().__init__(write_interval)

    def write_on_epoch_end(self, trainer: Trainer, pl_module: LightningModule, 
                           predictions: List[Any], batch_indices: List[Any]):
        X, X_adv, X_adv_preds = None, None, None
        for pred in predictions:
            for p in pred:
                # predictions will all be on CPU since PredictionEpochLoop 
                # transfers results to CPU to not overflow memory
                x, (x_adv, x_adv_pred) = p 
                X = x.detach() if X is None else torch.cat((X, x.detach()))
                X_adv = x_adv.detach() if X_adv is None else torch.cat((X_adv, x_adv.detach()))
                X_adv_preds = x_adv_pred if X_adv_preds is None else \
                    torch.cat((X_adv_preds, x_adv_pred))
        
        print (f'Rank: {torch.distributed.get_rank()}')
        print (X.shape, X_adv.shape, X.device, X_adv.device)
        print (batch_indices)
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            X_all = [torch.zeros_like(X_adv_preds) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(X_all, X_adv_preds)
            X_all = torch.cat(X_all, 0)
            print (f'All collected tensors: {X_all.shape}')

trainer = Trainer(accelerator='gpu', devices=devices,
                  num_nodes=num_nodes,
                  strategy=strategy, 
                  log_every_n_steps=1,
                  auto_select_gpus=True, deterministic=True,
                  max_epochs=1,
                  check_val_every_n_epoch=1,
                #   limit_val_batches=.5,
                  num_sanity_val_steps=0,
                  callbacks=[LitProgressBar(['loss', 'running_acc_clean', 'running_acc_adv']), 
                             adv_callback, 
                            #  RecordPredictions('epoch')
                             ])
# ## always use ddp -- works much faster, does not split batches
print (f'Global rank: {trainer.global_rank}, local rank: {trainer.local_rank}')
## DDP will spawn multiple processes
out = trainer.predict(m1, dataloaders=[dm.val_dataloader()])

if trainer.is_global_zero:
    ## do things on the main process
    print (out[0][0].shape, out[0][1][0].shape, out[0][1][1].shape)
# print (trainer.predict_loop.predictions)
# print (f'Latest preds: {}')
# trainer.predict(m1, dataloaders=dm.test_dataloader())
