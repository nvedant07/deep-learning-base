import torch
from typing import Optional
from .callbacks import AdvAttackWrapper
from .inference import inference_with_features

class InvertedRepWrapper(AdvAttackWrapper):

    def __init__(self, model: torch.nn.Module, seed: torch.Tensor, *args, **kwargs) -> None:
        super().__init__(model, *args, **kwargs)
        self.seed = seed # shape: (channels, width, height)
    
    def forward(self, x, *args, **kwargs):
        target_rep = inference_with_features(self.model, self.normalizer(x), *args)[1].detach()
        seeds = torch.cat((self.seed.clone(),) * len(x)).to(x.device)
        x, _ = self.attacker(seeds, target_rep, **kwargs)
        return x

    def predict_step(self, batch, batch_idx, dataloader_idx: Optional[int] = None):
        x, y  = batch
        inverted_rep = self(x, True, **self.attack_kwargs).detach()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            all_inverted_reps = self.all_gather(inverted_rep)
            all_inverted_reps = torch.cat([all_inverted_reps[i] for i in \
                range(len(all_inverted_reps))])
            return all_inverted_reps
        return inverted_rep
    
    def on_predict_epoch_end(self, results):
        for i in range(len(results)):
            cat_x = None
            for batch_res in results[i]:
                cat_x = batch_res if cat_x is None else torch.cat((cat_x, batch_res))
            results[i] = cat_x
        return results