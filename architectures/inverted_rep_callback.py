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
        seeds = torch.stack((self.seed.clone(),) * len(x)).to(x.device)
        ir, _ = self.attacker(seeds, target_rep, **kwargs)
        return x, ir

    def predict_step(self, batch, batch_idx, dataloader_idx: Optional[int] = None):
        x, y  = batch
        og, inverted_rep = self(x, True, **self.attack_kwargs).detach()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            all_inverted_reps = self.all_gather(inverted_rep)
            all_inverted_reps = torch.cat([all_inverted_reps[i] for i in \
                range(len(all_inverted_reps))])
            all_og = self.all_gather(og)
            all_og = torch.cat([all_og[i] for i in range(len(all_og))])
            return all_og, all_inverted_reps
        return og, inverted_rep
    
    def on_predict_epoch_end(self, results):
        for i in range(len(results)):
            cat_og, cat_ir = None, None
            for batch_res in results[i]:
                og, ir = batch_res
                cat_og = og if cat_og is None else torch.cat((cat_og, og))
                cat_ir = ir if cat_ir is None else torch.cat((cat_ir, ir))
            results[i] = (cat_og, cat_ir)
        return results