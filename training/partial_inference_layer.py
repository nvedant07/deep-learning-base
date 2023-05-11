from typing import Union, Any, Optional, List
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch


class ModifiedLinear(nn.Module):

    def __init__(self) -> None:
        super().__init__()
    
    def _copy_linear_features(self, mod):
        fts = ['in_features', 'out_features']
        for ft in fts:
            self.__setattr__(ft, mod.__dict__[ft])
    
    def forward(self, x):
        # should be implmented as necessary by inheriting module
        pass


class PartialLinear(ModifiedLinear):

    def __init__(self, neuron_indices: Union[torch.Tensor, list], linear: nn.Linear) -> None:
        """
        neuron_indices: indices of neurons used to do inference
        """
        super().__init__()
        self.neuron_indices = neuron_indices.clone().detach() if \
            isinstance(neuron_indices, torch.Tensor) else torch.tensor(neuron_indices)
        self.linear = linear
        self._copy_linear_features(linear)

    def forward(self, x):
        return self.linear(x[:,self.neuron_indices])


class PCALinear(ModifiedLinear):

    def __init__(self, num_neurons: int, linear: nn.Linear, projection_matrix: torch.Tensor) -> None:
        """
        projection_matrix: needs to be generated using PCA/some other dimensionality reduction method.
            shape = num_fts x num_fts
        """
        super().__init__()
        self.num_neurons = num_neurons
        self.linear = linear
        self.register_buffer('projection', projection_matrix[:,:self.num_neurons])
        self._copy_linear_features(linear)

    def forward(self, x):
        return self.linear(x @ self.projection)


class EnsembleHead(nn.Module):
    """
    Takes a bunch of predictor heads and outputs the average scores of each head
    """
    def __init__(self, all_heads, ensemble_type='soft'):
        super().__init__()
        self.num_heads = len(all_heads)
        self.ensemble_type = ensemble_type
        for idx, head in enumerate(all_heads):
            self.add_module(f'head_{idx}', head)

    def forward(self, x):
        if self.ensemble_type == 'soft':
            op = torch.mean(
                torch.stack([self.__getattr__(f'head_{i}')(x) for i in range(self.num_heads)], dim=1), 
                dim=1)
        else:
            op = torch.stack([self.__getattr__(f'head_{i}')(x) for i in range(self.num_heads)], dim=1)
        return op


class HardEnsembleLoss(_Loss):
    def __init__(self, base_loss: Optional[_Loss], *args, **kwds) -> None:
        super().__init__(*args, **kwds)
        self.base_loss = base_loss

    def forward(self, outputs: torch.Tensor, *args: Any, **kwds: Any) -> torch.Tensor:
        assert len(outputs.shape) == 3, \
            f'input should be [batch, num_ensembles, output_dim], found {outputs.shape} instead'
        num_heads = outputs.shape[1]
        head_wise_loss = []
        for i in range(num_heads):
            head_wise_loss.append(self.base_loss(outputs[:,i,:], *args, **kwds))
        return torch.mean(torch.stack(head_wise_loss, dim=-1), dim=-1)
