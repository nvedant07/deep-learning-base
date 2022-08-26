from typing import Union
import torch.nn as nn
import torch

class PartialLinear(nn.Module):

    def __init__(self, neuron_indices: Union[torch.Tensor, list], linear: nn.Linear) -> None:
        """
        neuron_indices: indices of neurons used to do inference
        """
        super().__init__()
        self.neuron_indices = neuron_indices.clone().detach() if \
            isinstance(neuron_indices, torch.Tensor) else torch.tensor(neuron_indices)
        self.linear = linear
        
    def forward(self, x):
        return self.linear(x[:,self.neuron_indices])


class EnsembleHead(nn.Module):
    """
    Takes a bunch of predictor heads and outputs the average scores of each head
    """
    def __init__(self, all_heads):
        super().__init__()
        self.all_heads = all_heads

    def forward(self, x):
        op = torch.mean(
            torch.stack([head(x) for head in self.all_heads], dim=1), 
            dim=1)
        print (op.shape)
        return op
