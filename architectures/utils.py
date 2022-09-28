from enum import Enum
from re import S
import torch as ch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models.feature_extraction import get_graph_node_names

class InputNormalize(pl.LightningModule):
    '''
    A module (custom layer) for normalizing the input to have a fixed 
    mean and standard deviation (user-specified).
    '''
    def __init__(self, new_mean, new_std):
        super().__init__()
        new_std = new_std[..., None, None]
        new_mean = new_mean[..., None, None]

        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        x = ch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean)/self.new_std
        return x_normalized


class FlattenNormalizeConcatenate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten(1)
        self.eps = 1e-10 # to make sure we don't divide by zero
    
    def forward(self, x):
        ## x is a list of features
        new_x = []
        for ft in x:
            flat_ft = self.flatten(ft)
            new_x.append(flat_ft/(
                ch.linalg.norm(flat_ft, ord=2, dim=0) + self.eps)
                )
        return ch.cat(new_x, dim=1)


def intermediate_layer_names(model: nn.Module):
    filtered_nodes = []
    if model.__class__.__name__ == 'VGG':
        filtered_nodes = [x['module'] for x in model.feature_info]+ ['pre_logits', 'head.global_pool']
    elif model.__class__.__name__ == 'VisionTransformer':
        _, node_names = get_graph_node_names(model)        
        block_number_to_layer = {}
        for n in node_names:
            if n.startswith('blocks.'):
                current_block = int(n.split('blocks.')[1].split('.')[0])
                if current_block not in block_number_to_layer:
                    block_number_to_layer[current_block] = [n]
                else:
                    block_number_to_layer[current_block].append(n)
            if n in ['fc_norm']:
                filtered_nodes.append(n)
        for block in sorted(block_number_to_layer.keys(), reverse=True):
            filtered_nodes = [block_number_to_layer[block][-1]] + filtered_nodes
    elif model.__class__.__name__ == 'ResNetV2' or model.__class__.__name__ == 'ResNet':
        _, node_names = get_graph_node_names(model)        
        for idx, n in enumerate(node_names):
            if n.endswith('.pool'):
                filtered_nodes.append(n)
            elif n.endswith('.add'):
                if len(node_names) > idx + 1 and node_names[idx+1].endswith('.act3'):
                    filtered_nodes.append(node_names[idx+1])
                else:
                    filtered_nodes.append(n)
    return filtered_nodes

def calc_est_grad(func, x, y, rad, num_samples):
    B, *_ = x.shape
    Q = num_samples//2
    N = len(x.shape) - 1
    with ch.no_grad():
        # Q * B * C * H * W
        extender = [1]*N
        queries = x.repeat(Q, *extender)
        noise = ch.randn_like(queries)
        norm = noise.view(B*Q, -1).norm(dim=-1).view(B*Q, *extender)
        noise = noise / norm
        noise = ch.cat([-noise, noise])
        queries = ch.cat([queries, queries])
        y_shape = [1] * (len(y.shape) - 1)
        l = func(queries + rad * noise, y.repeat(2*Q, *y_shape)).view(-1, *extender) 
        grad = (l.view(2*Q, B, *extender) * noise.view(2*Q, B, *noise.shape[1:])).mean(dim=0)
    return grad

def accuracy(output, target, topk=(1,), exact=False):
    """
        Computes the top-k accuracy for the specified values of k

        Args:
            output (ch.tensor) : model output (N, classes) or (N, attributes) 
                for sigmoid/multitask binary classification
            target (ch.tensor) : correct labels (N,) [multiclass] or (N,
                attributes) [multitask binary]
            topk (tuple) : for each item "k" in this tuple, this method
                will return the top-k accuracy
            exact (bool) : whether to return aggregate statistics (if
                False) or per-example correctness (if True)

        Returns:
            A list of top-k accuracies.
    """
    with ch.no_grad():
        # Binary Classification
        if len(target.shape) > 1:
            assert output.shape == target.shape, \
                "Detected binary classification but output shape != target shape"
            return [ch.round(ch.sigmoid(output)).eq(ch.round(target)).float().mean()], [-1.0] 

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        res_exact = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float()
            ck_sum = correct_k.sum(0, keepdim=True)
            res.append(ck_sum.mul_(100.0 / batch_size))
            res_exact.append(correct_k)

        if not exact:
            return res
        else:
            return res_exact


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.

    def update(self, val, n=1.):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)
