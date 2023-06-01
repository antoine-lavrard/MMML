import torch
from torchmetrics import Metric
from torch import nn
import inspect

class Projection(nn.Module):
    def __init__(self, n_features, n_outputs):
        super().__init__()
        self.mean_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)
        self.linear = nn.Linear(n_features, n_outputs)

    def forward(self, x):
        x = self.mean_pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
    
class WrappedLossFunction(Metric):
    """
    Wrap a loss function to compute a loss into a metric
    The loss is aggregated over multiple batches
    """

    def __init__(self, loss_function) -> None:
        super().__init__()
        self.loss_function = loss_function
        self.add_state(
            "loss", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        loss = self.loss_function(preds, target)
        self.loss += loss
        self.total += 1
        return loss

    def compute(self):
        return self.loss.float() / self.total


class WrapperLossMixup(Metric):
    """
    Wrap a loss function in order to apply it to mixed target, and save it
    Loss must be unreduced
    """

    def __init__(self, loss_function) -> None:
        super().__init__()
        
        if inspect.isclass(loss_function):
            assert not any([isinstance(loss_function.__mro__, Metric)]), "WrapperLossMixup take a function"
        self.loss_function = loss_function
        self.add_state(
            "loss", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        Args :
            preds (torch.Tensor(B,C)): raw predictions
            target (np.ndarray(B,3)): raw labels, permuted labels and lambdas for each mix
        """

        loss_a = self.loss_function(preds, target[:, 0].long())
        loss_b = self.loss_function(preds, target[:, 1].long())
        loss = ((1 - target[:, 2]) * loss_a + target[:, 2] * loss_b).mean()
        self.loss += loss
        self.total += 1

        return loss

    def compute(self):
        return self.loss.float() / self.total
