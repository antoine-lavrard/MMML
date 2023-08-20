import torch
from torchmetrics import Metric
from torch import nn
import inspect


class Projection(nn.Module):
    """
    Projection : adapts the feature maps from a backbone
    in order to be compatible with a classification task
    """

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


from torch.nn.functional import one_hot


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
    def __init__(self, loss_function, type_loss="classic") -> None:
        """
        Wrap a loss function in order to apply it to mixed target, and save it
        Loss must be unreduced
        """
        super().__init__()
        any(
            [parent_cls is Metric for parent_cls in loss_function.__class__.__mro__]
        ), "WrapperLossMixup take a function"

        self.loss_function = loss_function
        self.add_state(
            "loss", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum"
        )

        self.type_loss = type_loss

    def update(self, preds: torch.Tensor, target: torch.Tensor, lmbd, index):
        """
        Args :
            preds (torch.Tensor(B,C)): raw predictions
            target (np.ndarray(B,3)): raw labels, permuted labels and lambdas for each mix
        """
        if self.type_loss == "classic":
            loss_a = self.loss_function(
                preds, target.long()
            )  # assuming second part requires long input
            loss_b = self.loss_function(preds, target[index].long())
            loss = (lmbd * loss_a + (1 - lmbd) * loss_b).mean()
        elif self.type_loss == "hard":
            loss_a = self.loss_function(
                preds, target.long()
            )  # assuming second part requires long input
            loss_b = self.loss_function(preds, target[index].long())
            loss = ((lmbd <= 0.5) * loss_a + (lmbd > 0.5) * loss_b).mean()
        elif self.type_loss == "target_mix":
            target1 = one_hot(target.long())
            target2 = one_hot(target[index].long())
            loss = self.loss_function(
                preds, lmbd * target1 + (1 - lmbd) * target2
            ).mean()
        elif self.type_loss == "hard_target_mix":
            target1 = one_hot(target)
            target2 = one_hot(target[index])
            loss = self.loss_function(preds, (target1 + target2) * 1.0).mean()
        elif self.type_loss == "hard_mix":
            loss_a = self.loss_function(
                preds, target.long()
            )  # assuming second part requires long input
            loss_b = self.loss_function(preds, target[index].long())
            loss = (loss_a + loss_b).mean()

        self.loss += loss
        self.total += 1

        return loss

    def compute(self):
        return self.loss.float() / self.total


class MixCrossEntropyLoss(WrapperLossMixup):
    """
    Adaptation of the CrossEntropyLoss for mixed image.
    """

    def __init__(self, *args, type_loss="classic", **kwargs):
        if "reduction" in kwargs.keys():
            raise ValueError("reduction not supported")
        crossentropy = nn.CrossEntropyLoss(*args, **kwargs, reduction="none")
        super().__init__(crossentropy, type_loss=type_loss)
