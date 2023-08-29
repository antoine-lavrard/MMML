from torch.optim.lr_scheduler import LRScheduler
from torch.optim import lr_scheduler
import warnings
import math


class SGDR(LRScheduler):
    """
    Implementation of SGDR with k-decay

    Code & documentation adpated from timm :
    https://github.com/pprp/timm/blob/master/timm/scheduler/cosine_lr.py
    https://timm.fast.ai/SGDR

    Args :
        T0 : number of epochs before first restart
        cycle_mul : multiply the number of epochs at each restart (Default: 1)
        cycle_decay : multiply the lr at each restart (Default: 1)
        lr_min: minimum value of the lr (Default: 0)
        number_restart: number of restarts (Default: 0)
        k_decay : k-decay option based on k-decay: A New Method For Learning Rate Schedule - https://arxiv.org/abs/2004.05909

        warmup_t: number of epoch for warmup (Default: None)
        warmup_lr_init: initial value for the warmup (Default: 0)
        warmup_prefix : if true, the epoch of the warmup are
            deduced from the first cosinus anealing (Default: False)
    """

    def __init__(
        self,
        optimizer,
        T0=1,
        cycle_mul=1,
        cycle_decay=1,
        lr_min=0,
        k_decay=1,
        number_restart=0,
        warmup_t=0,
        warmup_lr_init=0,
        warmup_prefix=False,
        last_epoch=-1,
        verbose=False,
    ):
        self.lr_min = lr_min
        self.T0 = T0
        self.cycle_mul = cycle_mul
        self.cycle_decay = cycle_decay
        self.number_restart = number_restart

        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix

        self.k_decay = k_decay
        super().__init__(optimizer, last_epoch, verbose)

        if self.warmup_t != 0:
            self.warmup_steps = [
                (v - warmup_lr_init) / self.warmup_t for v in self.base_lrs
            ]
        else:
            self.warmup_steps = [1 for _ in self.base_lrs]

    def get_lr(self):
        """
        Adapted from timm
        """
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        t = self.last_epoch

        if (t < self.warmup_t) and (self.warmup_t != 0):
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.cycle_mul != 1:
                i = math.floor(
                    math.log(1 - t / self.T0 * (1 - self.cycle_mul), self.cycle_mul)
                )
                t_i = self.cycle_mul**i * self.T0
                t_curr = t - (1 - self.cycle_mul**i) / (1 - self.cycle_mul) * self.T0
            else:
                i = t // self.T0
                t_i = self.T0
                t_curr = t - (self.T0 * i)

            gamma = self.cycle_decay**i
            lr_max_values = [v * gamma for v in self.base_lrs]
            k = self.k_decay

            if i < self.number_restart:
                lrs = [
                    self.lr_min
                    + 0.5
                    * (lr_max - self.lr_min)
                    * (1 + math.cos(math.pi * t_curr**k / t_i**k))
                    for lr_max in lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_lrs]

        return lrs




# def get_polynomial_restart_with_decay(
#     optimizer,
#     gamma=0.9,
#     nu=0.65,
#     number_restart=4,
#     epoch_restart=100,
#     warmup_epochs=None,
# ):
#     """
#     Polynomial Learning Rate Policy with Warm Restart for Deep Neural Network

#     DOI : 10.1109/TENCON.2019.8929465
#     """

#     if warmup_epochs is not None:
#         scheduler_warmup = [
#             lr_scheduler.LinearLR(
#                 optimizer,
#                 start_factor=1.0 / (warmup_epochs + 1.0),
#                 end_factor=1,
#                 total_iters=warmup_epochs,
#             )
#         ]
#         constant_warmup = [
#             lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=warmup_epochs)
#         ]

#         milestones = [warmup_epochs] + [
#             warmup_epochs + (1 + i) * epoch_restart for i in range(number_restart)
#         ]
#     else:
#         constant_warmup = []
#         scheduler_warmup = []
#         milestones = [(1 + i) * epoch_restart for i in range(number_restart)]

#     cosinus_scheduler = lr_scheduler.SequentialLR(
#         optimizer,
#         schedulers=scheduler_warmup
#         + [
#             lr_scheduler.PolynomialLR(optimizer, total_iters=epoch_restart, power=gamma)
#             for i in range(number_restart + 1)
#         ],
#         milestones=milestones,
#     )

#     discounts = lr_scheduler.SequentialLR(
#         optimizer,
#         schedulers=constant_warmup
#         + [
#             lr_scheduler.ConstantLR(
#                 optimizer, factor=nu**i, total_iters=epoch_restart
#             )
#             for i in range(number_restart + 1)
#         ],
#         milestones=milestones,
#     )
#     # error when doing Chained_scheduler -> SequentialLr
#     scheduler = lr_scheduler.ChainedScheduler([cosinus_scheduler, discounts])

#     return scheduler
