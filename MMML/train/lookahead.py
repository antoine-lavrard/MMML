"""
copied from https://github.com/michaelrzhang/lookahead/
(only added set_to_none to zero_grad method & transfer slow weight if needed)
"""
from collections import defaultdict
from typing import Any, Dict

import torch
from torch.optim.optimizer import Optimizer

from MMML.utils import transfer_to


class Lookahead(Optimizer):
    r"""PyTorch implementation of the lookahead wrapper.

    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    """

    def __init__(self, optimizer, la_steps=5, la_alpha=0.8, pullback_momentum="none"):
        """optimizer: inner optimizer
        la_steps (int): number of lookahead steps
        la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
        pullback_momentum (str): change to inner optimizer momentum on interpolation update
        """
        self.optimizer = optimizer
        self._la_step = 0  # counter for inner optimizer
        self.la_alpha = la_alpha
        self._total_la_steps = la_steps
        pullback_momentum = pullback_momentum.lower()
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum

        self.state = defaultdict(dict)
        self.defaults = defaultdict(dict)
        # Cache the current optimizer parameters
        for group in optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["cached_params"] = torch.zeros_like(p.data)
                param_state["cached_params"].copy_(p.data)
                if self.pullback_momentum == "pullback":
                    param_state["cached_mom"] = torch.zeros_like(p.data)
        self.current_device = p.data.device
    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.state, self.optimizer, self.la_alpha, self._la_step, self._total_la_steps, self.pullback_momentum = state["state"], state["optimizer"], state["la_alpha"], state["_la_step"], state["_total_la_steps"], state["pullback_momentum"]
        
    def __getstate__(self):
        return {
            "state": self.state,
            "optimizer": self.optimizer,
            "la_alpha": self.la_alpha,
            "_la_step": self._la_step,
            "_total_la_steps": self._total_la_steps,
            "pullback_momentum": self.pullback_momentum,
        }

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def get_la_step(self):
        return self._la_step

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)"""

        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                param_state["backup_params"] = torch.zeros_like(p.data)
                param_state["backup_params"].copy_(p.data)
                p.data.copy_(param_state["cached_params"])

        self.current_device = p.data.device

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_state = self.state[p]
                p.data.copy_(param_state["backup_params"])
                del param_state["backup_params"]
        self.current_device = p.data.device

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self.optimizer.step(closure)
        self._la_step += 1

        if self._la_step >= self._total_la_steps:
            self._la_step = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    param_state = self.state[p]
                    if param_state != p.data.device:
                        param_state = transfer_to(param_state, p.data.device)
                    p.data.mul_(self.la_alpha).add_(
                        param_state["cached_params"], alpha=1.0 - self.la_alpha
                    )  # crucial line
                    param_state["cached_params"].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p][
                            "momentum_buffer"
                        ] = internal_momentum.mul_(self.la_alpha).add_(
                            1.0 - self.la_alpha, param_state["cached_mom"]
                        )
                        param_state["cached_mom"] = self.optimizer.state[p][
                            "momentum_buffer"
                        ]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(
                            p.data
                        )

        return loss
