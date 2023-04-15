r"""Cosine Annealing Linear Warmup
This module implements the Cosine Annealing Linear Warmup Scheduler in PyTorch.
Original:
    https://paperswithcode.com/method/cosine-annealing
References:
    | https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    | https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup
"""

import math

from typing import Union, List

import torch
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingLinearWarmup(_LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            first_cycle_steps: int,
            min_lrs: Union[float, List[float]] = None,
            cycle_mult: float = 1.,
            warmup_steps: int = 0,
            gamma: float = 1.,
            last_epoch: int = -1,
            min_lrs_pow: int = None,
                 ):

        '''
        :param optimizer: warped optimizer
        :param first_cycle_steps: number of steps for the first scheduling cycle
        :param min_lrs: same as eta_min, min value to reach for each param_groups learning rate
        :param cycle_mult: cycle steps magnification
        :param warmup_steps: number of linear warmup steps
        :param gamma: decreasing factor of the max learning rate for each cycle
        :param last_epoch: index of the last epoch
        :param min_lrs_pow: power of 10 factor of decrease of max_lrs (ex: min_lrs_pow=2, min_lrs = max_lrs * 10 ** -2
        '''
        assert warmup_steps < first_cycle_steps, "Warmup steps should be smaller than first cycle steps"
        assert min_lrs_pow is None and min_lrs is not None or min_lrs_pow is not None and min_lrs is None, \
            "Only one of min_lrs and min_lrs_pow should be specified"
        
        # inferred from optimizer param_groups
        max_lrs = [g["lr"] for g in optimizer.state_dict()['param_groups']]

        if min_lrs_pow:
            min_lrs = [i * (10 ** -min_lrs_pow) for i in max_lrs]

        if min_lrs:
            assert len(min_lrs)==len(max_lrs),\
                "The length of min_lrs should be the same as max_lrs, but found {} and {}".format(
                    len(min_lrs), len(max_lrs)
                )

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lrs = max_lrs  # first max learning rate
        self.max_lrs = max_lrs  # max learning rate in the current cycle
        self.min_lrs = min_lrs  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super().__init__(optimizer, last_epoch)

        assert len(optimizer.param_groups) == len(max_lrs),\
            "Expected number of max learning rates provided ({}) to be the same as the number of groups parameters ({})".format(
                len(max_lrs), len(optimizer.param_groups))
        
        assert len(optimizer.param_groups) == len(min_lrs),\
            "Expected number of min learning rates provided ({}) to be the same as the number of groups parameters ({})".format(
                len(max_lrs), len(optimizer.param_groups))

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for i, param_groups in enumerate(self.optimizer.param_groups):
            param_groups['lr'] = self.min_lrs[i]
            self.base_lrs.append(self.min_lrs[i])

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for (max_lr, base_lr) in
                    zip(self.max_lrs, self.base_lrs)]
        else:
            return [base_lr + (max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for (max_lr, base_lr) in zip(self.max_lrs, self.base_lrs)]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lrs = [base_max_lr * (self.gamma ** self.cycle) for base_max_lr in self.base_max_lrs]
        self.last_epoch = math.floor(epoch)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
