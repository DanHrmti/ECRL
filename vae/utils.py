"""
Utility functions
"""

import datetime
import os
import json
# torch
import torch


def reparameterize(mu, logvar):
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variaance of x
    :return z: the sampled latent variable
    """
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(mu).to(device)
    return mu + eps * std


def prepare_logdir(runname, src_dir='./', accelerator=None):
    td_prefix = datetime.datetime.now().strftime("%d%m%y_%H%M%S")
    dir_name = f'{td_prefix}_{runname}'
    path_to_dir = os.path.join(src_dir, dir_name)
    path_to_fig_dir = os.path.join(path_to_dir, 'figures')
    path_to_save_dir = os.path.join(path_to_dir, 'saves')
    if accelerator is not None and accelerator.is_main_process:
        os.makedirs(path_to_dir, exist_ok=True)
        os.makedirs(path_to_fig_dir, exist_ok=True)
        os.makedirs(path_to_save_dir, exist_ok=True)
    elif accelerator is None:
        os.makedirs(path_to_dir, exist_ok=True)
        os.makedirs(path_to_fig_dir, exist_ok=True)
        os.makedirs(path_to_save_dir, exist_ok=True)
    else:
        pass
    return path_to_dir


def save_config(src_dir, hparams):
    path_to_conf = os.path.join(src_dir, 'hparams.json')
    with open(path_to_conf, "w") as outfile:
        json.dump(hparams, outfile, indent=2)


def get_config(fpath):
    with open(fpath, 'r') as f:
        config = json.load(f)
    return config


def log_line(src_dir, line):
    log_file = os.path.join(src_dir, 'log.txt')
    with open(log_file, 'a') as fp:
        fp.writelines(line)


class LinearWithWarmupScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(self, optimizer, gamma=0.95, steps=(1, 2), factors=(1.0, 0.1), verbose=False):
        self.steps = steps
        self.factors = factors
        self.gamma = gamma
        super().__init__(optimizer, verbose=verbose)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch < self.steps[0]:
            # warmup
            lr_factor = 1.0
        elif self.steps[0] <= epoch < self.steps[1]:
            # noisy
            lr_factor = self.factors[1]
        else:
            # standard scheduling
            lr_factor = self.gamma ** (epoch - self.steps[0])
        return lr_factor
