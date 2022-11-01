import os
import yaml
from argparse import ArgumentParser


import torch
import torch.nn as nn
import torchsde
import numpy as np
from sde import OUSDE, ReverseSDE
from plots import plot
from scores import gaussian_fn
from functools import partial


def get_mean(t, r, a, shift):
    mean = np.power(r, t) * (a - shift) + shift
    return mean


def get_var(t, r, sigma2, gamma2):
    r2t = np.power(r, 2 * t)
    r2 = np.power(r, 2)
    var = r2t * sigma2 + (r2t - 1) * (gamma2 - r2 * sigma2) / (r2t - 1)
    return var


def reverse(a, b, var1, var2,
            plot_title,
            plot_dir=None):
    batch_size, state_size, t_size = 12, 1, 101
    eps = 1e-2
    r = min(2, (1-eps) * np.sqrt(var2/var1))
    drift = np.log(r)
    shift = (r * a - b) / (r - 1)
    diff2 = 2 * np.log(r) * (var2 - r * r * var1) / (r*r - 1)

    mean_fn = partial(get_mean, r=r, a=a, shift=shift)
    var_fn = partial(get_var, r=r, sigma2=var1, gamma2=var2)
    score_fn = partial(gaussian_fn, mean=mean_fn, variance=var_fn)

    rsde = ReverseSDE(drift, shift, np.sqrt(diff2), ratio=r, score=score_fn)
    ts = torch.linspace(0, 1, t_size)
    y0 = b + np.sqrt(var2) * torch.randn((batch_size, state_size))
    with torch.no_grad():
        ys = torchsde.sdeint(rsde, y0, ts, method='euler')

    plot_path = os.path.join(plot_dir, 'backward.png')
    plot(ts, ys, xlabel='$t$', ylabel='$Y_t$',
         title=plot_title, savepath=plot_path)


def advance(a, b, var1, var2, plot_title, plot_dir=None):
    batch_size, state_size, t_size = 12, 1, 101
    # a, b = 3., -3.
    # var1, var2 = 0.1, 1.0
    eps = 1e-2
    r = min(2, (1-eps) * np.sqrt(var2 / var1))
    m = (r * a - b) / (r - 1)

    dirft = np.log(r)
    shift = m
    diff2 = 2 * np.log(r) * (var2 - r * r * var1) / (r*r - 1)

    sde = OUSDE(dirft, shift, np.sqrt(diff2))
    ts = torch.linspace(0, 1, t_size)
    # y0 = torch.full(size=(batch_size, state_size), fill_value=0.1)
    y0 = a + np.sqrt(var1) * torch.randn((batch_size, state_size))
    with torch.no_grad():
        # (t_size, batch_size, state_size) = (100, 3, 1).
        ys = torchsde.sdeint(sde, y0, ts, method='euler')

    plot_path = os.path.join(plot_dir, 'forward.png')
    plot(ts, ys, xlabel='$t$', ylabel='$Y_t$',
         title=plot_title, savepath=plot_path)


def subprocess(args):
    base_dir = os.path.join('exp', args.logdir)
    os.makedirs(base_dir, exist_ok=True)

    a, var1 = 3.0, 1.0
    b, var2 = - 3.0, 1.0
    # mean1s =
    i_list = [1, 10, 100, 1000]
    for i in i_list:

        var1i = var1 / (i * i)
        # var2i = var2
        var2i = var2 / (i * i)

        plot_dir = os.path.join(base_dir, f'decay-id{i}')
        os.makedirs(plot_dir, exist_ok=True)
        plot_title = f'0: N({a}, {var1} / {i*i}); 1: N({b}, {var2}/ {i * i})'
        reverse(a, b, var1i, var2i,
                plot_title=plot_title, plot_dir=plot_dir)
        advance(a, b, var1i, var2i,
                plot_dir=plot_dir, plot_title=plot_title)


if __name__ == '__main__':
    parser = ArgumentParser(description='parser for toy problem')
    parser.add_argument('--logdir', type=str, default='default')
    parser.add_argument('--config', type=str, default='config/const.yaml')
    args = parser.parse_args()
    subprocess(args)
