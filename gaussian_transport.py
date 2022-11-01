import os
import yaml
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchsde
from sde import OUSDE


def transport(a, var1,
              b=0.0, var2=1.0,
              N=101, T=101, plot_path='default.png',
              plot_title='test'):
    '''
    transport from N(a, var1) to N(b, var2)

    a = b: r = 1
    a != b: m = 2a -b

    Args:
        - a: mean of x(0)
        - var1: std of x(0)
        - b: mean of x(1)
        - var2: std of x(1)
        - N: number of discrete steps
    '''
    const_mean = (a == b)
    if const_mean:
        r = 1
        m = 0
    else:
        eps = 1e-2
        r = min(2, (1-eps) * np.sqrt(var2 / var1))
        m = (r * a - b) / (r - 1)

    def density(x, t):

        if const_mean:
            m_t = a
            v_t = var1 + t * (var2 - var1)
        else:
            m_t = np.power(r, t) * (a-m) + m
            r2var1 = r * r * var1
            r2t = np.power(r, 2 * t)
            v_t = r2t * var1 + (var2 - r2var1) * (r2t - 1) / (r*r - 1)

        denorm = np.sqrt(2 * np.pi * v_t)

        pdf = np.exp(- np.power(x - m_t, 2) / (2 * v_t))
        return pdf / denorm

    x_min = min(a, b) - 2
    x_max = max(a, b) + 2
    x = np.linspace(x_min, x_max, N).reshape(N, 1)
    t = np.linspace(0, 1, T).reshape(1, T)

    pdf_map = density(x, t)
    fig, ax = plt.subplots()
    im = ax.imshow(pdf_map, origin='lower')
    ax.set_title(plot_title)
    ax.set_xlabel('Time')
    ax.set_ylabel('x')
    ax.set_xticks(ticks=[0, T - 1])
    ax.set_yticks(ticks=[0, N - 1])
    ax.set_xticklabels(['0', '1'])
    ax.set_yticklabels([f'{x_min:.2f}', f'{x_max:.2f}'])
    fig.colorbar(im, ax=ax)

    # # SDE simulation
    # dirft = np.log(r)
    # shift = 2 * a - b
    # diff2 = 2 * np.log(r) * (var2 - r * r * var1) / (r*r - 1)
    # fw_sde = OUSDE(dirft, shift, np.sqrt(diff2))

    # batch_size, state_size, t_size = 6, 1, T
    # ts = torch.linspace(0, 1, t_size)
    # # y0 = torch.full(size=(batch_size, state_size), fill_value=0.1)

    # y0 = a + np.sqrt(var1) * torch.randn((batch_size, state_size))
    # with torch.no_grad():
    #     # (t_size, batch_size, state_size) = (100, 3, 1).
    #     ys = torchsde.sdeint(fw_sde, y0, ts, method='euler')
    # ys = ys.squeeze().t()
    # for i, path in enumerate(ys):
    #     ax.plot(ts * (T-1), (path + x_max) * N / (x_max - x_min),
    #             label=f'sample path {i}')
    # plt.legend(prop={'size': 6})

    plt.savefig(plot_path)
    plt.close()


def subprocess(args):
    base_dir = os.path.join('exp', args.logdir)
    os.makedirs(base_dir, exist_ok=True)

    a, var1 = 3.0, 1.0
    b, var2 = - 3.0, 1.0
    # mean1s =
    for i in range(1, 4):

        var1i = var1 / (i * i)
        var2i = var2 / (1)
        # var2i = var2 / (i * i)

        plot_path = os.path.join(base_dir, f'{i}.png')
        plot_title = f'0: N({a}, {var1} / {i*i}); 1: N({b}, {var2})'
        transport(a, var1i, b, var2i,
                  plot_path=plot_path, plot_title=plot_title)


if __name__ == '__main__':
    parser = ArgumentParser(description='parser for toy problem')
    parser.add_argument('--logdir', type=str, default='default')
    parser.add_argument('--config', type=str, default='config/const.yaml')
    args = parser.parse_args()
    subprocess(args)
