import os
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

from utils.helper import dict2namespace

import torch

from models.mlp import MLP
from models.scores import Gscore
from utils.sde_lib import VPSDE

from utils.datasets import Gaussian
from sampling import get_sampling_fn

from plots import plot_samples, plot_dict


def scalar(x):
    return x


def subprocess(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    config = dict2namespace(config)
    # set up directories
    basedir = config.log.basedir
    basedir = os.path.join('exp', basedir)
    os.makedirs(basedir, exist_ok=True)
    sample_dir = os.path.join(basedir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)

    # set up model
    model = MLP(config).to(device)
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt['model'])

    # create ground true
    mean = torch.tensor(config.data.mean, dtype=torch.float32)
    std = torch.tensor(config.data.std, dtype=torch.float32)

    true_model = Gscore(data_mean=mean, data_var=std * std)
    # set up original dataset
    dataset = Gaussian(num_samples=config.data.num_samples,
                       mean=mean, std=std)

    # set up sampling function
    eps = config.sampling.eps
    sde = VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max)
    sample_shape = (args.num_samples, config.data.dim)
    sample_fn = get_sampling_fn(
        config, sde, shape=sample_shape, inverse_scaler=scalar, eps=eps, device=device)

    # generate samples from learned model
    samples, n = sample_fn(model)

    # generate samples from true score
    true_samples, n = sample_fn(true_model)

    # visualize
    fig_path = os.path.join(sample_dir, 'generated_samples.png')

    data_dict = {
        'learned model': samples.cpu().numpy(),
        'true score': true_samples.cpu().numpy(),
        'data': dataset.data.numpy()
    }
    plot_dict(data_dict, fig_path)
    # plot_samples(samples.cpu().numpy(), true_samples.cpu().numpy(), fig_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/gaussian.yml')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt', type=str,
                        default='exp/train-256/ckpts/model-5000.pt')
    parser.add_argument('--num_samples', type=int, default=128)
    args = parser.parse_args()
    subprocess(args)
