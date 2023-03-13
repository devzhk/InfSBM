import os
from omegaconf import OmegaConf
from argparse import ArgumentParser
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from models.mlp import VEPrecond, MLP

from train_utils.loss import VELoss

from utils.helper import save_ckpt
from utils.datasets import Gaussian



def train_fn(model, trainloader, optimizer, config, device):
    # setup directories
    basedir = config.log.basedir
    basedir = os.path.join('exp', basedir)
    os.makedirs(basedir, exist_ok=True)
    ckpt_dir = os.path.join(basedir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    fig_dir = os.path.join(basedir, 'figs')
    os.makedirs(fig_dir, exist_ok=True)
    # plot dataset samples
    data_samples = trainloader.dataset.data.numpy()

    # parse args
    grad_clip = config.optim.grad_clip
    num_epoch = config.train.num_epoch
    save_step = config.train.save_step
    eval_step = config.train.eval_step
    # setup sde
    loss_fn = VELoss(sigma_max=config.model.sigma_max, sigma_min=config.model.sigma_min)\

    # ---------
    pbar = tqdm(range(num_epoch))
    for e in pbar:
        train_loss = 0.0
        for data in trainloader:
            optimizer.zero_grad()
            data = data.to(device)
            loss = loss_fn(model, data)
            loss.backward()
            if grad_clip > 0.:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=grad_clip)
            optimizer.step()
            train_loss += loss.item()
        avg_loss = train_loss / len(trainloader)
        pbar.set_description(
            (
                f'Epoch: {e}. Train loss: {avg_loss}'
            )
        )
        if e % save_step == 0 and e > 0:
            ckpt_path = os.path.join(ckpt_dir, f'model-{e}.pt')
            save_ckpt(ckpt_path, model, optim=optimizer)
    print('Finish training')


def subprocess(args):
    config = OmegaConf.load(args.config) 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MLP(config).to(device)
    optimizer = Adam(model.parameters(), lr=config.optim.lr)

    mean = torch.tensor(config.data.mean, dtype=torch.float32)
    std = torch.tensor(config.data.std, dtype=torch.float32)
    dataset = Gaussian(num_samples=config.data.num_samples,
                       mean=mean, std=std)
    trainloader = DataLoader(
        dataset, batch_size=config.train.batchsize, shuffle=True)

    train_fn(model, trainloader, optimizer, config, device)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/gaussian.yml')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    subprocess(args)
