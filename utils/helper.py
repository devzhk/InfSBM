import argparse
import torch


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def save_ckpt(path,
              model,
              #   model_ema,
              optim=None,
              scheduler=None,
              args=None):
    '''
    saves checkpoint and configurations to dir/name
    :param args: dict of configuration
    :param g_ema: moving average

    :param optim:
    '''
    ckpt_path = path
    if args and args.distributed:
        model_ckpt = model.module
    else:
        model_ckpt = model
    state_dict = {
        'model': model_ckpt.state_dict(),
        # 'ema': model_ema.state_dict(),
        'args': args
    }

    if optim is not None:
        state_dict['optim'] = optim.state_dict()
    if scheduler is not None:
        state_dict['scheduler'] = scheduler.state_dict()

    torch.save(state_dict, ckpt_path)
    print(f'checkpoint saved at {ckpt_path}')
