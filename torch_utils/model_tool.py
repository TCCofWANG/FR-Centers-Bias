import torch
from torch.utils import data

import numpy as np
import random


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def seed_everything(seed, benchmark=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if benchmark:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_item(data):
    if isinstance(data, np.ndarray):
        return data.reshape(1)[0]
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().item()
    else:
        return data


def to_tensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor


def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()


def separate_bn_param(module: torch.nn.Module):
    params_decay = []
    params_no_decay = []
    for m in module.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            params_no_decay.extend([*m.parameters()])
        elif len(list(m.children())) == 0:
            params_decay.extend([*m.parameters()])

    assert len(list(module.parameters())) == len(
        params_decay) + len(params_no_decay)
    return params_no_decay, params_decay


def stat_param_num(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)