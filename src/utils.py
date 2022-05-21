import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data


def int2bool(i):
    i = int(i)
    assert i == 0 or i == 1
    return i == 1


def str2activation(activation):
    activations = {
        'relu': torch.nn.ReLU,
        'tanh': torch.nn.Tanh
    }
    return activations[activation]


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)
