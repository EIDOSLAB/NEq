import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.utils.data


def arg2bool(val):
    if isinstance(val, bool):
        return val
    
    elif isinstance(val, str):
        if val == "true":
            return True
        
        if val == "false":
            return False
    
    val = int(val)
    assert val == 0 or val == 1
    return val == 1


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)
