import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import wandb
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from classification.models import resnet32
from plots.thop import profile, clever_format

if __name__ == '__main__':
    plt.style.context("seaborn-pastel")
    
    model = resnet32()
    bs = 100
    input = torch.randn(bs, 3, 32, 32)
    total_ops, total_params, ret_dict = profile(model, inputs=(input,), ret_layer_info=True)
    
    layer_ops = {}
    
    for n, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm)):
            layer_ops[n] = m._buffers["total_ops"].item() * 2
    
    api = wandb.Api(timeout=60)
    runs = defaultdict()
    
    run = api.run(f"andreabrg/zero-grad-cifar-ablation/js7letd4")
    runs["adam"] = run.history()
    runs["adam"]["test.accuracy.top1"] *= 100
    
    run = api.run(f"andreabrg/zero-grad-cifar-ablation/xmgdjb9u")
    runs["sgd"] = run.history()
    runs["sgd"]["test.accuracy.top1"] *= 100
    
    print()
