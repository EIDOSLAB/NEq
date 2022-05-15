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
    neurons = {}
    
    for n, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm)):
            layer_ops[n] = m._buffers["total_ops"].item() * 2
            neurons[n] = m.weight.shape[0]
    
    api = wandb.Api(timeout=60)
    run = api.run(f"andreabrg/zero-grad-cifar-ablation/yhzr707b")
    config = json.loads(run.json_config)
    df = run.history()

    flops = []

    for i in range(len(df.index)):
    
        remaining_ops = 0
    
        for layer in layer_ops:
            if f"frozen_neurons_perc.layer.{layer}" in df:
                ops = layer_ops[layer]
                frozen_ops = ops * (df[f"frozen_neurons_perc.layer.{layer}"].iloc[[i]].values[0] / 100)
                remaining_ops += ops - frozen_ops
    
        flops.append(remaining_ops)

    fig, ax1 = plt.subplots(figsize=(9, 3))
    ax1.plot(df["lr"], alpha=0.7, color='#1f77b4')
    ax2 = ax1.twinx()
    ax2.plot(flops, alpha=0.7, color='#ff7f0e')
    ax1.set_ylabel('Learning Rate', color='#1f77b4')
    ax2.set_ylabel('FLOPs', color='#ff7f0e')
    ax1.set_xlabel("Epochs")
    fig.tight_layout()
    plt.savefig("lr-vs-flops.png", dpi=300)
    plt.savefig("lr-vs-flops.pdf", dpi=300)
    plt.clf()
