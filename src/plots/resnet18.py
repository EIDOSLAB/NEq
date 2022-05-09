import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import wandb
from matplotlib import pyplot as plt
from torch import nn
from torchvision.models import resnet18
from tqdm import tqdm

from classification.models import resnet32
from plots.thop import profile


def plot_scatter_flops(runs, layer_ops):
    # plt.figure(figsize=(9, 3), dpi=600)
    
    total_ops = sum(layer_ops.values())
    
    for eps in runs:
        mean = runs[eps].groupby(level=0).mean()
        std = runs[eps].groupby(level=0).std()
        min = runs[eps].groupby(level=0).min()
        max = runs[eps].groupby(level=0).max()
        
        remaining_ops = 0
        
        for layer in layer_ops:
            if f"frozen_neurons_perc.layer.{layer}" in mean:
                ops = layer_ops[layer]
                frozen_ops = ops * (mean[f"frozen_neurons_perc.layer.{layer}"] / 100)
                remaining_ops += ops - frozen_ops
        
        plt.errorbar(remaining_ops.mean() / total_ops * 100, mean["test.accuracy.top1"].iloc[[-1]],
                     label=f"eps={eps}", alpha=0.7, fmt="o", linewidth=1)
    
    plt.legend()
    plt.xlabel("Remaining FLOPS (%)")
    plt.ylabel("Classification Accuracy (%)")
    plt.tight_layout()
    plt.show()


def plot_scatter_frozen(runs):
    # plt.figure(figsize=(9, 3), dpi=600)
    
    mean = runs[eps].groupby(level=0).mean()
    std = runs[eps].groupby(level=0).std()
    min = runs[eps].groupby(level=0).min()
    max = runs[eps].groupby(level=0).max()
    
    print(eps, mean["frozen_neurons_perc.total"].mean())
    
    plt.errorbar(mean["frozen_neurons_perc.total"].mean(), mean["test.accuracy.top1"].iloc[[-1]],
                 label=f"eps={eps}", alpha=0.7, fmt="o", linewidth=1)
    
    plt.legend()
    plt.xlabel("Frozen Neurons (%)")
    plt.ylabel("Classification Accuracy (%)")
    plt.tight_layout()
    plt.show()


def plot_accuracy(runs):

    for eps in runs:
        mean = runs[eps].groupby(level=0).mean()["test.accuracy.top1"]
        std = runs[eps].groupby(level=0).std()["test.accuracy.top1"]
        min = runs[eps].groupby(level=0).min()["test.accuracy.top1"]
        max = runs[eps].groupby(level=0).max()["test.accuracy.top1"]
        
        plt.plot(np.arange(0, mean.shape[0]), mean, label=f"eps={eps}", alpha=0.7, linewidth=1)
        plt.fill_between(x=np.arange(0, mean.shape[0]), y1=min, y2=max, alpha=0.3)
    
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Classification Accuracy (%)")
    plt.tight_layout()
    plt.show()


def plot_frozen(runs):

    for eps in runs:
        mean = runs[eps].groupby(level=0).mean()["frozen_neurons_perc.total"]
        std = runs[eps].groupby(level=0).std()["frozen_neurons_perc.total"]
        min = runs[eps].groupby(level=0).min()["frozen_neurons_perc.total"]
        max = runs[eps].groupby(level=0).max()["frozen_neurons_perc.total"]
        
        plt.plot(np.arange(0, mean.shape[0]), mean, label=f"eps={eps}", alpha=0.7, linewidth=1)
        plt.fill_between(x=np.arange(0, mean.shape[0]), y1=min, y2=max, alpha=0.3)
    
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Frozen Neurons (%)")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plt.style.context("seaborn-pastel")
    
    model = resnet18()
    bs = 256
    input = torch.randn(bs, 3, 32, 32)
    total_ops, total_params, ret_dict = profile(model, inputs=(input,), ret_layer_info=True)

    layer_ops = {}

    for n, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm)):
            layer_ops[n] = m._buffers["total_ops"].item() * 2
    
    api = wandb.Api(timeout=60)
    
    id = "hm02748q"
    run = api.run(f"andreabrg/zero-grad/{id}")
    
    config = json.loads(run.json_config)
    df = run.history()
    print(df["frozen_neurons_perc.total"].mean())
    
    plot_scatter_flops(df, layer_ops)
    plot_scatter_frozen(df)
    plot_accuracy(df)
    plot_frozen(df)
