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
from plots.thop import profile


def plot_scatter_flops(runs, layer_ops):
    # plt.figure(figsize=(9, 3), dpi=600)
    
    total_ops = sum(layer_ops.values())
    
    for val_size in runs:
        mean = runs[val_size].groupby(level=0).mean()
        std = runs[val_size].groupby(level=0).std()
        min = runs[val_size].groupby(level=0).min()
        max = runs[val_size].groupby(level=0).max()
        
        remaining_ops = 0
        
        for layer in layer_ops:
            if f"frozen_neurons_perc.layer.{layer}" in mean:
                ops = layer_ops[layer]
                frozen_ops = ops * (mean[f"frozen_neurons_perc.layer.{layer}"] / 100)
                remaining_ops += ops - frozen_ops
        
        plt.errorbar(100 - (remaining_ops.mean() / total_ops * 100), mean["test.accuracy.top1"].iloc[[-1]],
                     label=f"size={val_size}", yerr=std["test.accuracy.top1"].iloc[[-1]], alpha=0.7, fmt="o",
                     linewidth=1)
    
    plt.legend(ncol=4)
    plt.xlabel("Saved FLOPS (%)")
    plt.ylabel("Classification Accuracy (%)")
    plt.tight_layout()
    plt.savefig("val.png", dpi=300)
    plt.savefig("val.pdf", dpi=300)
    plt.clf()


def main():
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
    df = pd.read_csv("../csv/resnet32-cifar10/ablation/val.csv")
    vals = df["val-size"].tolist()
    vals = sorted(set(vals))
    
    ids = defaultdict(list)
    
    for val_size in vals:
        ids[val_size] = df.loc[df['val-size'] == val_size]["ID"].tolist()
    
    runs = defaultdict()
    
    for val_size in tqdm(vals):
        dfs = []
        for id in tqdm(ids[val_size]):
            run = api.run(f"andreabrg/zero-grad-cifar-ablation/{id}")
            config = json.loads(run.json_config)
            df = run.history()
            dfs.append(df[[c for c in df.columns if "frozen_neurons_perc" in c] + ["test.accuracy.top1"]])
        
        runs[val_size] = pd.concat(dfs)
        runs[val_size]["test.accuracy.top1"] *= 100
    
    plot_scatter_flops(runs, layer_ops)


if __name__ == '__main__':
    main()
    # plot_scatter_frozen(runs)
    # plot_accuracy(runs)
    # plot_frozen(runs)
