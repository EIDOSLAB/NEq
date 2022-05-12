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
        
        plt.errorbar(100 - (remaining_ops.mean() / total_ops * 100), mean["test.accuracy.top1"].iloc[[-1]],
                     label=f"eps={eps}", yerr=std["test.accuracy.top1"].iloc[[-1]], alpha=0.7, fmt="o", linewidth=1)
    
    plt.legend(ncol=3)
    plt.xlabel("Saved FLOPS (%)")
    plt.ylabel("Classification Accuracy (%)")
    plt.tight_layout()
    plt.savefig("eps.png", dpi=300)
    plt.savefig("eps.pdf", dpi=300)
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
    df = pd.read_csv("../csv/resnet32-cifar10/ablation/eps.csv")
    epsess = df["eps"].tolist()
    epsess = sorted(set(epsess))
    
    ids = defaultdict(list)
    
    for eps in epsess:
        ids[eps] = df.loc[df['eps'] == eps]["ID"].tolist()
    
    runs = defaultdict()
    
    for eps in tqdm(epsess):
        dfs = []
        for id in tqdm(ids[eps]):
            run = api.run(f"andreabrg/zero-grad-cifar-ablation/{id}")
            config = json.loads(run.json_config)
            df = run.history()
            dfs.append(df[[c for c in df.columns if "frozen_neurons_perc" in c] + ["test.accuracy.top1"]])
        
        runs[eps] = pd.concat(dfs)
        runs[eps]["test.accuracy.top1"] *= 100
    
    plot_scatter_flops(runs, layer_ops)


if __name__ == '__main__':
    main()
    # plot_scatter_frozen(runs)
    # plot_accuracy(runs)
    # plot_frozen(runs)
