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


def plot_scatter_flops(runs, layer_ops, neurons):
    plt.figure(figsize=(9, 3), dpi=600)

    total_ops = sum(layer_ops.values())
    total_neurons = sum(neurons.values())
    
    for eps in runs:
        mean = runs[eps].groupby(level=0).mean()
        std = runs[eps].groupby(level=0).std()
        min = runs[eps].groupby(level=0).min()
        max = runs[eps].groupby(level=0).max()
        
        remaining_ops = 0
        remaining_neurons = 0

        for layer in layer_ops:
            if f"frozen_neurons_perc.layer.{layer}" in mean:
                ops = layer_ops[layer]
                frozen_ops = ops * (mean[f"frozen_neurons_perc.layer.{layer}"] / 100)
                remaining_ops += ops - frozen_ops
        
                remaining_neurons += (neurons[layer] * ((100 - mean[f"frozen_neurons_perc.layer.{layer}"]) / 100))

        backprop_flops = round(remaining_ops.mean(), 2)
        backprop_flops_std = round(remaining_ops.std() / 100, 2)
        
        plt.errorbar(backprop_flops, mean["test.accuracy.top1"].iloc[[-1]],
                     label=f"$\epsilon = {eps}$", yerr=std["test.accuracy.top1"].iloc[[-1]], alpha=0.7, fmt="o", linewidth=1)
        
        print(f"{eps} "
              f"& {clever_format(backprop_flops)} \pm {clever_format(backprop_flops_std)} "
              f"& {round(mean[f'test.accuracy.top1'].iloc[[-1]].values[0], 2)} \pm {round(std[f'test.accuracy.top1'].iloc[[-1]].values[0], 2)} \\")
    
    plt.legend(ncol=3)
    plt.xlabel("FLOPs")
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
    neurons = {}
    
    for n, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm)):
            layer_ops[n] = m._buffers["total_ops"].item() * 2
            neurons[n] = m.weight.shape[0]
    
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
    
    plot_scatter_flops(runs, layer_ops, neurons)


if __name__ == '__main__':
    main()
    # plot_scatter_frozen(runs)
    # plot_accuracy(runs)
    # plot_frozen(runs)
