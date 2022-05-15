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


def build_table(runs, layer_ops, neurons):
    total_ops = sum(layer_ops.values())
    total_neurons = sum(neurons.values())
    
    for val_size in runs:
        mean = runs[val_size].groupby(level=0).mean()
        std = runs[val_size].groupby(level=0).std()
        min = runs[val_size].groupby(level=0).min()
        max = runs[val_size].groupby(level=0).max()
        
        remaining_ops = 0
        remaining_neurons = 0

        for layer in layer_ops:
            if f"frozen_neurons_perc.layer.{layer}" in mean:
                ops = layer_ops[layer]
                frozen_ops = ops * (mean[f"frozen_neurons_perc.layer.{layer}"] / 100)
                remaining_ops += ops - frozen_ops
        
                remaining_neurons += (neurons[layer] * ((100 - mean[f"frozen_neurons_perc.layer.{layer}"]) / 100))

        images = int(val_size * 50000)
        trained_neurons = int(remaining_neurons.mean())
        trained_neurons_std = int(total_neurons * std["frozen_neurons_perc.total"].mean() / 100)
        backprop_flops = round(remaining_ops.mean(), 2)
        backprop_flops_std = round(remaining_ops.std() / 100, 2)
        accuracy = round(mean[f"test.accuracy.top1"].iloc[[-1]].values[0], 2)
        
        print(f"{images} & {trained_neurons} $\pm$ {trained_neurons_std}"
              f" & {clever_format(backprop_flops)} $\pm$ {clever_format(backprop_flops_std)}"
              f" & {accuracy} $\pm$ {round(std[f'test.accuracy.top1'].iloc[[-1]].values[0], 2)}")


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
    
    build_table(runs, layer_ops, neurons)


if __name__ == '__main__':
    main()
    # plot_scatter_frozen(runs)
    # plot_accuracy(runs)
    # plot_frozen(runs)
