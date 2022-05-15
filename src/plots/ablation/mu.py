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


def plot_frozen(runs, layer_ops):
    plt.figure(figsize=(9, 3), dpi=600)
    total_ops = sum(layer_ops.values())
    
    for mu in runs:
        mean = runs[mu].groupby(level=0).mean()
        std = runs[mu].groupby(level=0).std()
        min = runs[mu].groupby(level=0).min()
        max = runs[mu].groupby(level=0).max()
        
        y = []
        y1 = []
        y2 = []
        
        for i in range(len(mean.index)):
            
            remaining_ops = 0
            remaining_ops_min = 0
            remaining_ops_max = 0
            
            for layer in layer_ops:
                if f"frozen_neurons_perc.layer.{layer}" in mean:
                    ops = layer_ops[layer]
                    frozen_ops = ops * (mean[f"frozen_neurons_perc.layer.{layer}"].iloc[[i]].values[0] / 100)
                    remaining_ops += ops - frozen_ops
                    
                    frozen_ops = ops * (min[f"frozen_neurons_perc.layer.{layer}"].iloc[[i]].values[0] / 100)
                    remaining_ops_min += ops - frozen_ops
                    
                    frozen_ops = ops * (max[f"frozen_neurons_perc.layer.{layer}"].iloc[[i]].values[0] / 100)
                    remaining_ops_max += ops - frozen_ops
            
            y.append(remaining_ops)
            y1.append(remaining_ops_min)
            y2.append(remaining_ops_max)
        
        plt.plot(np.arange(0, mean.shape[0]), y, label=f"$\mu_{{eq}}={mu}$", alpha=0.7, linewidth=1)
        plt.fill_between(x=np.arange(0, mean.shape[0]), y1=y1, y2=y2, alpha=0.1)
    
    plt.legend(ncol=1)
    plt.xlabel("Epochs")
    plt.ylabel("FLOPs")
    plt.tight_layout()
    plt.savefig("mu-line.png", dpi=300)
    plt.savefig("mu-line.pdf", dpi=300)
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
    df = pd.read_csv("../csv/resnet32-cifar10/ablation/mu.csv")
    mus = df["velocity-mu"].tolist()
    mus = sorted(set(mus))
    
    ids = defaultdict(list)
    
    for mu in mus:
        ids[mu] = df.loc[df['velocity-mu'] == mu]["ID"].tolist()
    
    runs = defaultdict()
    
    for mu in tqdm(mus):
        dfs = []
        for id in tqdm(ids[mu]):
            run = api.run(f"andreabrg/zero-grad-cifar-ablation/{id}")
            config = json.loads(run.json_config)
            df = run.history()
            dfs.append(df[[c for c in df.columns if "frozen_neurons_perc" in c] + ["test.accuracy.top1"]])
        
        runs[mu] = pd.concat(dfs)
        runs[mu]["test.accuracy.top1"] *= 100
    
    plot_frozen(runs, layer_ops)


if __name__ == '__main__':
    main()
    # plot_scatter_frozen(runs)
    # plot_accuracy(runs)
    # plot_frozen(runs)
