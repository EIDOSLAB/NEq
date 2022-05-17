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
    df = pd.read_csv("../csv/resnet32-cifar10/ablation/adam-vs-sgd-epochs.csv")
    optim = df["optim"].tolist()
    optim = sorted(set(optim))
    
    ids = defaultdict(list)
    
    for opt in optim:
        ids[opt] = df.loc[df['optim'] == opt]["ID"].tolist()
    
    runs = defaultdict()
    
    for opt in ids:
        dfs = []
        for id in tqdm(ids[opt]):
            run = api.run(f"andreabrg/zero-grad-cifar-ablation/{id}")
            config = json.loads(run.json_config)
            df = run.history()
            dfs.append(df[[c for c in df.columns if "frozen_neurons_perc" in c] + ["test.accuracy.top1"]])
        
        runs[opt] = pd.concat(dfs)
        runs[opt]["test.accuracy.top1"] *= 100
        
    for opt in runs:
        mean = runs[opt].groupby(level=0).mean()
        std = runs[opt].groupby(level=0).std()
        min = runs[opt].groupby(level=0).min()
        max = runs[opt].groupby(level=0).max()
        
        plot_flops = []
        plot_neurons = []
        plot_flops_min = []
        plot_neurons_min = []
        plot_flops_max = []
        plot_neurons_max = []
        
        for i in tqdm(range(len(df.index))):
            
            remaining_ops = 0
            remaining_ops_min = 0
            remaining_ops_max = 0
            remaining_neur = 0
            remaining_neur_min = 0
            remaining_neur_max = 0
            
            for layer in layer_ops:
                if f"frozen_neurons_perc.layer.{layer}" in df:
                    ops = layer_ops[layer]
                    frozen_ops = ops * (mean[f"frozen_neurons_perc.layer.{layer}"].iloc[[i]].values[0] / 100)
                    remaining_ops += ops - frozen_ops
                    
                    frozen_ops = ops * (min[f"frozen_neurons_perc.layer.{layer}"].iloc[[i]].values[0] / 100)
                    remaining_ops_min += ops - frozen_ops
                    
                    frozen_ops = ops * (max[f"frozen_neurons_perc.layer.{layer}"].iloc[[i]].values[0] / 100)
                    remaining_ops_max += ops - frozen_ops
                    
                    neur = neurons[layer]
                    frozen_neur = neur * (mean[f"frozen_neurons_perc.layer.{layer}"].iloc[[i]].values[0] / 100)
                    remaining_neur += neur - frozen_neur
                    
                    neur = neurons[layer]
                    frozen_neur = neur * (min[f"frozen_neurons_perc.layer.{layer}"].iloc[[i]].values[0] / 100)
                    remaining_neur_min += neur - frozen_neur
                    
                    neur = neurons[layer]
                    frozen_neur = neur * (max[f"frozen_neurons_perc.layer.{layer}"].iloc[[i]].values[0] / 100)
                    remaining_neur_max += neur - frozen_neur
            
            plot_flops.append(remaining_ops)
            plot_flops_min.append(remaining_ops_min)
            plot_flops_max.append(remaining_ops_max)
            plot_neurons.append(int(remaining_neur))
            plot_neurons_max.append(int(remaining_neur_max))
            plot_neurons_min.append(int(remaining_neur_min))

        fig, axs = plt.subplots(1, 3, figsize=(12, 3))
        ax2_0 = axs[0].twinx()
        ax2_0.plot(plot_flops, alpha=0.7, color='#ff7f0e')
        ax2_0.fill_between(x=np.arange(0, len(plot_flops)), y1=plot_flops_min, y2=plot_flops_max, alpha=0.2,
                           color='#ff7f0e')
        axs[0].plot(df["lr"], alpha=0.7, color='#1f77b4')
        axs[0].set_ylabel('Learning Rate', color='#1f77b4', fontsize=15)
        ax2_0.set_ylabel('Bprop. FLOPs per iter.', color='#ff7f0e', fontsize=15)
        axs[0].set_xlabel("Epochs", fontsize=15)
        axs[0].set_yscale("log")
        axs[0].minorticks_off()

        ax2_1 = axs[1].twinx()
        ax2_1.plot(plot_neurons, alpha=0.7, color='#2ca02c')
        ax2_1.fill_between(x=np.arange(0, len(plot_flops)), y1=plot_neurons_min, y2=plot_neurons_max, alpha=0.2,
                           color='#2ca02c')
        axs[1].plot(df["lr"], alpha=0.7, color='#1f77b4')
        ax2_1.set_ylabel('Updated Neurons', color='#2ca02c', fontsize=15)
        axs[1].set_xlabel("Epochs", fontsize=15)
        axs[1].set_yscale("log")
        axs[1].minorticks_off()
        axs[1].set_yticks([])
        ax2_1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        ax2_2 = axs[2].twinx()
        ax2_2.plot(mean["test.accuracy.top1"], alpha=0.7, color='#d62728')
        ax2_2.fill_between(x=np.arange(0, len(plot_flops)), y1=min["test.accuracy.top1"], y2=max["test.accuracy.top1"],
                           alpha=0.2, color='#d62728')
        axs[2].plot(df["lr"], alpha=0.7, color='#1f77b4')
        ax2_2.set_ylabel('Accuracy', color='#d62728', fontsize=15)
        axs[2].set_xlabel("Epochs", fontsize=15)
        axs[2].set_yscale("log")
        axs[2].minorticks_off()
        axs[2].set_yticks([])
        ax2_2.set_yticks([50, 60, 70, 80, 90])

        axs[0].tick_params(axis='both', which='major', labelsize=13)
        axs[1].tick_params(axis='both', which='major', labelsize=13)
        axs[2].tick_params(axis='both', which='major', labelsize=13)

        ax2_0.tick_params(axis='both', which='major', labelsize=13)
        ax2_1.tick_params(axis='both', which='major', labelsize=13)
        ax2_2.tick_params(axis='both', which='major', labelsize=13)

        fig.tight_layout()
        plt.savefig(f"adam-vs-sgd-epochs-{opt}.png", dpi=300)
        plt.savefig(f"adam-vs-sgd-epochs-{opt}.pdf", dpi=300)
        plt.clf()
