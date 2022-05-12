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


def build_table(stoc_runs, eps_runs, layer_ops):
    total_ops = sum(layer_ops.values())
    
    for topk in stoc_runs:
        mean = stoc_runs[topk].groupby(level=0).mean()
        std = stoc_runs[topk].groupby(level=0).std()
        min = stoc_runs[topk].groupby(level=0).min()
        max = stoc_runs[topk].groupby(level=0).max()
        
        remaining_ops = 0
        
        for layer in layer_ops:
            if f"frozen_neurons_perc.layer.{layer}" in mean:
                ops = layer_ops[layer]
                frozen_ops = ops * (mean[f"frozen_neurons_perc.layer.{layer}"] / 100)
                remaining_ops += ops - frozen_ops
        
        approach = "Stochastic"
        frozen_neurons = round(mean[f"frozen_neurons_perc.total"].mean(), 2)
        saved_flops = round(100 - (remaining_ops.mean() / total_ops * 100), 2)
        accuracy = round(mean[f"test.accuracy.top1"].iloc[[-1]].values[0], 2)
        
        print(f"{approach} &"
              f" {frozen_neurons:.2f} $\pm$ {round(std['frozen_neurons_perc.total'].mean(), 2)} &"
              f" {saved_flops:.2f}  $\pm$ {round(std['frozen_neurons_perc.total'].mean(), 2)} &"
              f" {accuracy:.2f} $\pm$ {round(std['test.accuracy.top1'].iloc[[-1]].values[0], 2)}")

    for eps in eps_runs:
        mean = eps_runs[eps].groupby(level=0).mean()
        std = eps_runs[eps].groupby(level=0).std()
        min = eps_runs[eps].groupby(level=0).min()
        max = eps_runs[eps].groupby(level=0).max()
    
        remaining_ops = 0
    
        for layer in layer_ops:
            if f"frozen_neurons_perc.layer.{layer}" in mean:
                ops = layer_ops[layer]
                frozen_ops = ops * (mean[f"frozen_neurons_perc.layer.{layer}"] / 100)
                remaining_ops += ops - frozen_ops
    
        approach = f"Ours (\eps={eps})"
        frozen_neurons = round(mean[f"frozen_neurons_perc.total"].mean(), 2)
        saved_flops = round(100 - (remaining_ops.mean() / total_ops * 100), 2)
        accuracy = round(mean[f"test.accuracy.top1"].iloc[[-1]].values[0], 2)
    
        print(f"{approach} &"
              f" {frozen_neurons:.2f} $\pm$ {round(std['frozen_neurons_perc.total'].mean(), 2)} &"
              f" {saved_flops:.2f}  $\pm$ {round(std['frozen_neurons_perc.total'].mean(), 2)} &"
              f" {accuracy:.2f} $\pm$ {round(std['test.accuracy.top1'].iloc[[-1]].values[0], 2)}")


def plot_scatter_flops(runs, layer_ops):
    # plt.figure(figsize=(9, 3), dpi=600)
    
    total_ops = sum(layer_ops.values())
    
    for topk in runs:
        mean = runs[topk].groupby(level=0).mean()
        std = runs[topk].groupby(level=0).std()
        min = runs[topk].groupby(level=0).min()
        max = runs[topk].groupby(level=0).max()
        
        remaining_ops = 0
        
        for layer in layer_ops:
            if f"frozen_neurons_perc.layer.{layer}" in mean:
                ops = layer_ops[layer]
                frozen_ops = ops * (mean[f"frozen_neurons_perc.layer.{layer}"] / 100)
                remaining_ops += ops - frozen_ops
        
        label = f"topk={topk}" if topk != 0 else "Our"
        
        plt.errorbar(100 - (remaining_ops.mean() / total_ops * 100), mean["test.accuracy.top1"].iloc[[-1]],
                     label=label, alpha=0.7, fmt="o", linewidth=1)
    
    plt.legend()
    plt.xlabel("Saved FLOPS (%)")
    plt.ylabel("Classification Accuracy (%)")
    plt.tight_layout()
    plt.show()


def plot_scatter_frozen(runs):
    # plt.figure(figsize=(9, 3), dpi=600)
    
    for topk in runs:
        mean = runs[topk].groupby(level=0).mean()
        std = runs[topk].groupby(level=0).std()
        min = runs[topk].groupby(level=0).min()
        max = runs[topk].groupby(level=0).max()
        
        print(topk, mean["frozen_neurons_perc.total"].mean())
        
        label = f"topk={topk}" if topk != 0 else "Our"
        
        plt.errorbar(mean["frozen_neurons_perc.total"].mean(), mean["test.accuracy.top1"].iloc[[-1]],
                     label=label, alpha=0.7, fmt="o", linewidth=1)
    
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
        
        label = f"topk={topk}" if topk != 0 else "Our"
        
        plt.plot(np.arange(0, mean.shape[0]), mean, label=label, alpha=0.7, linewidth=1)
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
        
        label = f"topk={topk}" if topk != 0 else "Our"
        
        plt.plot(np.arange(0, mean.shape[0]), mean, label=label, alpha=0.7, linewidth=1)
        plt.fill_between(x=np.arange(0, mean.shape[0]), y1=min, y2=max, alpha=0.3)
    
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Frozen Neurons (%)")
    plt.tight_layout()
    plt.show()


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
    df_ids = pd.read_csv("csv/resnet32-cifar10/stoc-vs-eps.csv")
    topks = [v for v in df_ids["topk"].tolist() if v != 0]
    topks = sorted(set(topks), reverse=True)

    epsess = [v for v in df_ids["eps"].tolist() if v != "none"]
    epsess = sorted(set(epsess), reverse=True)
    
    stoc_ids = defaultdict(list)
    eps_ids = defaultdict(list)
    
    for topk in topks:
        stoc_ids[topk] = df_ids.loc[df_ids['topk'] == topk]["ID"].tolist()
    for eps in epsess:
        eps_ids[eps] = df_ids.loc[df_ids['eps'] == eps]["ID"].tolist()
    
    stoc_runs = defaultdict()
    eps_runs = defaultdict()
    
    for topk in tqdm(topks):
        dfs = []
        for id in tqdm(stoc_ids[topk]):
            run = api.run(f"andreabrg/zero-grad-cifar-ablation/{id}")
            config = json.loads(run.json_config)
            df = run.history()
            dfs.append(df[[c for c in df.columns if "frozen_neurons_perc" in c] + ["test.accuracy.top1"]])
        
        stoc_runs[topk] = pd.concat(dfs)
        stoc_runs[topk]["test.accuracy.top1"] *= 100

    for eps in tqdm(epsess):
        dfs = []
        for id in tqdm(eps_ids[eps]):
            run = api.run(f"andreabrg/zero-grad-cifar-ablation/{id}")
            config = json.loads(run.json_config)
            df = run.history()
            dfs.append(df[[c for c in df.columns if "frozen_neurons_perc" in c] + ["test.accuracy.top1"]])
    
        eps_runs[eps] = pd.concat(dfs)
        eps_runs[eps]["test.accuracy.top1"] *= 100
    
    build_table(stoc_runs, eps_runs, layer_ops)
    # plot_scatter_flops(stoc_runs, layer_ops)
    # plot_scatter_frozen(stoc_runs)
    # plot_accuracy(stoc_runs)
    # plot_frozen(stoc_runs)
