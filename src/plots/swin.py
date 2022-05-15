import json
from collections import defaultdict

import pandas as pd
import torch
import wandb
from matplotlib import pyplot as plt
from torch import nn
from torchvision.models import resnet18
from tqdm import tqdm

from plots.swin_transformer import SwinTransformer
from plots.thop import profile, clever_format


def build_table(stoc_runs, eps_runs, layer_ops, neurons):
    total_ops = sum(layer_ops.values())
    total_neurons = sum(neurons.values())
    
    print(clever_format(total_ops))
    print(total_neurons)
    
    for topk in stoc_runs:
        mean = stoc_runs[topk].groupby(level=0).mean()
        std = stoc_runs[topk].groupby(level=0).std()
        min = stoc_runs[topk].groupby(level=0).min()
        max = stoc_runs[topk].groupby(level=0).max()
        
        remaining_ops = 0
        remaining_neurons = 0
        
        for layer in layer_ops:
            if f"frozen_neurons_perc.layer.{layer}" in mean:
                ops = layer_ops[layer]
                frozen_ops = ops * (mean[f"frozen_neurons_perc.layer.{layer}"] / 100)
                remaining_ops += ops - frozen_ops
                
                remaining_neurons += (neurons[layer] * ((100 - mean[f"frozen_neurons_perc.layer.{layer}"]) / 100))
        
        approach = f"Stochastic (topk={topk})"
        trained_neurons = int(remaining_neurons.mean())
        backprop_flops = round(remaining_ops.mean(), 2)
        backprop_flops_delta = round(100 - (remaining_ops.mean() / total_ops * 100), 2)
        accuracy = round(mean[f"test.accuracy.top1"].iloc[[-1]].values[0], 2)
        
        print(f"{approach} &"
              f" {trained_neurons} $\pm$ {round(std['frozen_neurons_perc.total'].mean(), 2)} &"
              f" {clever_format(backprop_flops)}  $\pm$ {round(std['frozen_neurons_perc.total'].mean(), 2)} (-{backprop_flops_delta}\%) &"
              f" {accuracy:.2f} $\pm$ {round(std['test.accuracy.top1'].iloc[[-1]].values[0], 2)}")
    
    for eps in eps_runs:
        mean = eps_runs[eps].groupby(level=0).mean()
        std = eps_runs[eps].groupby(level=0).std()
        min = eps_runs[eps].groupby(level=0).min()
        max = eps_runs[eps].groupby(level=0).max()
        
        remaining_ops = 0
        remaining_neurons = 0
        
        for layer in layer_ops:
            if f"frozen_neurons_perc.layer.{layer}" in mean:
                ops = layer_ops[layer]
                frozen_ops = ops * (mean[f"frozen_neurons_perc.layer.{layer}"] / 100)
                remaining_ops += ops - frozen_ops
                
                remaining_neurons += (neurons[layer] * ((100 - mean[f"frozen_neurons_perc.layer.{layer}"]) / 100))
        
        approach = f"Ours (\eps={eps})"
        trained_neurons = int(remaining_neurons.mean())
        backprop_flops = round(remaining_ops.mean(), 2)
        backprop_flops_delta = round(100 - (remaining_ops.mean() / total_ops * 100), 2)
        accuracy = round(mean[f"test.accuracy.top1"].iloc[[-1]].values[0], 2)
        
        print(f"{approach} &"
              f" {trained_neurons} &"
              f" {clever_format(backprop_flops)} (-{backprop_flops_delta}\%) &"
              f" {accuracy:.2f}")


if __name__ == '__main__':
    plt.style.context("seaborn-pastel")
    model = SwinTransformer(img_size=224,
                            patch_size=4,
                            in_chans=3,
                            num_classes=1000,
                            embed_dim=128,
                            depths=[2, 2, 18, 2],
                            num_heads=[4, 8, 16, 32],
                            window_size=7,
                            mlp_ratio=4.0,
                            qkv_bias=True,
                            qk_scale=None,
                            drop_rate=0.0,
                            drop_path_rate=0.2,
                            ape=False,
                            patch_norm=True,
                            use_checkpoint=False, )
    bs = 64
    input = torch.randn(bs, 3, 224, 224)
    total_ops, total_params, ret_dict = profile(model, inputs=(input,), ret_layer_info=True)
    
    layer_ops = {}
    neurons = {}
    
    for n, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm)):
            if n != "head":
                layer_ops[n] = m._buffers["total_ops"].item() * 2
                neurons[n] = m.weight.shape[0]
    
    api = wandb.Api(timeout=60)
    df_ids = pd.read_csv("csv/swin-imagenet/stoc-vs-eps.csv")
    topks = [v for v in df_ids["topk"].tolist() if v != "-"]
    topks = sorted(set(topks), reverse=True)
    
    epsess = [v for v in df_ids["eps"].tolist() if v != "-"]
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
            run = api.run(f"andreabrg/zero-grad/{id}")
            config = json.loads(run.json_config)
            df = run.history()
            dfs.append(df[[c for c in df.columns if "frozen_neurons_perc" in c] + ["test.accuracy.top1"]])
        
        stoc_runs[topk] = pd.concat(dfs)
    
    for eps in tqdm(epsess):
        dfs = []
        for id in tqdm(eps_ids[eps]):
            run = api.run(f"andreabrg/zero-grad/{id}")
            config = json.loads(run.json_config)
            df = run.history()
            dfs.append(df[[c for c in df.columns if "frozen_neurons_perc" in c] + ["test.accuracy.top1"]])
        
        eps_runs[eps] = pd.concat(dfs)
    
    build_table(stoc_runs, eps_runs, layer_ops, neurons)
