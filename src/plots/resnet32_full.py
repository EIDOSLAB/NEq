import json

import numpy as np
import pandas as pd
import seaborn
import torch
import wandb
from matplotlib import pyplot as plt, rc
from torch import nn
from tqdm import tqdm

from classification.models import resnet32
from plots.thop import profile

if __name__ == '__main__':
    rc('font', family='Times New Roman')
    rc('text', usetex=True)
    plt.style.context("seaborn-pastel")
    
    model = resnet32()
    bs = 1
    input = torch.randn(bs, 3, 32, 32)
    total_ops, total_params, ret_dict = profile(model, inputs=(input,), ret_layer_info=True)
    
    layer_ops = {}
    neurons = {}
    dfs = []
    
    for n, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm)):
            layer_ops[n] = m._buffers["total_ops"].item() * 2
            neurons[n] = m.weight.shape[0]
    
    api = wandb.Api(timeout=60)
    
    run = api.run(f"andreabrg/zero-grad-cifar-ablation/a75es22z")
    config = json.loads(run.json_config)
    df = run.history()
    
    for indx, layer in tqdm(enumerate(layer_ops)):

        ops = layer_ops[layer]
        y = []
        
        for i in range(len(df[f"frozen_neurons_perc.layer.{layer}"].index)):
            frozen_ops = ops * (df[f"frozen_neurons_perc.layer.{layer}"].iloc[[i]].values[0] / 100)
            y.append(ops - frozen_ops)

        d = df[f"deltas.velocity.{layer}"]
        df_distr = pd.DataFrame(columns=["value", "bin", "epoch"])

        for i in tqdm(range(len(d.index))):
    
            df2 = pd.DataFrame()
    
            df2["value"] = d.iloc[i]["values"]
            df2["bin"] = d.iloc[i]["bins"][:-1]
            df2["epoch"] = i
    
            for j in range(len(df2.index)):
                df_distr.loc[-1] = [df2["value"].iloc[j], df2["bin"].iloc[j], df2["epoch"].iloc[j]]
                df_distr.index = df_distr.index + 1  # shifting index
                df_distr.sort_index(inplace=True)

        df_distr = df_distr.loc[df_distr.index.repeat(df_distr.value)].reset_index(drop=True)

        fig, axs = plt.subplots(1, 2, figsize=(17, 5.5))
        seaborn.histplot(df_distr, x="epoch", y="bin",
                         bins=(250, int(df_distr["value"].max().max()) * 5), ax=axs[0])
        mean = df_distr.groupby("epoch").mean()
        axs[0].plot(np.arange(0, df_distr["epoch"].max().max() + 1), mean["bin"], alpha=0.7, color="#d62728")
        axs[0].set_ylim(top=1.3, bottom=-1.3)
        axs[0].set_yticks([1, 0.5, 0, -0.5, -1])
        axs[0].tick_params(axis='both', which='major', labelsize=15)
        axs[0].set_ylabel(r"$v_{\Delta\phi}$", fontsize=20)
        axs[0].set_xlabel("Epochs", fontsize=20)

        axs[1].plot(np.arange(0, 250), y, alpha=0.7)
        axs[1].set_xlabel("Epochs", fontsize=20)
        axs[1].set_ylabel("Bprop. FLOPs per iter.", fontsize=20)
        axs[1].tick_params(axis='both', which='major', labelsize=15)
        axs[1].tick_params(axis='both', which='major', labelsize=15)
        axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        plt.tight_layout()
        plt.savefig(f"res32_full/{indx}_{layer}.pdf", dpi=300)
        plt.clf()
