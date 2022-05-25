import json

import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from torch import nn
from tqdm import tqdm

from classification.models import resnet32
from plots.thop import profile

if __name__ == '__main__':
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
    df = df[[c for c in df.columns if "frozen_neurons_perc" in c]]
    
    for layer in tqdm(layer_ops):
        data = df[f"frozen_neurons_perc.layer.{layer}"]

        ops = layer_ops[layer]
        y = []
        
        for i in range(len(data.index)):
            frozen_ops = ops * (data.iloc[[i]].values[0] / 100)
            y.append(ops - frozen_ops)

        plt.plot(np.arange(0, 250), y, alpha=0.7)
        plt.xlabel("Epochs", fontsize=20)
        plt.ylabel("Bprop. FLOPs per iter.", fontsize=15)

        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.tick_params(axis='both', which='major', labelsize=15)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        plt.tight_layout()
        plt.savefig(f"res32_full/{layer}.png", dpi=300)
        plt.savefig(f"res32_full/{layer}.pdf", dpi=300)
        plt.clf()
