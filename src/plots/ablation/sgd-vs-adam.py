import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import wandb
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, to_rgb
from torch import nn
from tqdm import tqdm

from classification.models import resnet32
from plots.thop import profile, clever_format
import numpy.matlib

if __name__ == '__main__':
    plt.style.context("seaborn-pastel")
    
    # model = resnet32()
    # bs = 100
    # input = torch.randn(bs, 3, 32, 32)
    # total_ops, total_params, ret_dict = profile(model, inputs=(input,), ret_layer_info=True)
    #
    # layer_ops = {}
    #
    # for n, m in model.named_modules():
    #     if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm)):
    #         layer_ops[n] = m._buffers["total_ops"].item() * 2
    
    api = wandb.Api(timeout=60)
    runs = defaultdict()
    
    run = api.run(f"andreabrg/zero-grad-cifar-ablation/js7letd4")
    runs["adam"] = run.history()
    runs["adam"]["test.accuracy.top1"] *= 100
    
    run = api.run(f"andreabrg/zero-grad-cifar-ablation/xmgdjb9u")
    runs["sgd"] = run.history()
    runs["sgd"]["test.accuracy.top1"] *= 100

    import seaborn

    data = runs["adam"]["deltas.dod.stage_3.4.conv_b"]
    fig, axs = plt.subplots()

    df = pd.DataFrame(columns=["value", "bin", "epoch"])

    for i in tqdm(range(len(data.index))):

        df2 = pd.DataFrame()
    
        df2["value"] = data.iloc[i]["values"]
        df2["bin"] = data.iloc[i]["bins"][:-1]
        df2["epoch"] = i
        
        for j in range(len(df2.index)):
            df.loc[-1] = [df2["value"].iloc[j], df2["bin"].iloc[j], df2["epoch"].iloc[j]]
            df.index = df.index + 1  # shifting index
            df.sort_index(inplace=True)

        # seaborn.histplot(
        #     df, x="epoch", y="bin", weights=df["value"], bins=len(df["bin"]), discrete=(True, False),
        #     log_scale=(False, True),
        # )
        #
        # plt.show()
        # print()
        # plt.clf()
    alpha_arr = [(v - df["value"].min()) / (df["value"].max() - df["value"].min()) for v in df["value"]]
    r, g, b = to_rgb("#1f77b4")
    # r, g, b, _ = to_rgba(color)
    color = [(r, g, b, alpha) for alpha in alpha_arr]
    # Same data but on linear color scale
    plt.scatter(df["epoch"], df["bin"], c=color, marker="_")
    plt.show()
    plt.clf()
