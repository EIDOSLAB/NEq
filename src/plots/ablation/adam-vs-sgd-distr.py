import json
from collections import defaultdict

import pandas as pd
import seaborn
import wandb
from matplotlib import pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    plt.style.context("seaborn-pastel")
    
    api = wandb.Api(timeout=60)
    df = pd.read_csv("../csv/resnet32-cifar10/ablation/adam-vs-sgd-distr.csv")
    optim = df["optim"].tolist()
    optim = sorted(set(optim))
    
    ids = []
    
    target_layer = "conv_1_3x3"

    for index, row in df.iterrows():
        run = api.run(f"andreabrg/zero-grad-cifar-ablation/{row['ID']}")
        data = run.history()
        
        dfs = []
        
        for plot in ["phi", "d_phi", "velocity"]:

            d = data[f"deltas.{plot}.{target_layer}"]
            df = pd.DataFrame(columns=["value", "bin", "epoch"])
            
            for i in tqdm(range(len(d.index))):
    
                df2 = pd.DataFrame()
    
                df2["value"] = d.iloc[i]["values"]
                df2["bin"] = d.iloc[i]["bins"][:-1]
                df2["epoch"] = i
    
                for j in range(len(df2.index)):
                    df.loc[-1] = [df2["value"].iloc[j], df2["bin"].iloc[j], df2["epoch"].iloc[j]]
                    df.index = df.index + 1  # shifting index
                    df.sort_index(inplace=True)
                    
            dfs.append(df.loc[df.index.repeat(df.value)].reset_index(drop=True))

        fig, axs = plt.subplots(3, 1, figsize=(10, 9))
        
        for ax_id, df in enumerate(dfs):
            seaborn.histplot(df, x="epoch", y="bin", bins=(250, int(df["value"].max().max())), ax=axs[ax_id])
            
        plt.tick_params(axis='both', which='major', labelsize=13)
        plt.xlabel("Epochs", fontsize=15)
        plt.ylabel("Value", fontsize=15)
        plt.tight_layout()
        plt.show()
        plt.clf()
