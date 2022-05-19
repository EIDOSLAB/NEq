import json
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn
import wandb
from matplotlib import pyplot as plt, rc
from tqdm import tqdm

if __name__ == '__main__':
    rc('font', family='Times New Roman')
    rc('text', usetex=True)
    
    plt.style.context("seaborn-pastel")
    
    api = wandb.Api(timeout=60)
    
    ids = [("pk87v6i4", 0), ("a75es22z", 0.5), ("pdprqd6x", 0.9)]
    
    target_layer = "stage_3.4.bn_b"
    
    dfs_dict = {}
    
    for id, mu in ids:
        run = api.run(f"andreabrg/zero-grad-cifar-ablation/{id}")
        data = run.history()
        
        plts = ["velocity"] if mu != 0 else ["phi", "d_phi"]
        
        dfs = {}
        
        for plot in plts:
            df = pd.DataFrame(columns=["value", "bin", "epoch"])
            
            d = data[f"deltas.{plot}.{target_layer}"]
            
            for i in tqdm(range(len(d.index))):
                
                df2 = pd.DataFrame()
                
                df2["value"] = d.iloc[i]["values"]
                df2["bin"] = d.iloc[i]["bins"][:-1]
                if plot == "phi":
                    df2["bin"] = 1 - df2["bin"]
                df2["epoch"] = i
                
                for j in range(len(df2.index)):
                    df.loc[-1] = [df2["value"].iloc[j], df2["bin"].iloc[j], df2["epoch"].iloc[j]]
                    df.index = df.index + 1  # shifting index
                    df.sort_index(inplace=True)
            
            dfs[plot] = df.loc[df.index.repeat(df.value)].reset_index(drop=True)
            
        dfs_dict[mu] = dfs

    for mu in dfs_dict:
        for plot in dfs_dict[mu]:
            fig, axs = plt.subplots(1, 1, figsize=(17, 5.5))
            seaborn.histplot(dfs_dict[mu][plot], x="epoch", y="bin",
                             bins=(250, int(dfs_dict[mu][plot]["value"].max().max()) * 10), ax=axs)
        
            mean = dfs_dict[mu][plot].groupby("epoch").mean()
            axs.plot(np.arange(0, dfs_dict[mu][plot]["epoch"].max().max() + 1), mean["bin"], alpha=0.7, color="#d62728")
        
            axs.set_ylim(top=1.3, bottom=-1.3)
            axs.set_yticks([1, 0.5, 0, -0.5, -1])

            axs.tick_params(axis='both', which='major', labelsize=40)
            
            if plot == "phi":
                y_label = r"$\phi$"
            elif plot == "d_phi":
                y_label = r"$\Delta\phi$"
            else:
                y_label = r"$v_{\Delta\phi}$"

            axs.set_ylabel(y_label, fontsize=50)
            axs.set_xlabel("Epochs", fontsize=50)
        
            plt.tight_layout()
            plt.savefig(f"distr_{mu}-{plot}.png", dpi=300)
            plt.savefig(f"distr_{mu}-{plot}.pdf", dpi=300)
            plt.clf()
