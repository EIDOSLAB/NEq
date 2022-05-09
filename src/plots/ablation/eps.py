import json
from collections import defaultdict

import numpy as np
import pandas as pd
import wandb
from matplotlib import pyplot as plt
from tqdm import tqdm


def plot_scatter(runs):
    # plt.figure(figsize=(9, 3), dpi=600)
    
    runs_acc = runs["accuracy"]
    runs_frozen = runs["frozen"]

    for eps in list(zip(runs_acc, runs_frozen)):
        mean_acc = runs_acc[eps[0]].mean(axis=0)[-1]
        std_acc = runs_acc[eps[0]].std(axis=0)[-1]
        max_acc = runs_acc[eps[0]].max(axis=0)[-1]
        min_acc = runs_acc[eps[0]].min(axis=0)[-1]
        mean_frozen = runs_frozen[eps[1]].mean(axis=0)[-1]
        std_frozen = runs_frozen[eps[1]].std(axis=0)[-1]
        max_frozen = runs_frozen[eps[1]].max(axis=0)[-1]
        min_frozen = runs_frozen[eps[1]].min(axis=0)[-1]
    
        plt.errorbar(mean_acc, mean_frozen,
                     xerr=std_acc, yerr=std_frozen,
                     label=f"eps={eps[0]}", alpha=0.7, fmt="o", linewidth=1)
    
    plt.legend()
    plt.xlabel("Classification Accuracy (%)")
    plt.ylabel("Classification Accuracy (%)")
    plt.tight_layout()
    plt.show()


def plot_accuracy(runs):
    # plt.figure(figsize=(9, 3), dpi=600)
    
    runs = runs["accuracy"]

    for eps in runs:
        mean = runs[eps].mean(axis=0)
        std = runs[eps].std(axis=0)
        max = runs[eps].max(axis=0)
        min = runs[eps].min(axis=0)
        plt.plot(np.arange(0, mean.shape[0]), mean, label=f"eps={eps}", alpha=0.7, linewidth=1)
        plt.fill_between(x=np.arange(0, mean.shape[0]), y1=min, y2=max, alpha=0.3)
    
    plt.legend()
    plt.xlabel("Classification Accuracy (%)")
    plt.ylabel("Frozen Neurons (%)")
    plt.tight_layout()
    plt.show()


def plot_frozen(runs):
    # plt.figure(figsize=(9, 3), dpi=600)
    
    runs = runs["frozen"]

    for eps in runs:
        mean = runs[eps].mean(axis=0)
        std = runs[eps].std(axis=0)
        max = runs[eps].max(axis=0)
        min = runs[eps].min(axis=0)
        plt.plot(np.arange(0, mean.shape[0]), mean, label=f"eps={eps}", alpha=0.7, linewidth=1)
        plt.fill_between(x=np.arange(0, mean.shape[0]), y1=min, y2=max, alpha=0.3)
    
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Frozen Neurons (%)")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plt.style.context("seaborn-pastel")
    
    api = wandb.Api(timeout=60)
    ids = pd.read_csv("csv/eps.csv")["ID"].tolist()
    epsess = pd.read_csv("csv/eps.csv")["eps"].tolist()
    epsess = set(epsess)
    
    runs = {
        "accuracy": {eps: [] for eps in epsess},
        "frozen":   {eps: [] for eps in epsess}
    }
    
    for id in tqdm(ids):
        run = api.run(f"andreabrg/zero-grad-cifar-ablation/{id}")
        config = json.loads(run.json_config)
        df = run.history()
        df = df[["test.accuracy.top1", "frozen_neurons_perc.total"]]
        df["seed"] = config["seed"]["value"]
        df["eps"] = config["eps"]["value"]
        
        runs["accuracy"][config["eps"]["value"]].append(df["test.accuracy.top1"].values*100)
        runs["frozen"][config["eps"]["value"]].append(df["frozen_neurons_perc.total"].values)
    
    for measure in runs:
        for eps in runs[measure]:
            runs[measure][eps] = np.stack(runs[measure][eps])
    
    plot_scatter(runs)
    plot_accuracy(runs)
    plot_frozen(runs)
