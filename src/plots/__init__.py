import matplotlib.pyplot as plt
import numpy as np
from torch import tensor
import torch


def plot_neurons_deltas(path):
    with open(path) as f:
        my_list = f.readlines()
        t = []
        to_plot = []
        for x in my_list:
            if "tensor" in x and len(t):
                to_plot.append(tensor(eval("".join(t)).clone().detach()).cpu().unsqueeze(0))
                t = []
            t.append(x)
        
        to_plot = torch.cat(to_plot)
        x = np.arange(to_plot.shape[0])
        for i in range(to_plot.shape[1]):
            plt.plot(x, to_plot[:, i])
            plt.xlim(right=30)
            plt.show()


if __name__ == '__main__':
    plot_neurons_deltas("C:\\Users\\Andrea\\Desktop\\layer4.0.bn2_deltas.txt")
