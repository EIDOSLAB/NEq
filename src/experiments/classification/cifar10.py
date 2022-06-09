from copy import deepcopy

import numpy as np
import torch
import wandb
from torch import nn

from experiments.templates.cifar10 import CIFAR10_Freeze_Bprop_Base
from neq import Hook
from utils import int2bool


class CIFAR10_Freeze_Bprop_Random_Constant(CIFAR10_Freeze_Bprop_Base):
    __help__ = "Implementation of Freezing Back-propagation experiments on CIFAR-10 (random, constant)"
    
    @staticmethod
    def load_config(parser):
        CIFAR10_Freeze_Bprop_Base.load_config(parser)
        parser.add_argument('--p', type=float, help='gradient masking probability', default=0)
    
    def __compute_masks(self, epoch):
        # TODO here we have to create a random tensor
        #  with int(m.weight.shape[0] * p) elements in range [0, m.weight.shape[0]]
        if epoch > 1:
            for n, m in self.model.named_modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    self.masks[n] = 0


class CIFAR10_Freeze_Bprop_Bottomk_Constant(CIFAR10_Freeze_Bprop_Base):
    __help__ = "Implementation of Freezing Back-propagation experiments on CIFAR-10 (bottomk, constant)"
    
    @staticmethod
    def load_config(parser):
        CIFAR10_Freeze_Bprop_Base.load_config(parser)
        parser.add_argument('--p', type=float, help='gradient masking probability', default=0)
    
    def __compute_masks(self, epoch):
        # TODO here we have to create a tensor
        #  with the indices of the int(m.weight.shape[0] * p) smallest parameters
        if epoch > 1:
            for n, m in self.model.named_modules():
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    self.masks[n] = 0


class CIFAR10_NEq(CIFAR10_Freeze_Bprop_Base):
    __help__ = "Implementation of NEq experiments on CIFAR-10"
    
    def __init__(self, opts):
        super().__init__(opts)
        self.ignored_layers = ["classifier"]
        self.hooks = {}
    
    @staticmethod
    def load_config(parser):
        CIFAR10_Freeze_Bprop_Base.load_config(parser)
        parser.add_argument('--eps', type=float, help='NEq eps', default=0)
        parser.add_argument('--mu', type=float, help='NEq velocity mu', default=0.5)
        parser.add_argument('--log-distributions', type=int2bool, help='log phi, delta phi, and velocity', default=0)
    
    def run(self):
        self.__attach_hooks()
        self.__activate_hooks(True)
        self.iter_dataloader("validation", self.validation, False)
        
        for k in self.hooks:
            self.hooks[k].reset(self.hooks[k].get_samples_activation())
        
        for epoch in range(1, self.opts.epochs + 1):
            self.__activate_hooks(False)
            self.run_epoch(epoch)
            
            self.__activate_hooks(True)
            self.iter_dataloader("validation", self.validation, False)
            
            self.__update_masks()
            self.__compute_masks(epoch)
    
    def __attach_hooks(self):
        for n, m in self.model.named_modules():
            if n not in self.ignored_layers and isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm)):
                self.hooks[n] = Hook(n, m, self.opts.mu)
    
    def __activate_hooks(self, active):
        for h in self.hooks:
            self.hooks[h].active = active
    
    def __compute_masks(self, epoch):
        neq_neurons = {}
        distributions = {"phi": {}, "delta-phi": {}, "velocity": {}}
        
        for k in self.hooks:
            if self.opts.log_distributions:
                phi = deepcopy(self.hooks[k].get_reduced_activation_delta().detach().clone())
                delta_phi = deepcopy(self.hooks[k].get_delta_of_delta().detach().clone())
            velocity = deepcopy(self.hooks[k].get_velocity().detach().clone())
            
            if epoch > 1:
                self.masks[k] = torch.where(torch.abs(velocity) < self.opts.eps)[0]
            else:
                self.masks[k] = torch.tensor([], device=velocity.device)
            
            self.masks[k] = self.masks[k].to(torch.long)
            
            if self.opts.log_distributions:
                distributions["phi"][f"{k}"] = wandb.Histogram(
                    np_histogram=np.histogram(phi.cpu().numpy(), bins=min(512, phi.shape[0]))
                )
                distributions["delta-phi"][f"{k}"] = wandb.Histogram(
                    np_histogram=np.histogram(delta_phi.cpu().numpy(), bins=min(512, delta_phi.shape[0]))
                )
                distributions["velocity"][f"{k}"] = wandb.Histogram(
                    np_histogram=np.histogram(velocity.cpu().numpy(), bins=min(512, velocity.shape[0]))
                )
            
            self.hooks[k].update_velocity()
            self.hooks[k].update_delta_buffer()
            
            self.hooks[k].reset()
            
            neq_neurons[f"{k}"] = (self.masks[k].shape[0] / velocity.shape[0]) * 100
        
        wandb.log({
            "neq-neurons": neq_neurons,
            "total":       (sum(neq_neurons.values()) / self.total_neurons) * 100
        }, step=epoch)
        wandb.log(distributions, step=epoch)