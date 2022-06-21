import numpy as np
import torch
import wandb
from torch import nn

from experiments.templates.resnet import CIFAR10_Freeze_Bprop_Base
from neq import Hook
from utils import arg2bool


class CIFAR10_Freeze_Bprop_Random_Constant(CIFAR10_Freeze_Bprop_Base):
    __help__ = "Implementation of Freezing Back-propagation experiments on CIFAR-10 (random, constant)"
    
    def __init__(self, opts):
        super().__init__(opts)
        self.tag = "freeze-bprop-random-constant"
    
    @staticmethod
    def load_config(parser):
        CIFAR10_Freeze_Bprop_Base.load_config(parser)
        parser.add_argument('--p', type=float, help='gradient masking probability', default=0)
    
    def compute_masks(self, epoch):
        super().compute_masks(epoch)
        for n, m in self.model.named_modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
                if epoch > self.opts.warmup:
                    self.masks[n] = torch.where(
                        torch.empty(m.weight.shape[0], device=self.device).uniform_() > 1 - self.opts.p
                    )[0]
                else:
                    self.masks[n] = torch.tensor([], device=self.device)
                
                self.masks[n] = self.masks[n].to(torch.long)


class CIFAR10_NEq(CIFAR10_Freeze_Bprop_Base):
    __help__ = "Implementation of NEq experiments on CIFAR-10"
    
    def __init__(self, opts):
        super().__init__(opts)
        self.ignored_layers = ["classifier"]
        self.hooks = {}
    
    @staticmethod
    def load_config(parser):
        CIFAR10_Freeze_Bprop_Base.load_config(parser)
        parser.add_argument('--eps', type=float,
                            help='NEq eps', default=0)
        parser.add_argument('--mu', type=float,
                            help='NEq velocity mu', default=0.5)
        parser.add_argument('--log-distributions', type=arg2bool,
                            help='log phi, delta phi, and velocity', default=False)
    
    def run(self):
        self.attach_hooks()
        self.activate_hooks(True)
        self.iter_dataloader("validation", self.validation, False)
        self.hook_step(0)
        
        for epoch in range(1, self.opts.epochs + 1):
            self.activate_hooks(False)
            self.run_epoch(epoch, False)
            
            self.activate_hooks(True)
            self.iter_dataloader("validation", self.validation, False)
            self.hook_step(epoch)
            
            self.compute_masks(epoch)
            self.log_masks(epoch)
    
    def attach_hooks(self):
        for n, m in self.model.named_modules():
            if n not in self.ignored_layers and isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm)):
                self.hooks[n] = Hook(n, m, self.opts.mu)
    
    def activate_hooks(self, active):
        for h in self.hooks:
            self.hooks[h].active = active
    
    def hook_step(self, neq_steps):
        print(f"NEq :: step {neq_steps}")
        for h in self.hooks:
            self.hooks[h].step()
    
    def compute_masks(self, epoch):
        super().compute_masks(epoch)
        distributions = {"y": {}, "phi": {}, "delta-phi": {}, "velocity": {}}
        
        for k in self.hooks:
            hook = self.hooks[k]
            
            if self.opts.log_distributions:
                self.log_distributions(distributions, hook, k)
            
            if epoch > self.opts.warmup and hook.velocity is not None:
                self.masks[k] = torch.where(torch.abs(hook.velocity) < self.opts.eps)[0]
            else:
                self.masks[k] = torch.tensor([], device=self.device)
            
            self.masks[k] = self.masks[k].to(torch.long)

        wandb.log(distributions, step=epoch)
    
    @staticmethod
    def log_distributions(distributions, hook, k):
        
        print(f"NEq :: log distributions for module {k}")
        
        if hook.y_prev is not None:
            print("NEq :: log y")
            print(hook.y_prev.mean(dim=0).shape)
            distributions["y"][f"{k}"] = wandb.Histogram(
                np_histogram=np.histogram(
                    hook.y_prev.mean(dim=0).cpu().numpy(),
                    bins=min(512, hook.y_prev.mean(dim=0).shape[0])
                )
            )
        if hook.phi_prev is not None:
            print("NEq :: log phi")
            print(hook.phi_prev.shape)
            distributions["phi"][f"{k}"] = wandb.Histogram(
                np_histogram=np.histogram(
                    hook.phi_prev.cpu().numpy(),
                    bins=min(512, hook.phi_prev.shape[0])
                )
            )
        if hook.delta_phi_prev is not None:
            print("NEq :: log delta phi")
            print(hook.delta_phi_prev.shape)
            distributions["delta-phi"][f"{k}"] = wandb.Histogram(
                np_histogram=np.histogram(
                    hook.delta_phi_prev.cpu().numpy(),
                    bins=min(512, hook.delta_phi_prev.shape[0])
                )
            )
        if hook.velocity is not None:
            print("NEq :: log velocity")
            print(hook.velocity.shape)
            distributions["velocity"][f"{k}"] = wandb.Histogram(
                np_histogram=np.histogram(
                    hook.velocity.cpu().numpy(),
                    bins=min(512, hook.velocity.shape[0])
                )
            )
