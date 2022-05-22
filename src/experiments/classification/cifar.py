from copy import deepcopy

import numpy as np
import torch
import wandb
from torch import nn
from torch.nn.functional import cross_entropy
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

from datasets import get_dataloaders
from experiments.templates.classification import ClassificationLearningExperiment
from metrics import Accuracy
from models import get_model
from neq import Hook
from optimizers import MaskedSGD, MaskedAdam
from utils import int2bool


class CIFAR10_Base(ClassificationLearningExperiment):
    def __init__(self, opts):
        super().__init__(opts)
        self.validation = None
    
    def get_experiment_key(self):
        return ""
    
    def compute_loss(self, outputs, minibatch):
        return {"ce": cross_entropy(outputs, minibatch[1])}
    
    def init_meters(self):
        return {"accuracy.top1": Accuracy(topk=(1,), config=self.opts)}
    
    def update_meters(self, meters, outputs, minibatch):
        meters["accuracy.top1"].update(outputs, minibatch[1])
    
    def optimize_metric(self):
        return None
    
    __help__ = "Template for learning experiments on CIFAR-10"
    
    @staticmethod
    def load_config(parser):
        ClassificationLearningExperiment.load_config(parser)
        parser.add_argument('--model', type=str, help='resnet variant',
                            choices=['resnet32-cifar10', 'resnet50-cifar10', 'resnet56-cifar10', 'resnet110-cifar10'],
                            default='resnet32-cifar10')
    
    def load_data(self):
        dataloaders = get_dataloaders(self.opts)
        self.validation = dataloaders["validation"]
        return {"train": dataloaders["train"], "test": dataloaders["test"]}
    
    def load_model(self):
        model, total_neurons = get_model(self.opts)
        
        return model, total_neurons
    
    def load_optimizer(self):
        if self.opts.optimizer == "sgd":
            optimizer = SGD(self.model.parameters(), lr=self.opts.lr,
                            weight_decay=self.opts.weight_decay, momentum=self.opts.momentum)
        if self.opts.optimizer == "adam":
            optimizer = Adam(self.model.parameters(), lr=self.opts.lr,
                             weight_decay=self.opts.weight_decay)
        
        scheduler = MultiStepLR(optimizer, milestones=[100, 150])
        
        return optimizer, scheduler


# TODO do the base class for gradient freezing (from which derive random and NEq)
class CIFAR10_NEq(CIFAR10_Base):
    __help__ = "Template for NEq experiments on CIFAR-10"
    
    def __init__(self, opts):
        super().__init__(opts)
        self.ignored_layers = ["classifier"]
        self.hooks = {}
        self.masks = {}
    
    @staticmethod
    def load_config(parser):
        CIFAR10_Base.load_config(parser)
        parser.add_argument('--eps', type=float, help='NEq eps', default=0)
        parser.add_argument('--mu', type=float, help='NEq velocity mu', default=0.5)
        parser.add_argument('--log-distributions', type=int2bool, help='log phi, delta phi, and velocity', default=0)
    
    def load_model(self):
        return super().load_model()
    
    def load_optimizer(self):
        named_params = list(map(list, zip(*list(self.model.named_parameters()))))
        if self.opts.optimizer == "sgd":
            optimizer = MaskedSGD(named_params[1], names=named_params[0], lr=self.opts.lr,
                                  weight_decay=self.opts.weight_decay, momentum=self.opts.momentum)
        if self.opts.optimizer == "adam":
            optimizer = MaskedAdam(named_params[1], names=named_params[0], lr=self.opts.lr,
                                   weight_decay=self.opts.weight_decay)
        
        scheduler = MultiStepLR(optimizer, milestones=[100, 150])
        
        return optimizer, scheduler
    
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
    
    def __update_masks(self):
        self.masks = {}
        
        for group in self.optimizer.param_groups:
            group["masks"] = self.masks
    
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
