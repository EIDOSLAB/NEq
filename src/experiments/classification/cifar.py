from copy import deepcopy

from torch import nn
from torch.nn.functional import cross_entropy
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

from datasets import get_dataloaders
from experiments.templates.classification import ClassificationLearningExperiment
from metrics import Accuracy
from models import get_model
from neq import Hook, get_mask
from optimizers import MaskedSGD, MaskedAdam


class CIFAR10_Base(ClassificationLearningExperiment):
    def __init__(self, opts):
        super().__init__(opts)
        self.validation = None
    
    def get_experiment_key(self):
        return ""
    
    def compute_loss(self, outputs, minibatch):
        return {"ce": cross_entropy(outputs, minibatch[1])}
    
    def init_meters(self):
        return {"accuracy": {"top1": Accuracy(topk=(1,), config=self.opts)}}
    
    def update_meters(self, meters, outputs, minibatch):
        meters["accuracy"]["top1"].update(outputs, minibatch[1])
    
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
        train, self.validation, test = get_dataloaders(self.opts)
        return {"train": train, "test": test}
    
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
        parser.add_argument('--velocity-mu', type=float, help='NEq velocity mu', default=0.5)
    
    def load_model(self):
        model, total_neurons = super().load_model()
        self.__attach_hooks()
        return model, total_neurons
    
    def load_optimizer(self):
        named_params = list(map(list, zip(*list(self.model.named_parameters()))))
        if self.opts.optim == "sgd":
            optimizer = MaskedSGD(named_params[1], names=named_params[0], lr=self.opts.lr,
                                  weight_decay=self.opts.weight_decay, momentum=self.opts.momentum)
        if self.opts.optim == "adam":
            optimizer = MaskedAdam(named_params[1], names=named_params[0], lr=self.opts.lr,
                                   weight_decay=self.opts.weight_decay)
        
        scheduler = MultiStepLR(optimizer, milestones=[100, 150])
        
        return optimizer, scheduler
    
    def run(self):
        self.__activate_hooks(True)
        self.iter_dataloader("validation", self.validation, False)
        
        for epoch in range(1, self.opts.epochs + 1):
            self.__activate_hooks(False)
            self.run_epoch(epoch)
            
            self.__activate_hooks(True)
            self.iter_dataloader("validation", self.validation, False)
            
            self.__compute_masks(epoch)
    
    def __attach_hooks(self):
        for n, m in self.model.named_modules():
            if n not in self.ignored_layers and isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm)):
                self.hooks[n] = Hook(n, m, self.opts.velocity_mu)
    
    def __activate_hooks(self, active):
        for h in self.hooks:
            self.hooks[h].active = active
            
    def __compute_masks(self, epoch):
        for k in self.hooks:
        
            phi = deepcopy(self.hooks[k].get_reduced_activation_delta().detach().clone())
            d_phi = deepcopy(self.hooks[k].get_delta_of_delta().detach().clone())
            velocity = deepcopy(self.hooks[k].get_velocity().detach().clone())
        
            get_mask(self.opts, epoch, k, velocity, self.masks)
        
            self.hooks[k].update_velocity()
            self.hooks[k].update_delta_buffer()
        
            self.hooks[k].reset()
