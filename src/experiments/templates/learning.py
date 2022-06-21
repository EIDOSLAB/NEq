import os
import time
from abc import abstractmethod
from collections import defaultdict

import torch
import wandb
from filelock import FileLock
from tqdm import tqdm

import metrics
import utils
from experiments.templates.base import ExperimentBase


class LearningExperiment(ExperimentBase):
    def __init__(self, opts):
        super().__init__(opts)
        self.total_neurons = None
        self.dataloaders = None
        self.scheduler = None
        self.optimizer = None
        self.model = None
        self.scaler = None
        self.project_name = "NEq"
        self.tag = "learning-base"
        self.last_logs = {}
    
    @staticmethod
    def load_config(parser):
        parser.add_argument('--root', type=str,
                            help='data root', default='/data')
        
        parser.add_argument('--epochs', type=int,
                            help='training epochs', default=250)
        parser.add_argument('--lr', type=float,
                            help='initial learning rate', default=0.1)
        parser.add_argument('--weight-decay', type=float,
                            help='weight decay', default=5e-4)
        parser.add_argument('--momentum', type=float,
                            help='momentum', default=0.9)
        parser.add_argument('--batch-size', type=int,
                            help='training batch size', default=100)
        parser.add_argument('--dataset', type=str,
                            help='dataset', default="cifar10")
        parser.add_argument('--validation-size', type=float,
                            help='validation size', default=0.1)
        parser.add_argument('--seed', type=int,
                            help='random seed', default=0)
        parser.add_argument('--optimizer', type=str, choices=['sgd', 'adam'],
                            help='optmizer', default='sgd')
        
        parser.add_argument('--num-workers', type=int,
                            help='number of dataloader workers', default=8)
        parser.add_argument('--save-checkpoints', type=utils.arg2bool,
                            help='enable checkpoint saving', default=False)
        parser.add_argument('--amp', type=utils.arg2bool,
                            help='use amp', default=True)
    
    def initialize(self, logging=True):
        print('Initializing', self.__class__.__name__, 'experiment')
        # Set the reproducibility seeds
        utils.set_seed(self.opts.seed)
        
        # Create the amp grad scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.opts.amp)
        
        # Build the model
        self.model, self.total_neurons = self.load_model()
        self.model = self.model.to(self.device)
        
        # Build the optimizer
        self.optimizer = self.load_optimizer()
        
        # Build the scheduler
        self.scheduler = self.load_scheduler()
        
        if self.optimizer is not None:
            self.opts.optimizer = self.optimizer.__class__.__name__
        
        if self.scheduler is not None:
            self.opts.scheduler = self.scheduler.__class__.__name__
        
        if logging:
            # Initialize logging instruments (wandb and tensorboard)
            # TODO add SummaryWriter
            wandb.init(config=self.opts,
                       project=self.project_name,
                       tags=[self.tag])
        
        # Load the dataloaders
        with FileLock(self.__class__.__name__):
            self.dataloaders = self.load_data()
        
        print('Model:', self.model.__class__.__name__)
        print('Optimizer:', self.optimizer)
        print('Scheduler:', self.scheduler)
        
        print('\n---- Model details ----')
        print(self.model)
    
    def run(self):
        for epoch in range(1, self.opts.epochs + 1):
            self.run_epoch(epoch)
    
    @abstractmethod
    def load_model(self):
        raise NotImplementedError
    
    @abstractmethod
    def load_optimizer(self):
        raise NotImplementedError
    
    @abstractmethod
    def load_scheduler(self):
        raise NotImplementedError
    
    @abstractmethod
    def compute_loss(self, outputs, minibatch):
        raise NotImplementedError
    
    @abstractmethod
    def init_meters(self):
        raise NotImplementedError
    
    @abstractmethod
    def update_meters(self, meters, outputs, minibatch):
        raise NotImplementedError
    
    @abstractmethod
    def optimize_metric(self):
        raise NotImplementedError
    
    def predict(self, minibatch):
        x = minibatch[0]
        return self.model(x)
    
    def predict_Y(self, dataloader, index):
        labels = torch.tensor([])
        outputs = torch.tensor([], device=self.device)
        
        for minibatch in tqdm(dataloader, desc='Predicting:'):
            minibatch[0], minibatch[1] = minibatch[0].to(self.device), minibatch[1].to(self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(self.scaler is not None):
                    y = self.predict(minibatch)
            
            outputs = torch.cat((outputs, y[index[0]].detach()), dim=0)
            
            # TODO: this is not clean
            if isinstance(index[1], list) or isinstance(index[1], tuple):
                curr_labels = torch.tensor([])
                for i in index[1]:
                    curr_labels = torch.cat((curr_labels, minibatch[i][:, None]), dim=1)
            else:
                curr_labels = minibatch[index[1]]
            
            labels = torch.cat((labels, curr_labels), dim=0)
        
        return outputs, labels
    
    def iter_dataloader(self, name, dataloader, train):
        meters = self.init_meters()
        running_losses = defaultdict(metrics.AverageMeter)
        
        self.model.train(train)
        for idx, minibatch in enumerate(dataloader):
            minibatch[0], minibatch[1] = minibatch[0].to(self.device), minibatch[1].to(self.device)
            with torch.set_grad_enabled(train):
                with torch.cuda.amp.autocast(enabled=self.opts.amp):
                    outputs = self.predict(minibatch)
                    losses = self.compute_loss(outputs, minibatch)
                    total_loss = sum(losses.values())
            
            if train:
                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            self.update_meters(meters, outputs, minibatch)
            for k, v in losses.items():
                running_losses[k].update(v.item(), len(minibatch[0]))
            
            if (idx + 1) % 1 == 0:
                print(f'{name} :: [ step {idx + 1}/{len(dataloader)} ] :: ',
                      ' '.join([f'running {n}={l.value()}' for n, l in running_losses.items()]))
        
        logs = {name: m.value() for name, m in meters.items()}
        logs.update({k: m.value() for k, m in running_losses.items()})
        return logs
    
    def fit(self):
        # Iterate over the dataset
        return self.iter_dataloader('train', self.dataloaders['train'], True)
    
    def test(self):
        for name, dataloader in self.dataloaders.items():
            if name == 'train':
                continue
            
            logs = self.iter_dataloader(name, dataloader, False)
            yield name, logs
    
    def step_schedulers(self):
        if self.scheduler is not None:
            self.scheduler.step()
    
    def run_epoch(self, epoch):
        print(f'\n----- Epoch {epoch} -----')
        
        wandb.log({"epochs": epoch}, step=epoch)
        
        # Train on the dataloader for one epoch
        start_t = time.perf_counter()
        train_logs = self.fit()
        end_t = time.perf_counter()
        
        # Print logs
        print('Train:', train_logs)
        wandb.log({'train': train_logs, 'train_t': end_t - start_t}, step=epoch)
        
        # Perform tests on the other dataloaders and print logs
        for test_name, test_logs in self.test():
            print(f'{test_name}:', test_logs)
            wandb.log({test_name: test_logs}, step=epoch)
        
        # Scheduler step
        self.step_schedulers()
        
        optimize_metric = self.optimize_metric()
        if optimize_metric is not None:
            wandb.log(optimize_metric, step=epoch)
        
        # Log the learning rates
        wandb.log({f"lr-{x}": group["lr"] for x, group in enumerate(self.optimizer.param_groups)}, step=epoch)
        
        # Save the model checkpoint
        if self.opts.save_checkpoints:
            print('Saving checkpoint to', wandb.run.dir)
            torch.save({'model': self.model.state_dict()}, os.path.join(wandb.run.dir, 'model.pt'))
    
    def __str__(self):
        # TODO find a better naming convention
        return f'{self.__class__.__name__}-'
