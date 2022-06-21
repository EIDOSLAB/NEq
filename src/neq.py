import torch
from torch import nn


class Hook:
    
    def __init__(self, name, module, momentum=0):
        self.name = name
        self.module = module
        self.momentum = momentum
        
        self.activations = []
        
        self.y_prev = None
        self.phi_prev = None
        self.delta_phi_prev = None
        self.velocity = None
        
        self.active = True
        
        self.hook = module.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        if not self.active:
            return
        
        if isinstance(module, nn.Linear):
            flat_output = output.view(output.shape[0], output.shape[1])
        elif isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            flat_output = output.view(output.shape[0], output.shape[1], -1)
        
        self.activations.append(flat_output)
    
    def step(self):
        if not self.active:
            return
        
        print(f"NEq :: module {self.name}")
        
        y_curr = torch.cat(self.activations)
        print("NEq :: evaluate y_curr")
        
        if self.y_prev is not None:
            phi_curr = torch.cosine_similarity(self.y_prev.to(torch.float64), y_curr.to(torch.float64), dim=2)
            phi_curr = phi_curr.to(torch.float16).mean(dim=0)
            print("NEq :: evaluate phi_curr")
            
            if self.phi_prev is not None:
                delta_phi_curr = phi_curr - self.phi_prev
                print("NEq :: evaluate delta_phi_curr")
                
                if self.delta_phi_prev is not None:
                    
                    if self.velocity is not None:
                        self.velocity = delta_phi_curr - self.momentum * self.velocity
                        print("NEq :: update velocity")
                    else:
                        self.velocity = delta_phi_curr - self.momentum * self.delta_phi_prev
                        print("NEq :: evaluate velocity")
                
                self.delta_phi_prev = delta_phi_curr.detach().clone()
                print("NEq :: update delta_phi_prev")
            
            self.phi_prev = phi_curr.detach().clone()
            print("NEq :: update phi_prev")
        
        self.y_prev = y_curr.detach().clone()
        print("NEq :: update y_prev")
        
        self.activations = []
    
    def close(self) -> None:
        self.hook.remove()
    
    def activate(self, active):
        self.active = active
