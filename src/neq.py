import torch


class Hook:
    
    def __init__(self, name, module, momentum=0) -> None:
        self.name = name
        self.module = module
        self.samples_activation = []
        self.previous_activations = None
        self.activation_deltas = 0
        self.total_samples = 0
        
        self.momentum = momentum
        self.delta_buffer = 0
        self.velocity = 0
        
        self.active = True
        
        self.hook = module.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        
        if not self.active:
            return
        
        # TODO sort this mess
        reshaped_output = output.view((output.shape[0], output.shape[1], -1) if len(output.shape) > 2
                                      else (output.shape[0], output.shape[1]))
        
        # if self.config.dataset in ["imagenet", "coco"]:
        #     reshaped_output = reshaped_output.cpu()
        
        if self.previous_activations is None:
            self.samples_activation.append(reshaped_output)
        else:
            previous = self.previous_activations[self.total_samples:output.shape[0] + self.total_samples].float()
            delta = 1 - cosine_similarity(
                reshaped_output.float(),
                previous,
                dim=0 if len(output.shape) <= 2 else 2  # TODO do this but with nn.layer
            )
            
            if len(output.shape) > 2:
                delta = torch.sum(delta, dim=0)
            
            self.activation_deltas += delta
            
            self.previous_activations[self.total_samples:output.shape[0] + self.total_samples] = reshaped_output
            self.total_samples += output.shape[0]
    
    def get_samples_activation(self):
        return torch.cat(self.samples_activation)
    
    def get_reduced_activation_delta(self):
        return self.activation_deltas / self.total_samples
    
    def get_delta_of_delta(self):
        reduced_activation_delta = self.get_reduced_activation_delta()
        delta_of_delta = self.delta_buffer - reduced_activation_delta
        
        return delta_of_delta
    
    def get_velocity(self):
        self.velocity += self.get_delta_of_delta()
        
        return self.velocity
    
    def update_delta_buffer(self):
        self.delta_buffer = self.get_reduced_activation_delta()
    
    def update_velocity(self):
        self.velocity *= self.momentum
        self.velocity -= self.get_delta_of_delta()
    
    def reset(self, previous_activations=None):
        self.samples_activation = []
        self.activation_deltas = 0
        self.total_samples = 0
        if previous_activations is not None:
            self.previous_activations = previous_activations
    
    def close(self) -> None:
        self.hook.remove()
    
    def activate(self, active):
        self.active = active


def cosine_similarity(x1, x2, dim, eps=1e-8):
    x1_squared_norm = torch.pow(x1, 2).sum(dim=dim, keepdim=True)
    x2_squared_norm = torch.pow(x2, 2).sum(dim=dim, keepdim=True)
    
    # x1_squared_norm.clamp_min_(eps)
    # x2_squared_norm.clamp_min_(eps)
    
    x1_norm = x1_squared_norm.sqrt_()
    x2_norm = x2_squared_norm.sqrt_()
    
    x1_normalized = x1.div(x1_norm).nan_to_num(nan=0, posinf=0, neginf=0)
    x2_normalized = x2.div(x2_norm).nan_to_num(nan=0, posinf=0, neginf=0)
    
    mask_1 = (torch.abs(x1_normalized).sum(dim=dim) <= eps) * (torch.abs(x2_normalized).sum(dim=dim) <= eps)
    mask_2 = (torch.abs(x1_normalized).sum(dim=dim) > eps) * (torch.abs(x2_normalized).sum(dim=dim) > eps)
    
    cos_sim_value = torch.sum(x1_normalized * x2_normalized, dim=dim)
    
    return mask_2 * cos_sim_value + mask_1
