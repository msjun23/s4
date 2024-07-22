import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import neuron, layer, surrogate

    
class SequenceMaskingNet(nn.Module):
    def __init__(self, dim=512, mode='m'):
        super().__init__()
        
        self.lin = layer.Linear(in_features=dim, out_features=dim, step_mode=mode)
        self.lif = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode=mode)
        
    def forward(self, x):
        # x: [L, B, D]
        reverse_x = x.flip(dims=[0])            # Reverse order of x, [L, B, D]
        x = torch.stack([x, reverse_x], dim=2)  # [L, B, 2, D]
        x = self.lif(self.lin(x))               # [L, B, 2, D]
        x = x[:,:,0,:] + x[:,:,1,:].flip(dims=[0])  # [L, B, D]
        return x
    
class ImageMaskingNet(nn.Module):
    def __init__(self, dim=512, mode='m', height=128, width=128):
        super().__init__()
        
        self.height = height
        self.width  = width
        
        # w: width, h: height
        self.lin_w = layer.Linear(in_features=dim, out_features=dim, step_mode=mode)
        # self.lin_h = layer.Linear(in_features=dim, out_features=dim, step_mode=mode)
        self.lif_w = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode=mode)
        # self.lif_h = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode=mode)
        
    def forward(self, x):
        # x: [L, B, D]
        L, B, D = x.shape
        
        x = x.view(self.height, self.width, B, D).permute(1, 2, 0, 3)   # [H, W, B, D] -> [W, B, H, D]
        x = x.unsqueeze(dim=2)                      # [W, B, 1, H, D]
        reverse_x = x.flip(dims=[0])                # Reverse order of x, [W, B, 1, H, D]
        x = torch.cat([x, reverse_x], dim=2)        # [W, B, 2, H, D]
        x = self.lif_w(self.lin_w(x))               # Bi-directional SNN [W, B, 2, H, D]
        x = x[:,:,0,:,:] + x[:,:,1,:,:].flip(dims=[0])  # [W, B, H, D]
        x = x.permute(2, 0, 1, 3)                   # [W, B, H, D] -> [H, W, B, D]
        x = x.reshape(self.height*self.width, B, D) # [H, W, B, D] -> [L, B, D]
        return x