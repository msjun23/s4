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
        x = self.lin(x)
        x = self.lif(x)
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
        x = self.lif_w(self.lin_w(x))               # [W, B, H, D]
        x = x.permute(2, 0, 1, 3)                   # [W, B, H, D] -> [H, W, B, D]
        x = x.reshape(self.height*self.width, B, D) # [H, W, B, D] -> [L, B, D]
        return x