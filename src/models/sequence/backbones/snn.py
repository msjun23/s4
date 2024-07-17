import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import neuron, layer, surrogate

class SNNBlock(nn.Module):
    def __init__(
            self, 
            step_mode='m', 
            # dim=128, 
            _layer=None, 
        ):
        super().__init__()
        
        self.step_mode = step_mode
        self.l_max = _layer.l_max
        
        # snn
        # self.lin = layer.Linear(in_features=dim, out_features=dim, step_mode=step_mode)
        self.snn = neuron.LIFNode(step_mode=step_mode, surrogate_function=surrogate.ATan())
        
    def forward(self, x, timesteps=-1):
        if self.step_mode == 'm':
            assert timesteps > 0
            # Repeat the input tensor by the number of timesteps
            x = x.unsqueeze(0).repeat(timesteps, 1, 1, 1)
            
            # To perform spiking operations with a fixed shape (L of each batch can be different for dataset)
            T, B, L, H = x.shape
            if self.l_max is not None:
                x = torch.cat([x, x.new_zeros((T, B, self.l_max - L, H))], dim=2)   # [T, B, L_max, H]
            
            # Restore original input shape (== sequence length)
            y = self.snn(x)[:,:,:L,:]
        elif self.step_mode == 's':
            B, L, H = x.shape
            if self.l_max is not None:
                x = torch.cat([x, x.new_zeros((B, self.l_max - L, H))], dim=1)   # [B, L_max, H]
                
            y = self.snn(x)[:,:L,:]
            # x = self.lin(x)
            # y = self.snn(x)
        
        return y
    
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
        self.lin_h = layer.Linear(in_features=dim, out_features=dim, step_mode=mode)
        self.lif_w = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode=mode)
        self.lif_h = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode=mode)
        
    def forward(self, x):
        # x: [L, B, D]
        L, B, D = x.shape
        x = x.view(self.height, self.width, B, D).permute(1, 2, 0, 3)   # [H, W, B, D] -> [W, B, H, D]
        x = self.lif_w(self.lin_w(x))
        x = x.permute(2, 1, 0, 3)   # [W, B, H, D] -> [H, B, W, D]
        x = self.lif_h(self.lin_h(x))
        
        x = x.permute(0, 2, 1, 3)   # [H, B, W, D] -> [H, W, B, D]
        x = x.reshape(self.height*self.width, B, D) # [H, W, B, D] -> [L, B, D]
        return x