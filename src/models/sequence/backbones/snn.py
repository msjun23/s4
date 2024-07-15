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
    
class DownsamplingLayer(nn.Module):
    def __init__(self, dim, input_length, output_length):
        super(DownsamplingLayer, self).__init__()
        self.conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=1, stride=input_length // output_length)

    def forward(self, x):
        # x: [seq_len, batch_size, feature_dim]
        x = x.permute(1, 2, 0)  # Change to [batch_size, feature_dim, seq_len]
        x = self.conv(x)
        x = x.permute(2, 0, 1)  # Change back to [seq_len, batch_size, feature_dim]
        return x
    
class InputMaskingNet(nn.Module):
    def __init__(self, dim=512, mode='m'):
        super().__init__()
        
        self.downsampling = DownsamplingLayer(dim=dim, input_length=16384, output_length=1024)
        self.lin = layer.Linear(in_features=dim, out_features=dim, step_mode=mode)
        self.lif = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode=mode)
        
    def forward(self, x):
        L, B, H = x.shape
        # Down sampling large sequence length
        x = self.downsampling(x)
        x = self.lin(x)
        x = self.lif(x)
        # Up sampling to original sequence length
        x = x.permute(1, 2, 0)  # [batch_size, dim, sequence_length]
        x = F.interpolate(x, size=L, mode='linear', align_corners=True).permute(2, 0, 1)
        return x