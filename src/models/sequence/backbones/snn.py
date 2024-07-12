import torch
import torch.nn as nn

from spikingjelly.activation_based import neuron, layer, surrogate


class InputMaskingNet(nn.Module):
    def __init__(self, dim=512, mode='m'):
        super().__init__()
        
        self.lin = layer.Linear(in_features=dim, out_features=dim, step_mode=mode)
        self.lif = neuron.LIFNode(surrogate_function=surrogate.ATan(), step_mode=mode)
        
    def forward(self, x):
        x = self.lin(x)
        x = self.lif(x)
        return x