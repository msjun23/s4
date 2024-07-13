import torch
import torch.nn as nn
import torch.nn.functional as F


class ImportanceWeighting(nn.Module):
    def __init__(self, input_dim):
        super(ImportanceWeighting, self).__init__()
        self.weight_layer = nn.Linear(input_dim, 1)  # Learnable weight layer
    
    def forward(self, x):
        # Calculate weights for the input sequence
        weights = self.weight_layer(x)  # (batch_size, seq_len, 1)
        weights = torch.sigmoid(weights)  # (batch_size, seq_len, 1)
        
        # Multiply the input sequence by the weights
        weighted_x = x * weights  # (batch_size, seq_len, embed_dim)
        
        return weighted_x, weights
    
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # Encoder step
        encoded = F.relu(self.encoder(x))
        # Decoder step
        decoded = self.decoder(encoded)
        return decoded