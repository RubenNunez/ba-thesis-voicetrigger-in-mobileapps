
"""wake-up model"""

import torch
import torch.nn as nn


class WakeupModel_LSTM(nn.Module):
    def __init__(self, num_classes, feature_size, hidden_size, num_layers, 
                 dropout, bidirectional, device='cpu'):
        
        super(WakeupModel_LSTM, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.direction = 2 if bidirectional else 1
        self.device = device
        self.layer_norm = nn.LayerNorm(feature_size)
        self.LSTM = nn.LSTM(feature_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout, 
                            bidirectional=bidirectional)
        self.classifier = nn.Linear(hidden_size * self.direction, num_classes)

    def _init_hidden(self, batch_size):
        n, d, hs = self.num_layers, self.directions, self.hidden_size
        return (torch.zeros(n*d, batch_size, hs).to(self.device),
                torch.zeros(n*d, batch_size, hs).to(self.device))
    
    def forward(self, x):
        x = self.layer_norm(x)
        hidden = self._init_hidden(x.size()[1])
        out, (hn, cn) = self.LSTM(x, hidden)
        out = self.classifier(hn)
        return out
        
                 
