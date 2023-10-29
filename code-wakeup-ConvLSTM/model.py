import torch
import torch.nn as nn
import torch.nn.functional as F


# inspiration: https://github.com/omarzanji/ditto_activation/blob/main/main.py
# Input: [Batch_Size, Channels, Height, Width] = [Batch_Size, 1, 128, 128]
class WakeupTriggerConvLSTM(nn.Module):
    def __init__(self, device):
        super(WakeupTriggerConvLSTM, self).__init__()

        self.device = device
        self.dropout = nn.Dropout(0.5)

        # Convolutional Layers - Running convolutions on the temporal axis
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(4, 128), stride=4) # Stride and kernel along width is as long as the frequency dimension
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 1), stride=4)
        self.bn2 = nn.BatchNorm2d(64)

        # LSTM Layer - To capture the longer-term temporal relationships
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=3, batch_first=True)

        # Fully Connected Layers
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)

        B, C, H, W = x.shape
        x = x.squeeze(-1).transpose(1, 2)
        
        # LSTM Processing
        h0 = torch.zeros(3, B, 32).to(self.device)
        c0 = torch.zeros(3, B, 32).to(self.device)
        x, _ = self.lstm(x, (h0, c0))
        
        # Use the last sequence from LSTM for prediction
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x
   