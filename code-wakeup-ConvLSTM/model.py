import torch
import torch.nn as nn
import torch.nn.functional as F


# inspiration: https://github.com/omarzanji/ditto_activation/blob/main/main.py
# Input: [Batch_Size, Channels, Height, Width] = [Batch_Size, 1, 128, 128]
import torch
import torch.nn as nn
import torch.nn.functional as F

class WakeupTriggerConvLSTM(nn.Module):
    def __init__(self, device):
        super(WakeupTriggerConvLSTM, self).__init__()

        self.device = device
        self.dropout = nn.Dropout(0.5)

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(4, 128), stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 1), stride=4)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(2, 1), stride=2)
        self.bn3 = nn.BatchNorm2d(128)

        # LSTM
        self.lstm = nn.LSTM(input_size=128, hidden_size=32, num_layers=3, batch_first=True, dropout=0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(32, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Preparing data for LSTM
        B, C, H, W = x.shape
        x = x.squeeze(-1).transpose(1, 2)
        
        # LSTM layer
        h0 = torch.zeros(3, B, 32).to(self.device)
        c0 = torch.zeros(3, B, 32).to(self.device)
        x, _ = self.lstm(x, (h0, c0))
        
        # Final layers
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x
