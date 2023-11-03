import torch
import torch.nn as nn
import torch.nn.functional as F


# inspiration: https://github.com/omarzanji/ditto_activation/blob/main/main.py
# Input: [Batch_Size, Channels, Height, Width] = [Batch_Size, 1, 128, 256]
# represents 2 seconds of audio at 16kHz
import torch
import torch.nn as nn
import torch.nn.functional as F

class WakeupTriggerConvLSTM2s(nn.Module):
    def __init__(self, device):
        super(WakeupTriggerConvLSTM2s, self).__init__()

        self.device = device
        self.dropout = nn.Dropout(0.5)

        # Adjust the convolution layers to accommodate the new input shape
        # Input: [Batch_Size, 1, 128, 256]

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(4, 4), stride=4)  # Adjust kernel size and stride
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 4), stride=4)  # Adjust kernel size and stride
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(2, 4), stride=2)  # Adjust kernel size and stride
        self.bn3 = nn.BatchNorm2d(128)

        # LSTM
        self.lstm = nn.LSTM(input_size=896, hidden_size=32, num_layers=3, batch_first=True, dropout=0.5)

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
        x = x.view(B, H, C*W)
        
        # LSTM layer
        h0 = torch.zeros(3, B, 32).to(self.device)
        c0 = torch.zeros(3, B, 32).to(self.device)
        x, _ = self.lstm(x, (h0, c0))
        
        # Final layers
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
