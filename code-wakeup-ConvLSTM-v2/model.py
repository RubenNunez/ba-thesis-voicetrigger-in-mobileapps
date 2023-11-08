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
        self.dropout = nn.Dropout(0.75)

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(4, 4), stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 4), stride=4)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(2, 4), stride=2)
        self.bn3 = nn.BatchNorm2d(128)

        # LSTM
        self.lstm = nn.LSTM(input_size=896, hidden_size=8, num_layers=2, batch_first=True, dropout=0.0)

        # Fully connected layers
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        # Preparing data for LSTM
        B, C, H, W = x.shape
        # Flatten the features while keeping the batch size and sequence length dimensions
        x = x.permute(0, 2, 1, 3)  # B, H, C, W
        x = x.reshape(B, H, -1)  # Flatten the features

        # LSTM layer
        h0 = torch.zeros(2, B, 8).to(self.device)
        c0 = torch.zeros(2, B, 8).to(self.device)
        x, _ = self.lstm(x, (h0, c0))

        
        # Final layers
        x = x[:, -1, :] 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
