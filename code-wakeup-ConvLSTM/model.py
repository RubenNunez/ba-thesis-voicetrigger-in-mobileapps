import torch
import torch.nn as nn
import torch.nn.functional as F


# inspiration: https://github.com/omarzanji/ditto_activation/blob/main/main.py
# Input: [Batch_Size, Channels, Height, Width] = [Batch_Size, 1, 128, 128]
class WakeupTriggerConvLSTM (nn.Module):
    def __init__(self, device):
        super(WakeupTriggerConvLSTM , self).__init__()

        self.device = device

                # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=4)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=4)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=4)
        self.bn3 = nn.BatchNorm2d(64)

        # LSTM Layer
        self.lstm = nn.LSTM(input_size=64, hidden_size=16, num_layers=1, batch_first=True, bidirectional=True)

        # Fully Connected Layers
        self.fc4 = nn.Linear(256, 64)
        self.fc6 = nn.Linear(96, 32)
        self.fc7 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)

        B, C, H, W = x.shape
        x = x.view(B, 1, -1)
        
        # Pass through fc4
        x_fc4 = F.relu(self.fc4(x))
        
        # LSTM Processing
        h0 = torch.zeros(2, B, 16).to(self.device)
        c0 = torch.zeros(2, B, 16).to(self.device)
        x_lstm, _ = self.lstm(x_fc4, (h0, c0))

        # Concatenate LSTM and fc4 outputs
        x_concat = torch.cat((x_fc4, x_lstm), 2)

        # Further FC layers
        x = F.relu(self.fc6(x_concat))
        x = torch.sigmoid(self.fc7(x))

        return x
    