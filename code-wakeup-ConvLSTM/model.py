import torch
import torch.nn as nn
import torch.nn.functional as F


# inspiration: https://github.com/omarzanji/ditto_activation/blob/main/main.py
# Input: [Batch_Size, Channels, Height, Width] = [Batch_Size, 1, 128, 128]
class WakeupTriggerConvLSTM(nn.Module):
    def __init__(self, device):
        super(WakeupTriggerConvLSTM, self).__init__()

        self.device = device

        self.dropout = nn.Dropout(0.5) # 50% dropout

        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=4)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=4)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=4)
        self.bn3 = nn.BatchNorm2d(64)

        # Calculate the output shape after convolutions
        self.fc4 = nn.Linear(256, 64) 
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, num_layers=3, batch_first=True) # Here, each sequence item has only 1 feature.

        # Fully Connected Layers
        self.fc6 = nn.Linear(64 + 16, 32) # 64 from fc4 and 16 from LSTM
        self.fc7 = nn.Linear(32, 1)
    
    # Input: [Batch_Size, Channels, Height, Width] = [Batch_Size, 1, 128, 128]
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)

        B, C, H, W = x.shape
        x = x.view(B, C * H * W)
        
        # Pass through fc4
        x_fc4 = F.relu(self.fc4(x))
        x_fc4 = self.dropout(x_fc4)
        
        # Reshape for LSTM: [Batch_Size, Sequence_Length, Feature_Size]
        x_fc4 = x_fc4.unsqueeze(-1)  # adds the feature dimension

        # LSTM Processing
        h0 = torch.zeros(3, B, 16).to(self.device)
        c0 = torch.zeros(3, B, 16).to(self.device)
        x_lstm, _ = self.lstm(x_fc4, (h0, c0))

        # Concatenate LSTM and fc4 outputs
        x_concat = torch.cat((x_fc4.squeeze(-1), x_lstm[:, -1, :]), 1) # only take the last sequence from LSTM

        # Further FC layers
        x = F.relu(self.fc6(x_concat))
        x = torch.sigmoid(self.fc7(x))

        return x

    