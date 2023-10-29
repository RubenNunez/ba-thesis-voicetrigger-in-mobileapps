import torch
import torch.nn as nn
import torch.nn.functional as F


# inspiration: https://github.com/omarzanji/ditto_activation/blob/main/main.py
# Input: [Batch_Size, Channels, Height, Width] = [Batch_Size, 1, 128, 128]
class WakeupTriggerConvLSTM (nn.Module):
    def __init__(self, use_cuda=False):
        super(WakeupTriggerConvLSTM , self).__init__()

        self.use_cuda = use_cuda

        self.conv1 = nn.Conv2d(1, 28, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(28)

        self.conv2 = nn.Conv2d(28, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 60, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(60)

        self.lstm = nn.LSTM(input_size=120, hidden_size=20, num_layers=1, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(40, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.max_pool2d(x, kernel_size=2, padding=1)

        x = F.relu(self.conv3(x))
        x = self.bn3(x)

        B, C, H, W = x.shape
        x = x.view(B, 2, -1)

        h0 = torch.zeros(2, B, 20)
        c0 = torch.zeros(2, B, 20)

        if self.use_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        x, _ = self.lstm(x, (h0, c0))
        x = F.relu(self.fc1(x[:, -1, :]))
        x = F.dropout(x, p=0.3)

        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.3)

        x = torch.sigmoid(self.fc3(x))

        return x
