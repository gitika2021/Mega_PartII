import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridConvNet(nn.Module):
    def __init__(self, input_length=120, n=1):
        super().__init__()
        
        def ch(val):
            return int(val * n)
        self.bn_input = nn.BatchNorm1d(1, eps=0.001, momentum=0.01)
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=ch(16), kernel_size=5, padding=2)
        self.conv1_2 = nn.Conv1d(in_channels=ch(16), out_channels=ch(16), kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2) # 120 -> 60

        self.conv2_1 = nn.Conv1d(in_channels=ch(16), out_channels=ch(32), kernel_size=5, padding=2)
        self.conv2_2 = nn.Conv1d(in_channels=ch(32), out_channels=ch(32), kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2) # 60 -> 30

        self.conv3_1 = nn.Conv1d(in_channels=ch(32), out_channels=ch(64), kernel_size=5, padding=2)
        self.conv3_2 = nn.Conv1d(in_channels=ch(64), out_channels=ch(64), kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=2, padding=2) # 30 -> 15
        self.flat_features = ch(64) * 15
        
        self.fc1 = nn.Linear(self.flat_features, ch(256))
        self.fc2 = nn.Linear(ch(256), ch(256))
        self.fc3 = nn.Linear(ch(256), 1444) 
        self.conv2d_1 = nn.Conv2d(in_channels=1, out_channels=ch(32), kernel_size=3, padding=1)
        self.conv2d_2 = nn.Conv2d(in_channels=ch(32), out_channels=ch(32), kernel_size=3, padding=1)
        self.conv2d_3 = nn.Conv2d(in_channels=ch(32), out_channels=ch(16), kernel_size=3, padding=1)
        self.conv2d_4 = nn.Conv2d(in_channels=ch(16), out_channels=1, kernel_size=3, padding=1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)            
        x = self.bn_input(x)
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = self.pool3(x)
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(-1, 1, 38, 38)
        x = F.relu(self.conv2d_1(x))
        x = F.relu(self.conv2d_2(x))
        x = F.relu(self.conv2d_3(x))
        x = self.conv2d_4(x)
        x = torch.sigmoid(x)
        x = x.squeeze(1)

        return x