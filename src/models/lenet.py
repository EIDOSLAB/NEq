#  Copyright (c) 2021 EIDOSLab. All rights reserved.
#  This file is part of the EIDOSearch library.
#  See the LICENSE file for licensing terms (BSD-style).

import torch.nn.functional as F
from torch import nn


class LeNet300(nn.Module):
    def __init__(self, in_features=784, n_classes=10):
        super(LeNet300, self).__init__()
        self.fc1 = nn.Linear(in_features, 300, bias=True)
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(300, 100, bias=True)
        self.r2 = nn.ReLU()
        self.fc3 = nn.Linear(100, n_classes, bias=True)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.r1(x)
        x = self.fc2(x)
        x = self.r2(x)
        x = self.fc3(x)
        return x


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, bias=True)
        self.r1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, bias=True)
        self.r2 = nn.ReLU()
        self.fc1 = nn.Linear(4 * 4 * 50, 500, bias=True)
        self.r3 = nn.ReLU()
        self.fc2 = nn.Linear(500, 10, bias=True)
    
    def forward(self, img):
        output = self.conv1(img)
        output = self.r1(output)
        output = F.max_pool2d(output, 2)
        output = self.conv2(output)
        output = self.r2(output)
        output = F.max_pool2d(output, 2)
        output = output.view(img.size(0), -1)
        output = self.fc1(output)
        output = self.r3(output)
        output = self.fc2(output)
        
        return output
