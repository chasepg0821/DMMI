import pandas as pd
import numpy as np
import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *

class cnn3d(nn.Module):
    def __init__(self):
        super(cnn3d, self).__init__()
        self.cnv1 = self._def_cnv_layer(1, 16)
        self.cnv1_norm = nn.BatchNorm3d(16)
        self.cnv2 = self._def_cnv_layer(16, 32)
        self.cnv2_norm = nn.BatchNorm3d(32)
        self.cnv3 = self._def_cnv_layer(32, 64)
        self.cnv3_norm = nn.BatchNorm3d(64)
        self.full_conn1 = nn.Linear(10*10*10*64, 128)
        self.fc1_norm = nn.BatchNorm1d(128)
        self.full_conn2 = nn.Linear(128, 20)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.3)

    def _def_cnv_layer(self, i_c, o_c):
        cnv_layer = nn.Sequential(
            nn.Conv3d(
                i_c, 
                o_c, 
                kernel_size=(4, 4, 4), 
                stride=2,
                padding=2,
                ),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
            )
        return cnv_layer

    def forward(self, x):
        x = self.cnv1(x)
        x = self.cnv1_norm(x)
        x = self.cnv2(x)
        x = self.cnv2_norm(x)
        x = self.cnv3(x)
        x = self.cnv3_norm(x)
        x = x.view(x.size(0), -1)
        x = self.full_conn1(x)
        x = self.relu(x)
        x = self.fc1_norm(x)
        x = self.drop(x)
        x = self.full_conn2(x)

        return x


## CNN with two conv layers, global average pooling, and two dense layers.
class Net(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, 1)
        self.max_pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 16, 5, 1)
        self.glob_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, out_features)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool(x)
        x = F.relu(self.conv2(x))
        x = self.glob_avg_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Model(torch.nn.Module): 
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=3)
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3)
        self.conv4 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(1749600, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool3d(x, kernel_size=2, stride=2)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool3d(x, kernel_size=2, stride=2)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        return x