import torch
import torch.nn as nn
import torch.nn.functional as F

class cnn3d(nn.Module):
    def __init__(self):
        super(cnn3d, self).__init__()
        self.cnv1 = self._def_cnv_layer(1, 16)
        self.cnv1_norm = nn.BatchNorm3d(16)
        self.cnv2 = self._def_cnv_layer(16, 32)
        self.cnv2_norm = nn.BatchNorm3d(32)
        self.cnv3 = self._def_cnv_layer(32, 64)
        self.cnv3_norm = nn.BatchNorm3d(64)
        self.full_conn1 = nn.Linear(83200, 256)
        self.fc1_norm = nn.BatchNorm1d(256)
        self.full_conn2 = nn.Linear(256, 6)
        self.relu = nn.ReLU()
        self.drop1 = nn.Dropout3d(0.25)
        self.drop2 = nn.Dropout(0.5)

    def _def_cnv_layer(self, i_c, o_c):
        cnv_layer = nn.Sequential(
            nn.Conv3d(
                i_c, 
                o_c,
                kernel_size=(5, 5, 5), 
                stride=1,
                padding=0,
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
        x = self.drop1(x)
        x = torch.flatten(x,1)
        x = self.full_conn1(x)
        x = self.relu(x)
        x = self.drop2(x)
        x = self.fc1_norm(x)

        x = self.full_conn2(x)

        return F.softmax(x)