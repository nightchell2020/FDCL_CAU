import torch.nn as nn
import torch
import torch.nn.functional as F

class Basicblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Basicblock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(2,stride=2)

    def forward(self,x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.pool(out)
        out = F.relu(out)
        return out

class WDCNN(nn.Module):
    def __init__(self, in_channels=13, n_class=10000, use_feauture=False):
        super(WDCNN,self).__init__()
        self.name = 'WDCNN'
        self.use_feature=use_feauture
        self.layer0 = nn.BatchNorm1d(in_channels)
        self.layer1 = Basicblock(in_channels, 16,
                                 kernel_size=64,
                                 stride=16,
                                 padding=24)
        self.layer2 = Basicblock(16,32)
        self.layer3 = Basicblock(32,64)
        self.layer4 = Basicblock(64,64)
        self.layer5 = Basicblock(64,64,padding=0)
        self.n_features = 64*255
        self.fc = nn.Linear(self.n_features, n_class)
        self.fc2 = nn.Linear(n_class, 512)



    def forward(self,x):
        f0 = self.layer0(x)
        f1 = self.layer1(f0)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        f5 = self.layer5(f4)
        features = (f0,f1,f2,f3,f4,f5)
        activations = self.fc(features[-1].view(-1,self.n_features))
        activations2 = self.fc2(activations)
        out = activations2
        return out