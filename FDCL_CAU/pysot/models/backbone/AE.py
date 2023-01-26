import torch.nn as nn
import torch
import torch.nn.functional as F

class BasicEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(BasicEncoderBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool1d(2,stride=2)

    def forward(self,x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pool(out)
        return out

class BasicDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0):
        super(BasicDecoderBlock, self).__init__()
        self.tconv = nn.ConvTranspose1d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self,x):
        out = self.tconv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class ClassificationBlock(nn.Module):
    def __init__(self, in_channels, num_class):
        super(ClassificationBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels,num_class),
            nn.Dropout(0.2),
            nn.Sigmoid()
        )
    def forward(self,x): #x.shape=[Batch,Length,Channel]
        x = x.permute(0,2,1)
        out = self.block(x)
        return out

class AutoEncoder(nn.Module):
    def __init__(self, in_channel=131072, use_feauture=False):
        super(AutoEncoder,self).__init__()
        self.name = 'AE'
        self.use_feauture = use_feauture
        self.encoder_layer0 = nn.BatchNorm1d(in_channel)
        self.encoder_layer1 = BasicEncoderBlock(in_channel, 1024)
        self.encoder_layer2 = BasicEncoderBlock(1024, 100)
        self.decoder_layer1 = BasicDecoderBlock(100, 1024)
        self.decoder_layer2 = BasicDecoderBlock(1024, in_channel,kernel_size=3)
        # self.classification_layer = ClassificationBlock(in_channel,2)

    def forward(self,x):
        x = x.permute(0,2,1)
        f = self.encoder_layer0(x)
        e1_f = self.encoder_layer1(f)
        e2_f = self.encoder_layer2(e1_f)
        d1_f = self.decoder_layer1(e2_f)
        d2_f = self.decoder_layer2(d1_f)
        # features = (f,e1_f,e2_f,d1_f,d2_f)
        d2_f = d2_f.permute(0,2,1)
        # out = self.classification_layer(d2_f)
        return d2_f #out