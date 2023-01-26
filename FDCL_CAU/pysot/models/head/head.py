import torch
from torch import nn

class Head(torch.nn.Module):
    def __init__(self):
        super(Head,self).__init__()
        self.fc = nn.Linear(512,1)
    def forward(self,x,y):
        distance =  abs(x - y)
        distance = self.fc(distance)
        prediction = torch.sigmoid(distance)
        return prediction

def L1distance(signal1,signal2):
    distance = abs(signal1 - signal2)
    fc = nn.Linear(512,1)
    distance = fc(distance)
    prediction = torch.sigmoid(distance)
    return prediction

def cossim(x, y):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    prediction = 1-cos(x, y)
    return prediction

def bcls(signal1, signal2):
    layer = nn.Sequential(
        nn.Linear(in_features=131072,out_features=2),
        nn.Dropout(0.2),
        nn.Sigmoid()
    )
    prediction = layer(signal1)
    return prediction

class HeadSelector(object):
    def __init__(self,name):
        self.head_type = name
    def __call__(self, signal1,signal2):
        if self.head_type == 'cossim':
            distance = cossim(signal1,signal2)
            return distance
        elif self.head_type == 'b_cls':
            out = bcls(signal1,signal2)
            return out
        else :
            return print("WrongType")

def head_selector(name):
    head_selector = HeadSelector(name)
    return head_selector