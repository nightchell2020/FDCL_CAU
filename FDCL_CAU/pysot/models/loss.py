import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)

def cross_entropy_loss(pred,label):
    pred = pred.view(-1,1)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5

def binary_cross_entropy_loss(pred,label):
    pred = pred.view(-1)
    label = label.view(-1)
    loss = nn.BCELoss()
    loss_value = loss(pred.to(torch.float32),label.to(torch.float32))
    return loss_value

def mean_square_loss(pred,label):
    # pred = pred.view(-1)
    # label = label.view(-1)
    loss = nn.MSELoss()
    loss_value = loss(pred.to(torch.float32), label.to(torch.float32))
    return loss_value

class LossComputation():
    def __init__(self,name):
        self.loss_type = name
    def __call__(self,pred,label):
        if self.loss_type=='BCE':
            cls_loss = binary_cross_entropy_loss(pred,label)
        elif self.loss_type=='CE':
            cls_loss = cross_entropy_loss(pred,label)
        elif self.loss_type=='MSE':
            cls_loss = mean_square_loss(pred,label)
        else :
            print("Unexpected Loss Type")
            cls_loss = 0
        return cls_loss

def make_loss_evaluator(name, **kwargs):
    loss_evaluator = LossComputation(name)
    return loss_evaluator