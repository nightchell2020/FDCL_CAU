from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.backbone import get_backbone
from pysot.models.head.head import head_selector, Head
from pysot.models.loss import make_loss_evaluator

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder,self).__init__()

        self.type = "AEcls" #or "Siam"
        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        self.head = head_selector(cfg.HEAD.TYPE)

        self.loss_evaluater = make_loss_evaluator(cfg.LOSS.TYPE, **cfg.LOSS.KWARGS)

    def reference_signal(self, s):
        ref_signal = self.backbone(s)
        self.ref_signal = ref_signal

    def cls (self, s):
        s2 = self.backbone(s)
        predict_score = self.head(self.ref_signal,s2)
        return {
                'score' : predict_score
                }

    def pred_sigmoid(self, cls):
        sigmoid = torch.nn.Sigmoid()
        cls = sigmoid(cls)
        return cls

    def forward(self, data):
        """only in training"""
        if self.type == "Siam" :
            # input signal = [length x 13]
            signal1 = data['signal1'].cuda()#.reshape(1,13,-1)
            signal2 = data['signal2'].cuda()#.reshape(-1,13,1)
            label = data['status'].cuda() #label.shape=[B]
            signal1 = signal1.to(torch.float32)
            signal2 = signal2.to(torch.float32)
            #signal size : [1x13x131072]

            s1 = self.backbone(signal1)
            s2 = self.backbone(signal2)

            prediction = self.head(s1,s2)

            loss = self.loss_evaluater(prediction, label)

            # get loss
            outputs = {}
            outputs['total_loss'] = loss
            return outputs

        elif self.type == "AEcls":
            '''Unsupervised Learning AutoEncoder'''
            signal = data['signal2'].cuda() # signal.shape=[B,C,L]
            signal = signal.to(torch.float32)

            output = self.backbone(signal)

            loss = self.loss_evaluater(output, signal)

            outputs = {}
            outputs['total_loss'] = loss
            return outputs