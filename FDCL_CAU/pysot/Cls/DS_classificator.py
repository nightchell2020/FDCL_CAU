from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import torch
import cv2

from pysot.core.config import cfg
from pysot.Cls.base_cls import SiameseTracker

class DSClassificator(SiameseTracker):
    def __init__(self, model):
        super(DSClassificator,self).__init__()
        # self.score_size = cfg.CLASSIFICATOR.SCORE_SIZE
        self.model = model
        self.model.eval()

    def init(self, signal, **kwargs):
        self.model.reference_signal(signal)

    def classificate(self,signal):
        """
        param :
            signal: [length,13]
        return:
            score: [0~1]
        """
        outputs = self.model.cls(signal)

        score = outputs['score']

        return {
            'score':score,
        }