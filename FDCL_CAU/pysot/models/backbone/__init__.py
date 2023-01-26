from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.backbone.ResNet1D import MSResNet
from pysot.models.backbone.WDCNN import WDCNN
from pysot.models.backbone.AE import AutoEncoder


BACKBONES = {'ResNet': MSResNet,
            'WDCNN': WDCNN,
            'AutoEncoder' : AutoEncoder
            }


def get_backbone(name, **kwargs):
    return BACKBONES[name](**kwargs)
