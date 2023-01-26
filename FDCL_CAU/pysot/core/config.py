# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = "DS_AnomalyDet"

__C.CUDA = True

# ------------------------------------------------------------------------ #
# Training options
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()


__C.TRAIN.THR_HIGH = 0.6

__C.TRAIN.apnchannel = 256

__C.TRAIN.clsandlocchannel = 256

__C.TRAIN.groupchannel = 32

__C.TRAIN.THR_LOW = 0.3

__C.TRAIN.NEG_NUM = 16

__C.TRAIN.POS_NUM = 16

__C.TRAIN.TOTAL_NUM = 64

__C.TRAIN.PR = 1

__C.TRAIN.CLS_WEIGHT = 1.0

__C.TRAIN.LOC_WEIGHT = 3.0

__C.TRAIN.SHAPE_WEIGHT =2.0


__C.TRAIN.BASE_SIZE = 8

__C.TRAIN.OUTPUT_SIZE = 21 #25

__C.TRAIN.RESUME = ''

__C.TRAIN.PRETRAINED = 1

__C.TRAIN.LARGER=2.0

__C.TRAIN.LOG_DIR = './logs'

__C.TRAIN.SNAPSHOT_DIR = './snapshot'

__C.TRAIN.EPOCH = 100

__C.TRAIN.START_EPOCH = 0

__C.TRAIN.BATCH_SIZE = 2

__C.TRAIN.NUM_GPU = 2

__C.TRAIN.NUM_WORKERS = 0

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0001

__C.TRAIN.w1=1.0

__C.TRAIN.R1=1.5

__C.TRAIN.R2=1.2

__C.TRAIN.w2=1.0

__C.TRAIN.w3=1.0

__C.TRAIN.range=2.0


__C.TRAIN.MASK_WEIGHT = 1

__C.TRAIN.PRINT_FREQ = 20

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0

__C.TRAIN.BASE_LR = 0.005

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log'

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step'

__C.TRAIN.LR_WARMUP.EPOCH = 5

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# for detail discussion
__C.DATASET.NEG = 0.2

__C.DATASET.GRAY = 0.0

__C.DATASET.NAMES = ('A', 'B', 'C')

__C.DATASET.A = CN()
__C.DATASET.A.ROOT1 = '/home/one/DSdata/train/A/normal' #'D:/DSdata/A/data2/train/normal'
__C.DATASET.A.ROOT2 = '/home/one/DSdata/train/A/pool'
__C.DATASET.A.ANNO1 = '/home/one/DSdata/train/A/normal.json'
__C.DATASET.A.ANNO2 = '/home/one/DSdata/train/A/pool.json'
__C.DATASET.A.NUM_USE = 10000

__C.DATASET.B = CN()
__C.DATASET.B.ROOT1 = '/home/one/DSdata/train/B/normal'
__C.DATASET.B.ROOT2 = '/home/one/DSdata/train/B/pool'
__C.DATASET.B.ANNO1 = '/home/one/DSdata/train/B/normal.json'
__C.DATASET.B.ANNO2 = '/home/one/DSdata/train/B/pool.json'
__C.DATASET.B.NUM_USE = 10000

__C.DATASET.C = CN()
__C.DATASET.C.ROOT = 'D:/DSdata/C'
__C.DATASET.C.ANNO = 'D:/DSdata/C/train.json'
__C.DATASET.C.NUM_USE = 100


# ------------------------------------------------------------------------ #
# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

__C.BACKBONE.TYPE = 'WDCNN' #'ResNet'

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''

# Train layers
__C.BACKBONE.TRAIN_LAYERS = ['layer3', 'layer4', 'layer5']

# Train channel_layer
__C.BACKBONE.CHANNEL_REDUCE_LAYERS = []

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10

# ------------------------------------------------------------------------ #
# Head options
# ------------------------------------------------------------------------ #
__C.HEAD = CN()

__C.HEAD.TYPE = 'Cossim' #'b_cls' 'Cossim'

__C.HEAD.KWARGS = CN(new_allowed=True)

__C.HEAD.W1 = 1.0

# ------------------------------------------------------------------------ #
# Loss options
# ------------------------------------------------------------------------ #
__C.LOSS = CN()

__C.LOSS.TYPE = 'BCE' #'CE'

__C.LOSS.KWARGS = CN(new_allowed=True)

__C.LOSS.W1 = 1.0

# ------------------------------------------------------------------------ #
# Classificator options
# ------------------------------------------------------------------------ #
__C.CLASSIFICATOR = CN()

__C.CLASSIFICATOR.TYPE = 'DSClassificator'

# Scale penalty
__C.CLASSIFICATOR.PENALTY_K = 0.04

# Window influence
__C.CLASSIFICATOR.WINDOW_INFLUENCE = 0.44

# Interpolation learning rate
__C.CLASSIFICATOR.LR = 0.4

__C.CLASSIFICATOR.w1=1.0

__C.CLASSIFICATOR.w2=1.0


__C.CLASSIFICATOR.LARGER=1.4
# Exemplar size
__C.CLASSIFICATOR.EXEMPLAR_SIZE = 127

# Instance size
__C.CLASSIFICATOR.INSTANCE_SIZE = 255

# Base size
__C.CLASSIFICATOR.BASE_SIZE = 8

__C.CLASSIFICATOR.STRIDE = 8

# Context amount
__C.CLASSIFICATOR.CONTEXT_AMOUNT = 0.5

# Long term lost search size
__C.CLASSIFICATOR.LOST_INSTANCE_SIZE = 831

# Long term confidence low
__C.CLASSIFICATOR.CONFIDENCE_LOW = 0.85

# Long term confidence high
__C.CLASSIFICATOR.CONFIDENCE_HIGH = 0.998

# Mask threshold
__C.CLASSIFICATOR.MASK_THERSHOLD = 0.30
