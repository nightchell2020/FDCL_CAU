from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.core.config import cfg
from pysot.Cls.DS_classificator import DSClassificator

CLS = {
          'DSClassificator': DSClassificator
         }


def build_classificator(model):
    return CLS[cfg.CLASSIFICATOR.TYPE](model)
