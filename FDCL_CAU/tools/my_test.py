# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np


from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.Cls.cls_builder import build_classificator
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from my_eval import evaluation


parser = argparse.ArgumentParser(description='DS anomaly detection')
parser.add_argument('--dataset', default='A', type=str,
        help='datasets')
parser.add_argument('--config', default='../experiments/config.yaml', type=str,
        help='config file')
parser.add_argument('--snapshot', default='snapshot/checkpoint_e20.pth', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
# parser.add_argument('--vis', default=False, type=bool,
#         help='whether visualzie result')
parser.add_argument('--gpu_id', default='0', type=str,
        help="gpu id")

args = parser.parse_args()

if args.gpu_id != 'not_set':
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

torch.set_num_threads(1)

def main():
    # load config
    cfg.merge_from_file(args.config)

    dataset_root = os.path.join('D:/DSdata/test', args.dataset)

    for epoch in range(0, 1000):
        snapshot = 'snapshot/checkpoint_e{}.pth'.format(epoch+1)
        # create model
        model = ModelBuilder()

        # load model
        model = load_pretrain(model, snapshot).cuda().eval()

        # build classificator
        classificator = build_classificator(model)

        # create dataset
        dataset = DatasetFactory.create_dataset(name=args.dataset,
                                                dataset_root=dataset_root,
                                                load_sig=False)

        model_name = snapshot.split('/')[-1].split('.')[0]


        toc = 0
        pred_statuses = []
        tic = cv2.getTickCount()
        ref_signals = dataset.ref_signals
        ref_signal = torch.tensor(ref_signals['sig'].reshape((1, 13, 131072)), dtype=torch.float32).cuda()
        classificator.init(ref_signal)

        for idx, test_signals in enumerate(dataset):
            test_signal = torch.tensor(test_signals.sig['sig'].reshape((1,13,131072)),dtype=torch.float32).cuda()
            outputs = classificator.classificate(test_signal)
            pred_status = outputs['score']
            pred_statuses.append(pred_status)

            toc += cv2.getTickCount() - tic
            if idx==0:
                cv2.destroyAllWindows()

        # save results
        signal_path = os.path.join('results', 'B', model_name)
        if not os.path.isdir(signal_path):
            os.makedirs(signal_path)
        result_path = os.path.join(signal_path, '{}.txt'.format(dataset.name))
        with open(result_path, 'w') as f:
            for x in pred_statuses:
                if isinstance(x, int):
                    f.write("{:d}\n".format(x))
                else :
                    x = x.item()
                    f.write("{:f}\n".format(x))

        print('Time : {:4.1f}s Speed: {:3.1f}fps'.format(toc, idx/toc))
        evaluation(args.dataset, model_name)


if __name__ == '__main__':
    main()
