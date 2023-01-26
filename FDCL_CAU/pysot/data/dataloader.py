from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import json
import sys
import scipy.io
import numpy as np
import logging
from pysot.core.config import cfg
from torch.utils.data import Dataset

logger = logging.getLogger("global")


class Subdataset(object):
    def __init__(self,name, root1, root2, anno1, anno2, num_use, start_idx):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.name = name
        self.root1 = root1
        self.root2 = root2
        self.anno1 = anno1
        self.anno2 = anno2
        self.num_use = num_use
        self.start_idx = start_idx
        logger.info("loading" + name)
        with open(self.anno1, 'r') as f:
            meta_data1 = json.load(f)
        with open(self.anno2, 'r') as k:
            meta_data2 = json.load(k)
        self.labels1 = meta_data1
        self.labels2 = meta_data2
        self.num1 = len(self.labels1)
        self.num2 = len(self.labels2)
        self.num_use = self.num1 if self.num_use == -1 else self.num_use
        self.signals1 = list(meta_data1.keys())
        self.signals2 = list(meta_data2.keys())
        logger.info("{} loaded".format(self.name))
        self.path_format = '{}.mat'
        self.pick = self.shuffle()

    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num1, self.path_format))

    def shuffle(self):
        lists1 = list(range(self.start_idx, self.start_idx + self.num1))
        pick1 = []
        lists2 = list(range(self.start_idx, self.start_idx + self.num2))
        pick2 = []
        while len(pick1) < self.num_use:
            np.random.shuffle(lists1)
            pick1 += lists1
        while len(pick2) < self.num_use:
            np.random.shuffle(lists2)
            pick2 += lists2
        return pick1[:self.num_use], pick2[:self.num_use]

    # def get_signal_anno(self, parts, date): #파츠별 mat파일일 때
    #     date = "{:14d}".format(int(date))
    #     signal_path = os.path.join(self.root,
    #                               self.path_format.format(parts))
    #     signal_anno = self.labels[parts][date]
    #     return signal_path, signal_anno

    def get_signal1_anno(self, date, root): #날짜별 mat파일일 때
        date = "{:14d}".format(int(date))

        signal_path = os.path.join(root,
                                  self.path_format.format(date))
        signal_anno = self.labels1[date]

        return signal_path, signal_anno

    def get_signal2_anno(self, date, root): #날짜별 mat파일일 때
        date = "{:14d}".format(int(date))

        signal_path = os.path.join(root,
                                  self.path_format.format(date))
        signal_anno = self.labels2[date]

        return signal_path, signal_anno

    def get_positive_pair(self, index1, index2):
        date1 = self.signals1[index1]
        # date1 = np.random.choice(list(dates1.keys()))
        date2 = self.signals2[index2]
        # date2 = np.random.choice(list(dates2.keys()))
        root1 = self.root1
        root2 = self.root2

        return self.get_signal1_anno(date1,root1), \
            self.get_signal2_anno(date2,root2)

    # def get_random_target(self, index=-1):
    #     if index == -1:
    #         index = np.random.randint(0, self.num)
    #     dates = self.signals1[index]
    #     date = np.random.choice(list(dates.keys()))
    #     root = self.root1
    #     return self.get_signal_anno(date,root)

    def __len__(self):
        return self.num1


class ClsDataset(Dataset):
    def __init__(self,):
        super(ClsDataset,self).__init__()

        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0

        for filename in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, filename)
            sub_dataset = Subdataset(
                filename,
                subdata_cfg.ROOT1,
                subdata_cfg.ROOT2,
                subdata_cfg.ANNO1,
                subdata_cfg.ANNO2,
                subdata_cfg.NUM_USE,
                start
            )
            start +=sub_dataset.num1
            self.num += sub_dataset.num_use

            sub_dataset.log()
            self.all_dataset.append(sub_dataset)
        self.pick = self.shuffle()

    def shuffle(self):
        pick1 = []
        pick2 = []
        m = 0

        while m < self.num:
            p1 = []
            p2 = []
            for sub_dataset in self.all_dataset:
                sub_p1,sub_p2 = sub_dataset.pick
                p1 += sub_p1
                p2 += sub_p2
            np.random.shuffle(p1)
            np.random.shuffle(p2)
            pick1 += p1
            pick2 += p2
            m = len(pick1)

        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick1[:self.num], pick2[:self.num]



    def _find_dataset1(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num1 > index:
                return dataset, index - dataset.start_idx

    def _find_dataset2(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num2 > index:
                return dataset, index - dataset.start_idx

    def _get_status(self, signal):
        status = signal[0]
        return status

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        index1 = self.pick[0][index]
        index2 = self.pick[1][index]
        dataset1, index1 = self._find_dataset1(index1)
        dataset2, index2 = self._find_dataset2(index2)

        # get signal
        sig_left ,sig_right = dataset1.get_positive_pair(index1,index2)
        signal_l = scipy.io.loadmat(sig_left[0])
        signal_r = scipy.io.loadmat(sig_right[0])

        # get fault
        status1 = int(self._get_status(sig_left[1]))
        status2 = int(self._get_status(sig_right[1]))
        if status1 == status2 :
            status = 0
        else :
            status = 1




        return {
                'signal1' : signal_l['sig'],
                'signal2' : signal_r['sig'],
                'status' : status,
                'status1' : status1,
                'status2' : status2
        }





        # for filename in glob.glob('./data/*.mat'):  # mat 파일 불러오기
        #     mat = scipy.io.loadmat(filename)
        #     mat = np.ravel(mat['vibration'][:], order='C')  # 진동 데이터 1-d array 변환
        #     sign = np.append(sign, mat)