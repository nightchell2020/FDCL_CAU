import os
import cv2
import re
import numpy as np
import json
import scipy.io

from glob import glob

class Signal(object):
    def __init__(self, date, root, gt_stat, load_sig=False):
        self.date = date
        self.gt_stat = gt_stat
        self.signal_root = [os.path.join(root,date)]
        self.pred_stat = {}
        self.sig = scipy.io.loadmat(self.signal_root[0])
        self.sigs = None
        # self.sigs=None
        #
        # if load_sig :
        #     self.sigs = scipy.io.loadmat(self.signal_root[0])
        # else :
        #     sig = scipy.io.loadmat(self.signal_root[0])
        #     assert sig is not None, self.signal_root[0]

    def load_classificator(self, path, classificator_names=None, store=True):
        """
        Args:
            path(str): path to result
            tracker_name(list): name of tracker
        """
        if not classificator_names:
            classificator_names = [x.split('/')[-1] for x in glob(path)
                                   if os.path.isdir(x)]
        if isinstance(classificator_names, str):
            classificator_names = [classificator_names]
        for name in classificator_names:
            stat_file = os.path.join(name, 'A' +'.txt')
            if os.path.exists(stat_file):
                with open(stat_file, 'r') as f :
                    pred_stat = [list(map(float, x.strip().split(',')))
                            for x in f.readlines()]

                # if len(pred_stat) != len(self.gt_stat):
                #     print(name, len(pred_stat), len(self.gt_stat), self.date)
                if store:
                    self.pred_stat[name] = pred_stat
                else:
                    return pred_stat
            else:
                print(stat_file)
        self.classificator_names = list(self.pred_stat.keys())


    def __len__(self):
        return len(self.signal_root[0])
    #
    def __getitem__(self, idx):
        # if self.sigs is None:
        #     return scipy.io.loadmat(self.signal_root[0]), self.gt_stat
        # else:
        #     return self.sigs, self.gt_stat
        return self.sig, self.gt_stat
    #
    def __iter__(self):
        # for i in range(len(self.signal_names)):
        # if self.sigs is not None:
        #     yield self.sigs, self.gt_stat
        # else:
        #     yield scipy.io.loadmat(self.signal_root[0]), self.gt_stat
        return self.sig, self.gt_stat
    #
    #
    # def show(self, pred_trajs={}, linewidth=2, show_name=False):
    #     """
    #         pred_trajs: dict of pred_traj, {'tracker_name': list of traj}
    #                     pred_traj should contain polygon or rectangle(x, y, width, height)
    #         linewith: line width of the bbox
    #     """
    #     assert self.imgs is not None
    #     video = []
    #     cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
    #     colors = {}
    #     if len(pred_trajs) == 0 and len(self.pred_stat) > 0:
    #         pred_trajs = self.pred_stat
    #     for i, (roi, img) in enumerate(zip(self.gt_stat,
    #             self.imgs[self.start_frame:self.end_frame+1])):
    #         img = img.copy()
    #         if len(img.shape) == 2:
    #             img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    #         else:
    #             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #         img = self.draw_box(roi, img, linewidth, (0, 255, 0),
    #                 'gt' if show_name else None)
    #         for name, trajs in pred_trajs.items():
    #             if name not in colors:
    #                 color = tuple(np.random.randint(0, 256, 3))
    #                 colors[name] = color
    #             else:
    #                 color = colors[name]
    #             img = self.draw_box(trajs[0][i], img, linewidth, color,
    #                     name if show_name else None)
    #         cv2.putText(img, str(i+self.start_frame), (5, 20),
    #                 cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0), 2)
    #         cv2.imshow(self.name, img)
    #         cv2.waitKey(40)
    #         video.append(img.copy())
    #     return video
