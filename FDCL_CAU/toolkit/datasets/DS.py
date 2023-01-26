import glob
import json
import os

from tqdm import tqdm
import scipy.io
from .Dataset import Dataset
from .Signal import Signal

class DSSignal(Signal):
    """
    Args:
        date: video name
        root: dataset root
        signal_dir: signal directory
        sig_names: signal names(dates)
        gt_stat: groundtruth status
        attr: attribute of video
    """
    def __init__(self, date, root, gt_stat, load_sig=False):
        super(DSSignal, self).__init__(date, root, gt_stat, load_sig)
        # if not load_sig:
        #     sig_name = os.path.join(root,self.signal_names)
        #     sig = scipy.io.loadmat(sig_name)

    # def load_tracker(self, path, tracker_names=None):
    #     """
    #     Args:
    #         path(str): path to result
    #         tracker_name(list): name of tracker
    #     """
    #     if not tracker_names:
    #         tracker_names = [x.split('/')[-1] for x in glob(path)
    #                 if os.path.isdir(x)]
    #     if isinstance(tracker_names, str):
    #         tracker_names = [tracker_names]
    #     # self.pred_trajs = {}
    #     for name in tracker_names:
    #         traj_file = os.path.join(path, name, self.name+'.txt')
    #         if os.path.exists(traj_file):
    #             with open(traj_file, 'r') as f :
    #                 self.pred_trajs[name] = [list(map(float, x.strip().split(',')))
    #                         for x in f.readlines()]
    #             if len(self.pred_trajs[name]) != len(self.gt_traj):
    #                 print(name, len(self.pred_trajs[name]), len(self.gt_traj), self.name)
    #         else:
    #
    #     self.tracker_names = list(self.pred_trajs.keys())

class DSDataset(Dataset):
    """
    Args:
        name:  dataset name
        dataset_root, dataset root dir
    """
    def __init__(self, name, dataset_root, load_sig=False):
        super(DSDataset, self).__init__(name, dataset_root)
        with open(os.path.join(dataset_root, name+'.json'), 'r') as f:
            meta_data = json.load(f)

        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)
        self.test_signals = {}
        self.ref_signals = {}
        self.ref_signals = scipy.io.loadmat("D:/DSdata/test/ref/20180101033506.mat") # A:20180101005559 B:
        # self.ref_signals['20180101005559'] = DSSignal(date='20180101005559', root = 'D:/DSdata/train/A/normal', gt_stat=0, load_sig=False)
        for date in pbar:
            pbar.set_postfix_str(date)
            self.test_signals[date] = DSSignal(date=date,
                                               root=dataset_root,
                                               gt_stat=meta_data[date],
                                               load_sig=load_sig)
        self.attr = {}
        self.attr['ALL'] = list(self.test_signals.keys())
