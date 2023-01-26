import numpy as np
from toolkit.datasets.Signal import Signal
import os
import torch.nn
import torchvision

dataset_root = os.path.join(os.path.dirname(__file__), '../../tools/results/result.txt')
class Accuracy_benchmark :
    def __init__(self,dataset):
        self.dataset = dataset

    def eval_accuracy(self, eval_classificators=None):
        if eval_classificators is None:
            eval_classificators = self.dataset.cls_names
        if isinstance(eval_classificators, str):
            eval_classificators = [eval_classificators]

        accuracy_ret = {}
        for cls_name in eval_classificators:
            gt_stat_record = []
            cls_stat_record = []


            # cls_stat = Signal.load_classificator(self=self.dataset.test_signals, path=self.dataset.classificator_path, classificator_names=cls_name,store= False)
            # np.transpose(cls_stat)
            aaa = self.dataset.test_signals
            for date in list(aaa.keys()):
                gt_stat = np.array([int(aaa[date].gt_stat)])
                gt_stat_record = np.append(gt_stat_record, gt_stat)
                cls_stat = aaa[date].load_classificator(self.dataset.classificator_path, cls_name, False)
                cls_stat_record = np.array(cls_stat)
            n_frame = len(gt_stat_record)
            accuracy_ret[cls_name] = success_overlap(gt_stat_record, cls_stat_record, n_frame)
        return accuracy_ret

    def show_result(self, accuracy_ret):
        cls_accuracy = {}
        for cls_name in accuracy_ret.keys():
            accuracy = np.mean(list(accuracy_ret[cls_name]))
            cls_accuracy[cls_name] = accuracy

        cls_name_len = max((max([len(x) for x in accuracy_ret.keys()]) + 2), 12)
        header = ("|{:^" + str(cls_name_len) + "}|{:^9}|").format(
            "Classificator name", "Accuracy")
        formatter = "|{:^" + str(cls_name_len) + "}|{:^9.3f}|"
        print('-' * len(header))
        print(header)
        print('-' * len(header))
        for cls_name in accuracy_ret.keys():
            success = cls_accuracy[cls_name]
            print(formatter.format(cls_name, success))
        print('-'*len(header))
        with open(dataset_root, 'a') as f:
            tmp = formatter.format(cls_name, success)
            f.writelines(tmp+'\n')




def distance_ratio(value1, value2):
    '''Compute overlap ratio between two rects
    Args
        value:2d array of N x [d]
    Return:
        distance
    '''
    distance = abs(value1 - value2)
    return distance

def success_overlap(gt_stat, result_stat, n_frame):
    thresholds_overlap = [0.4]#np.arange(0.1, 0.5, 0.05)
    success = np.zeros(len(thresholds_overlap))
    result_stat = result_stat.reshape((len(result_stat),))
    distance= distance_ratio(gt_stat, result_stat)
    for i in range(len(thresholds_overlap)):
        success[i] = np.sum(distance < thresholds_overlap[i]) / float(n_frame)
    return success

