from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from toolkit.datasets import DSDataset
from toolkit.evaluation import Accuracy_benchmark


def accuracy_error(gt_label, pred_label):
    n_signal = len(pred_label)
    correct = 0
    if gt_label == pred_label:
        correct += 1
    else :
        correct = correct
    accuracy = correct / n_signal * 100
    return accuracy
# def main(dataset='A',cls_prefix='',cls_path='./results', num=4):

def evaluation(dataset='B', cls_prefix='', cls_path='./results', num=4):
    cls_dir = os.path.join(cls_path, dataset)
    classificators = glob(os.path.join(cls_path,
                                 dataset,
                                 cls_prefix + '*'))
    classificators = [x.split('\\')[-1] for x in classificators]

    assert len(classificators) > 0
    num = min(num, len(classificators))

    root = os.path.realpath(os.path.join(os.path.dirname(__file__),
                                         'D:/DSdata/test'))
    root = os.path.join(root, dataset)

    dataset = DSDataset(dataset,root)
    dataset.set_classificator(cls_dir, classificators)
    benchmark = Accuracy_benchmark(dataset)

    accuracy_ret = {}
    with Pool(processes=num) as pool:
        for ret in tqdm(pool.imap_unordered(benchmark.eval_accuracy, classificators), desc='eval success'):
            accuracy_ret.update(ret)
    benchmark.show_result(accuracy_ret)

# if __name__ == '__main__':
#     main()
