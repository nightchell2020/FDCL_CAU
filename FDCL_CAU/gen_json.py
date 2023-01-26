from __future__ import unicode_literals

import os
import scipy.io

import json
from collections import OrderedDict


dataset_path = 'D:/DSdata/test/B'#'D:/DSdata/train/B/normal' #'D:/DSdata/A/data'
# dataset_path = 'D:/DSdata/test/A'


# def parse_and_sched(dl_dir='.'):
#     js = {}
#     sensors = os.listdir(dataset_path)
#     for sensor in sensors:
#         data = scipy.io.loadmat(os.path.join(dataset_path, sensor))
#         sensor = sensor.split('.')[0]
#         for idx in range(data['Date'].size):
#             date = '%14d' % (data['Date'][idx])
#             status = '%1d' % (data['label'][idx])
#
#             if sensor not in js:
#                 js[sensor] = {}
#             if date not in js[sensor]:
#                 js[sensor][date] = {}
#             js[sensor][date] = status
#         json.dump(js, open('train.json', 'w'), indent=4, sort_keys=True)
#     print('All signals downloaded')

def parse_and_sched(dl_dir='.'):
    js = {}
    dates = os.listdir(dataset_path)
    for date in dates:
        data = scipy.io.loadmat(os.path.join(dataset_path, date))
        date = date.split('.')[0]
        status = '%1d' % (data['label'])

        if date not in js:
            js[date] = {}
        js[date] = status
    json.dump(js,open('B.json','w'), indent=4, sort_keys=True)

    print('All signals downloaded')
if __name__ == '__main__':
    parse_and_sched()





