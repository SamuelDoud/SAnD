import threading
import random
import os

import numpy as np
import torch

from utils import common_utils


def save_results(names, ts, predictions, labels, path):
    n_tasks = 25
    common_utils.create_directory(os.path.dirname(path))
    with open(path, 'w') as f:
        header = ["stay", "period_length"]
        header += ["pred_{}".format(x) for x in range(1, n_tasks + 1)]
        header += ["label_{}".format(x) for x in range(1, n_tasks + 1)]
        header = ",".join(header)
        f.write(header + '\n')
        for name, t, pred, y in zip(names, ts, predictions, labels):
            line = [name]
            line += ["{:.6f}".format(t)]
            line += ["{:.6f}".format(a) for a in pred]
            line += [str(a) for a in y]
            line = ",".join(line)
            f.write(line + '\n')
            
def read_and_extract_features(reader, period, features):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    # ret = common_utils.read_chunk(reader, 100)
    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    return (X, ret['y'], ret['name'])

def load_data(reader, discretizer, normalizer, max_seq_len, small_part=False, return_names=False):
    N = reader.get_number_of_examples()
    if small_part:
        N = 1000
    ret = common_utils.read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    for i in range(len(data)):
        p = max(0, max_seq_len - data[i].shape[0])
        data[i] = np.pad(data[i][:max_seq_len], ((0,p), (0,0)))
    whole_data = (torch.tensor(np.array(data)), torch.tensor(labels))
    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}