from random import shuffle
from os import makedirs, path
import json

import numpy as np

from utils.feature_extractor import extract_features

def read_chunk(reader, chunk_size):
    data = {}
    for i in range(chunk_size):
        ret = reader.read_next()
        for k, v in ret.items():
            if k not in data:
                data[k] = []
            data[k].append(v)
    data["header"] = data["header"][0]
    return data

def sort_and_shuffle(data, batch_size):
    """ Sort data by the length and then make batches and shuffle them.
        data is tuple (X1, X2, ..., Xn) all of them have the same length.
        Usually data = (X, y).
    """
    assert len(data) >= 2
    data = list(zip(*data))

    shuffle(data)

    old_size = len(data)
    rem = old_size % batch_size
    head = data[:old_size - rem]
    tail = data[old_size - rem:]
    data = []

    head.sort(key=(lambda x: x[0].shape[0]))

    mas = [head[i: i+batch_size] for i in range(0, len(head), batch_size)]
    shuffle(mas)

    for x in mas:
        data += x
    data += tail

    data = list(zip(*data))
    return data


def create_directory(directory):
    if not path.exists(directory):
        makedirs(directory)


def pad_zeros(arr, min_length=None):
    """
    `arr` is an array of `np.array`s

    The function appends zeros to every `np.array` in `arr`
    to equalize their first axis lenghts.
    """
    dtype = arr[0].dtype
    max_len = max([x.shape[0] for x in arr])
    ret = [np.concatenate([x, np.zeros((max_len - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
           for x in arr]
    if (min_length is not None) and ret[0].shape[0] < min_length:
        ret = [np.concatenate([x, np.zeros((min_length - x.shape[0],) + x.shape[1:], dtype=dtype)], axis=0)
               for x in ret]
    return np.array(ret)

def extract_features_from_rawdata(chunk, header, period, features):
    with open("utils/resources/channel_info.json") as channel_info_file:
        channel_info = json.loads(channel_info_file.read())
    data = [convert_to_dict(X, header, channel_info) for X in chunk]
    return extract_features(data, period, features)


def convert_to_dict(data, header, channel_info):
    """ convert data from readers output in to array of arrays format """
    ret = [[] for i in range(data.shape[1] - 1)]
    for i in range(1, data.shape[1]):
        ret[i-1] = [(t, x) for (t, x) in zip(data[:, 0], data[:, i]) if x != ""]
        channel = header[i]
        if len(channel_info[channel]['possible_values']) != 0:
            ret[i-1] = list(map(lambda x: (x[0], channel_info[channel]['values'][x[1]]), ret[i-1]))
        ret[i-1] = list(map(lambda x: (float(x[0]), float(x[1])), ret[i-1]))
    return ret
