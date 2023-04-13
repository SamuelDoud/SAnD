import time
from pyhealth.datasets import MIMIC3Dataset
from typing import *
from pyhealth.tasks import mortality_prediction_mimic3_fn
from torch.utils.data import Dataset

from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from core.model import SAnD
from data.mimiciii import get_mimic_iii
from utils.trainer import NeuralNetworkClassifier


def get_features_and_label(new_dataset):
    patient_procedure: Dict[str, List[str]] = dict()
    patient_label: Dict[str, int] = dict()

    for i in range(len(new_dataset.samples)):
        sample = new_dataset.samples[i]
        patient_id = sample['patient_id']
        visit_id = sample['visit_id']
        procedures = sample['procedures']
        if patient_id not in patient_procedure:
            patient_procedure[patient_id] = []
        if patient_id not in patient_label:
            patient_label[patient_id] = None
        patient_procedure[patient_id].append(procedures[0])
        patient_label[patient_id] = sample['label']

    patients = sorted(list(patient_label.keys()))
    label = [patient_label[p] for p in patients]
    procedures = [patient_procedure[p] for p in patients]

    return procedures, label


def get_freq_code_and_code2idx(procedures):
    freq_codes = []

    '''
    Append all codes that appears more than 50 times in freq_codes list.
    '''

    seqs = procedures

    cnt_dict = {}
    for i in range(len(seqs)):
        for j in range(len(seqs[i])):
            for each_code in seqs[i][j]:
                if each_code not in cnt_dict:
                    cnt_dict[each_code] = 1
                else:
                    cnt_dict[each_code] += 1

    for each_code in cnt_dict:
        if cnt_dict[each_code] >= 100:
            freq_codes.append(each_code)

    print(freq_codes)
    print(len(freq_codes))

    code2idx = {code: idx for idx, code in enumerate(freq_codes)}
    print(code2idx)

    return freq_codes, code2idx


class CustomDataset(Dataset):

    def __init__(self, seqs, hfs):
        """
        TODO: Store `seqs`. to `self.x` and `hfs` to `self.y`.

        Note that you DO NOT need to covert them to tensor as we will do this later.
        Do NOT permute the data.
        """
        self.x = seqs
        self.y = hfs

    def __len__(self):
        """
        TODO: Return the number of samples (i.e. patients).
        """

        return len(self.x)

    def __getitem__(self, index):
        """
        TODO: Generates one sample of data.

        Note that you DO NOT need to covert them to tensor as we will do this later.
        """

        return self.x[index], self.y[index]


def collate_fn(data):
    sequences, labels = zip(*data)
    y = torch.tensor(labels, dtype=torch.long)
    num_patients = len(sequences)
    num_visits = [len(patient) for patient in sequences]
    max_num_visits = max(num_visits)

    x = torch.zeros((num_patients, max_num_visits, len(freq_codes)), dtype=torch.float)
    for i_patient, patient in enumerate(sequences):
        for j_visit, visit in enumerate(patient):
            for code in visit:
                """
                TODO: 1. check if code is in freq_codes;
                      2. obtain the code index using code2idx;
                      3. set the correspoindg element in x to 1.
                """
                if code in freq_codes:
                    x[i_patient, j_visit, code2idx[code]] = 1

                y[i_patient] = labels[i_patient]

    masks = torch.sum(x, dim=-1) > 0

    return x, masks, y


def split_data(data, train: float, val: float, test: float):
    err = 1e-5
    if 1 - err < (train + val + test) < 1 + err == False:
        raise Exception(f"{train=} + {val=} + {test=} = {train+val+test}. Needs to be 1.")
    length = len(data)
    end_train = int(len(data) * train)
    end_val = int(len(data) * val) + end_train

    return data[:end_train], data[end_train:end_val], data[end_val:]


dataset = get_mimic_iii()
new_dataset = dataset.set_task(mortality_prediction_mimic3_fn)

procedures, label = get_features_and_label(new_dataset)
freq_codes, code2idx = get_freq_code_and_code2idx(procedures)

dataset = CustomDataset(procedures, label)

loader = DataLoader(dataset, batch_size=100000, collate_fn=collate_fn)
loader_iter = iter(loader)
x, masks, y = next(loader_iter)

x_train, x_val, x_test = split_data(x, 0.8, 0.1, 0.1)
y_train, y_val, y_test = split_data(y, 0.8, 0.1, 0.1)

train_ds = TensorDataset(x_train, y_train)
val_ds = TensorDataset(x_val, y_val)
test_ds = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_ds, batch_size=128)
val_loader = DataLoader(val_ds, batch_size=128)
test_loader = DataLoader(test_ds, batch_size=128)

in_feature = 73
seq_len = 28
n_heads = 2
factor = 32
num_class = 2
num_layers = 6

clf = NeuralNetworkClassifier(
    SAnD(in_feature, seq_len, n_heads, factor, num_class, num_layers),
    nn.CrossEntropyLoss(),
    optim.Adam, optimizer_config={"lr": 1e-5, "betas": (0.9, 0.98), "eps": 4e-09, "weight_decay": 5e-4},
    experiment=Experiment(f"{time.time()}", project_name="sand_demo_testing")
    # Note: This is Zhentao's personal key but OK for sharing here.
)

# training network
clf.fit(
    {
        "train": train_loader,
        "val": val_loader
    },
    epochs=1
)

# evaluating
clf.evaluate(test_loader)

# save
clf.save_to_file("save_params/")
