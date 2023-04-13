from comet_ml import Experiment
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from core.model import SAnD
from data.mimiciii import get_mimic_iii
from utils.trainer import NeuralNetworkClassifier
from pyhealth.datasets import split_by_patient
from pyhealth.tasks import mortality_prediction_mimic3_fn
from pyhealth.tokenizer import Tokenizer

def collate_fn_dict(batch):
    _, _, _, _, _, labels, seq = zip(*batch)

    y = torch.tensor(labels, dtype=torch.float)
    
    num_patients = len(set(patient_id))
    num_codes = [len(visit) for patient in sequences for visit in patient]

    max_num_visits = max
    max_num_codes = max(num_codes)
    
    x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype=torch.long)
    for i_patient, patient in enumerate(sequences):
        for j_visit, visit in enumerate(patient):
            # your code here
            diff = max_num_codes - len(visit)
            pad = (0, diff)
            padded_visit = torch.tensor(np.pad(visit, pad, "constant"))
            mask = torch.cat((torch.ones(len(visit)), torch.zeros(diff)))
            x[i_patient][j_visit] = padded_visit
    return x

def get_dataloader(dataset, batch_size, shuffle=False):
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_dict
    )
    return dataloader

"""
In Hospital Mortality: Mortality prediction is vital during rapid triage and risk/severity assessment. In Hospital
Mortality is defined as the outcome of whether a patient dies
during the period of hospital admission or lives to be discharged. This problem is posed as a binary classification one
where each data sample spans a 24-hour time window. True
mortality labels were curated by comparing date of death
(DOD) with hospital admission and discharge times. The
mortality rate within the benchmark cohort is only 13%.
"""
in_feature = 23
seq_len = 256
n_heads = 32
factor = 32
num_class = 2
num_layers = 6

dataset = get_mimic_iii().set_task(mortality_prediction_mimic3_fn)

'''
x_train = torch.randn(512, 256, 23)  # [N, seq_len, features]
x_val = torch.randn(128, 256, 23)  # [N, seq_len, features]
x_test = torch.randn(512, 256, 23)  # [N, seq_len, features]

y_train = torch.randint(0, 9, (512,))
y_val = torch.randint(0, 9, (128,))
y_test = torch.randint(0, 9, (512,))

train_ds = TensorDataset(x_train, y_train)
val_ds = TensorDataset(x_val, y_val)
test_ds = TensorDataset(x_test, y_test)
'''

# data split
train_dataset, val_dataset, test_dataset = split_by_patient(dataset, [0.8, 0.1, 0.1])

tokenizers = {}

def tokenizer_helper(sample, key: str) -> np.array:
    if key not in tokenizers:
        alls =  {s for l in [sample[key][0] for sample in dataset.samples] for s in l}
        tokenizers[key] = Tokenizer(list(alls))
    tokenizer = tokenizers[key]
    items = sample[key][0]
    item_table = np.zeros(shape=(tokenizer.get_vocabulary_size()))
    item_indicies = tokenizer.convert_tokens_to_indices(items)
    item_table[item_indicies] = True
    return item_table

new_samples = {}
for sample in dataset.samples:
    sample_data = np.concatenate((tokenizer_helper(sample, "drugs"), tokenizer_helper(sample, "conditions"), tokenizer_helper(sample, "procedures")))
    if sample["patient_id"] not in new_samples:
        new_samples["patient_id"] = {}
        new_samples["patient_id"]["data"] = []
        new_samples["patient_id"]["label"]

# create dataloaders (they are <torch.data.DataLoader> object)
train_loader = get_dataloader(train_dataset, batch_size=seq_len, shuffle=True)
val_loader = get_dataloader(val_dataset, batch_size=seq_len, shuffle=False)
test_loader = get_dataloader(test_dataset, batch_size=seq_len, shuffle=False)


clf = NeuralNetworkClassifier(
    SAnD(in_feature, seq_len, n_heads, factor, num_class, num_layers),
    nn.CrossEntropyLoss(),
    optim.Adam, optimizer_config={
        "lr": 1e-5, "betas": (0.9, 0.98), "eps": 4e-09, "weight_decay": 5e-4},
    experiment=Experiment(
        api_key="eQ3INeSsFGUYKahSdEtjhry42",
        project_name="general",
        workspace="samdoud"
    )
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
clf.save_to_file("./save_params/")
