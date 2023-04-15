from comet_ml import Experiment
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler

from core.model import SAnD
from data.mimiciii import get_mortality_dataset
from utils.trainer import NeuralNetworkClassifier
from pyhealth.tokenizer import Tokenizer

"""
In Hospital Mortality: Mortality prediction is vital during rapid triage and risk/severity assessment. In Hospital
Mortality is defined as the outcome of whether a patient dies
during the period of hospital admission or lives to be discharged. This problem is posed as a binary classification one
where each data sample spans a 24-hour time window. True
mortality labels were curated by comparing date of death
(DOD) with hospital admission and discharge times. The
mortality rate within the benchmark cohort is only 13%.
"""
dataset = get_mortality_dataset()


def split_data(data, train: float, val: float, test: float):
    err = 1e-5
    if 1 - err < (train + val + test) < 1 + err == False:
        raise Exception(
            f"{train=} + {val=} + {test=} = {train+val+test}. Needs to be 1."
        )
    length = len(data)
    end_train = int(len(data) * train)
    end_val = int(len(data) * val) + end_train

    return data[:end_train], data[end_train:end_val], data[end_val:]


tokenizers = {}


def tokenizer_helper(sample, key: str) -> np.array:
    if key not in tokenizers:
        alls = {s for l in [sample[key][0] for sample in dataset.samples] for s in l}
        tokenizers[key] = Tokenizer(list(alls))
    tokenizer = tokenizers[key]
    items = sample[key][0]
    item_table = np.zeros(shape=(tokenizer.get_vocabulary_size()))
    item_indicies = tokenizer.convert_tokens_to_indices(items)
    item_table[item_indicies] = True
    return item_table


# warm up the tokenizer
sample = dataset.samples[0]

(
    tokenizer_helper(sample, "drugs"),
    tokenizer_helper(sample, "conditions"),
    tokenizer_helper(sample, "procedures"),
)
n_tokens = sum(v.get_vocabulary_size() for v in tokenizers.values())

in_feature = n_tokens
n_heads = 32
factor = 32
num_class = 2
num_layers = 6
batch_size = 128
val_ratio = 0.2
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

patients = len(dataset.patient_to_index)
seq_len = max(len(v) for v in dataset.patient_to_index.values())
new_dataset = torch.zeros(
    (
        patients,
        seq_len,
        n_tokens,
    )
)
new_labels = torch.zeros((patients,))

i = 0
for p_id, visits in dataset.patient_to_index.items():
    for n_visit, sample_idx in enumerate(visits):
        sample = dataset.samples[sample_idx]
        sample_data = np.concatenate(
            (
                tokenizer_helper(sample, "drugs"),
                tokenizer_helper(sample, "conditions"),
                tokenizer_helper(sample, "procedures"),
            )
        )
        new_dataset[i][n_visit] = torch.tensor(sample_data)
        new_labels[i] = max(new_labels[i], sample["label"])
    i += 1

train_data, _, test_data = split_data(new_dataset, 0.9, 0.0, 0.1)
train_labels, _, test_labels = split_data(new_labels, 0.9, 0.0, 0.1)

both_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

clf = NeuralNetworkClassifier(
    SAnD(in_feature, seq_len, n_heads, factor, num_class, num_layers),
    nn.CrossEntropyLoss(),
    optim.Adam,
    optimizer_config={
        "lr": 1e-5,
        "betas": (0.9, 0.98),
        "eps": 4e-09,
        "weight_decay": 5e-4,
    },
    experiment=Experiment(
        api_key="eQ3INeSsFGUYKahSdEtjhry42", project_name="general", workspace="samdoud"
    ),
)

# training network
k_rotations = 20
epochs_per = 3
x = []
for k in range(k_rotations):
    train_dataset_size = len(both_dataset)
    indices = list(range(train_dataset_size))
    split = int(np.floor(val_ratio * train_dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        both_dataset, batch_size=batch_size, sampler=train_sampler
    )
    val_loader = DataLoader(both_dataset, batch_size=batch_size, sampler=val_sampler)

    clf.fit(
        {"train": train_loader, "val": val_loader},
        epochs=epochs_per,
        start_epoch=k * epochs_per,
        total_epochs=k_rotations * epochs_per,
        train_dataset_size=train_dataset_size - split,
    )

# evaluating
clf.evaluate(test_loader)

# save
clf.save_to_file("./save_params/")
