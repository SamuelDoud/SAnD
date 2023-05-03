import os
import string
from comet_ml import Experiment
from mimic3_benchmarks.mimic3benchmark.readers import InHospitalMortalityReader
from mimic3_benchmarks.mimic3models.preprocessing import Discretizer, Normalizer
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from core.model import SAnD
from utils.ihm_utils import load_data
from utils.functions import get_weighted_sampler, get_weights
from utils.trainer import NeuralNetworkClassifier

batch_size = 256
n_heads = 8
factor = 12 # M
num_class = 2
num_layers = 4 # N
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_reader = InHospitalMortalityReader(dataset_dir="mimic3_benchmarks/data/in-hospital-mortality/train",
                              listfile="mimic3_benchmarks/data/in-hospital-mortality/train_listfile.csv")
val_reader = InHospitalMortalityReader(dataset_dir=r"mimic3_benchmarks/data/in-hospital-mortality/train",
                              listfile="mimic3_benchmarks/data/in-hospital-mortality/val_listfile.csv")
test_reader = InHospitalMortalityReader(dataset_dir="mimic3_benchmarks/data/in-hospital-mortality/test",
                              listfile="mimic3_benchmarks/data/in-hospital-mortality/test_listfile.csv")


discretizer = Discretizer(timestep=1.0,
                          store_masks=True,
                          impute_strategy='previous',
                          start_time='zero')

discretizer_header = discretizer.transform(train_reader.read_example(0)["X"])[1].split(',')
cont_channels = [i for (i, x) in enumerate(discretizer_header) if x.find("->") == -1]

normalizer = Normalizer(fields=cont_channels)  # choose here which columns to standardize
normalizer_state = 'train/ihm_ts1.0.input_str-previous.start_time-zero.normalizer'
normalizer.load_params(normalizer_state)


# Read data
train_raw = load_data(train_reader, discretizer, normalizer)
val_raw = load_data(val_reader, discretizer, normalizer)
test_raw = load_data(val_reader, discretizer, normalizer)

N, seq_len, feature_count = train_raw[0].shape

train_ds = TensorDataset(train_raw[0], train_raw[1])
val_ds = TensorDataset(val_raw[0], val_raw[1])
test_ds = TensorDataset(test_raw[0], test_raw[1])

train_loader = DataLoader(train_ds, batch_size=batch_size)#sampler=get_weighted_sampler(y_train))
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size)


clf = NeuralNetworkClassifier(
    SAnD(feature_count, seq_len, n_heads, factor, num_class, num_layers, dropout_rate=0.3),
    nn.CrossEntropyLoss(weight=torch.tensor(get_weights(train_raw[1], level=2), dtype=torch.float32).to(device=device)),
    optim.Adam, optimizer_config={
        "lr": 0.0005, "betas": (0.9, 0.98), "eps": 4e-09, "weight_decay": 5e-4},
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
    validation=True,
    epochs=epochs
)

# evaluating
clf.evaluate(test_loader)

# save
clf.save_to_file("./save_params/")
