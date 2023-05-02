import string
from comet_ml import Experiment
import numpy as np
from mimic3_benchmarks.mimic3benchmark.readers import InHospitalMortalityReader

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from core.model import SAnD
from utils.functions import get_weighted_sampler, get_weights
from utils.trainer import NeuralNetworkClassifier

batch_size = 8

train_reader = InHospitalMortalityReader(dataset_dir=r"C:\Users\Samuel\Desktop\SAnD\mimic3_benchmarks\data\in-hospital-mortality\train",
                              listfile=r"C:\Users\Samuel\Desktop\SAnD\mimic3_benchmarks\data\in-hospital-mortality\train_listfile.csv")
val_reader = InHospitalMortalityReader(dataset_dir=r"C:\Users\Samuel\Desktop\SAnD\mimic3_benchmarks\data\in-hospital-mortality\train",
                              listfile=r"C:\Users\Samuel\Desktop\SAnD\mimic3_benchmarks\data\in-hospital-mortality\val_listfile.csv")
test_reader = InHospitalMortalityReader(dataset_dir=r"C:\Users\Samuel\Desktop\SAnD\mimic3_benchmarks\data\in-hospital-mortality\test",
                              listfile=r"C:\Users\Samuel\Desktop\SAnD\mimic3_benchmarks\data\in-hospital-mortality\test_listfile.csv")
train_reader_count = train_reader.get_number_of_examples()
test_reader_count = test_reader.get_number_of_examples()
val_reader_count = val_reader.get_number_of_examples()
feature_count = len(train_reader.read_example(0)['header'])

print(len(train_reader.read_example(1)['X']))
#t = max(len(test_reader.read_example(x)['X']) for x in range(test_reader_count))
#tr = max(len(train_reader.read_example(x)['X']) for x in range(train_reader_count))
#v =max(len(val_reader.read_example(x)['X']) for x in range(val_reader_count))
seq_len = 2879
x_train = torch.zeros((train_reader_count, seq_len, feature_count))  # [N, seq_len, features]
x_val = torch.zeros((val_reader_count, seq_len, feature_count))  # [N, seq_len, features]
x_test = torch.zeros((test_reader_count, seq_len, feature_count))  # [N, seq_len, features]

y_train = torch.zeros((train_reader_count,), dtype=torch.int)
y_val = torch.zeros((val_reader_count,), dtype=torch.int)
y_test = torch.zeros((test_reader_count,), dtype=torch.int)

def to_float(x: str) -> float:
    return float(x.split(' ')[0]) if x and x[0].isnumeric() else 0
to_float_vec = np.vectorize(to_float)
for i in range(train_reader_count):
    ex = train_reader.read_example(i)
    x = torch.tensor(to_float_vec(ex['X']))
    x_train[i,:x.shape[0]] = x
    y_train[i] = torch.tensor(ex['y'])
for i in range(val_reader_count):
    ex = val_reader.read_example(i)
    x = torch.tensor(to_float_vec(ex['X']))
    x_val[i,:x.shape[0]] = x
    y_val[i] = torch.tensor(ex['y'])
for i in range(test_reader_count):
    ex = test_reader.read_example(i)
    x = torch.tensor(to_float_vec(ex['X']))
    x_test[i,:x.shape[0]] = x
    y_test[i] = torch.tensor(ex['y'])

train_ds = TensorDataset(x_train, y_train)
val_ds = TensorDataset(x_val, y_val)
test_ds = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_ds, batch_size=batch_size)
val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=batch_size*2)

n_heads = 4
factor = 12
num_class = 2
num_layers = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clf = NeuralNetworkClassifier(
    SAnD(feature_count, seq_len, n_heads, factor, num_class, num_layers),
    nn.CrossEntropyLoss(weight=torch.tensor(get_weights(y_train), dtype=torch.float32).to(device=device)),
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
    validation=True,
    epochs=10
)

# evaluating
clf.evaluate(test_loader)

# save
clf.save_to_file("./save_params/")
