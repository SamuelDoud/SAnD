from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from core.model import SAnD
from utils.trainer import NeuralNetworkClassifier

x_train = torch.randn(512, 256, 23)  # [N, seq_len, features]
x_val = torch.randn(128, 256, 23)  # [N, seq_len, features]
x_test = torch.randn(512, 256, 23)  # [N, seq_len, features]

y_train = torch.randint(0, 9, (512,))
y_val = torch.randint(0, 9, (128,))
y_test = torch.randint(0, 9, (512,))

train_ds = TensorDataset(x_train, y_train)
val_ds = TensorDataset(x_val, y_val)
test_ds = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_ds, batch_size=128)
val_loader = DataLoader(val_ds, batch_size=128)
test_loader = DataLoader(test_ds, batch_size=128)

in_feature = 23
seq_len = 256
n_heads = 32
factor = 32
num_class = 10
num_layers = 6

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
