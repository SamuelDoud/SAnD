from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from SAnD.core.model import SAnD
from SAnD.utils.trainer import NeuralNetworkClassifier


datase

x_train = torch.randn(1024, 256, 23)    # [N, seq_len, features]
x_val = torch.randn(128, 256, 23)       # [N, seq_len, features]
x_test =  torch.randn(512, 256, 23)     # [N, seq_len, features]

y_train = torch.randint(0, 9, (1024, ))
y_val = torch.randint(0, 9, (128, ))
y_test = torch.randint(0, 9, (512, ))


train_ds = TensorDataset(x_train, y_train)
val_ds = TensorDataset(x_val, y_val)
test_ds = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_ds, batch_size=128)
val_loader = DataLoader(val_ds, batch_size=128)
test_loader = DataLoader(test_ds, batch_size=128)

