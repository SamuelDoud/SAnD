import logging

from comet_ml import Experiment
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from core.model import SAnD
from data.mimiciii import get_mortality_dataset as mimiciii_mortality
from data.mimiciv import get_mortality_dataset as mimiciv_mortality
from data.tokenizers import Tokenizers

from utils.functions import (
    get_pca,
    pca_transform,
    split_data,
    get_weighted_sampler,
)
from utils.trainer import NeuralNetworkClassifier

"""
In Hospital Mortality: Mortality prediction is vital during rapid triage and risk/severity assessment. In Hospital
Mortality is defined as the outcome of whether a patient dies
during the period of hospital admission or lives to be discharged. This problem is posed as a binary classification one
where each data sample spans a 24-hour time window. True
mortality labels were curated by comparing date of death
(DOD) with hospital admission and discharge times. The
mortality rate within the benchmark cohort is only 13%.
"""
datasets = [mimiciv_mortality]

clamp_seq = 10
max_visits = 100000
for i in range(len(datasets)):
    dataset = datasets[i]()  # don't load this until need, then purge the old one
    tokenizers = Tokenizers(
        ["conditions", "procedures"], dataset=dataset, depth=20
    )
    # warm up the tokenizer

    n_tokens = tokenizers.vocabulary_size()

    n_visits = min(len(dataset.visit_to_index), max_visits)
    seq_len = min(max(len(v) for v in dataset.patient_to_index.values()), clamp_seq)
    new_dataset = torch.zeros(
        (
            n_visits,
            seq_len,
            n_tokens,
        ),
        dtype=torch.bool,
    )


    n_heads = 4
    factor = 12
    num_class = 2
    num_layers = 6
    batch_size = 256
    val_ratio = 0.2
    seed = 1234
    pca_dim = n_tokens
    n_features = pca_dim
    in_feature = n_features
    torch.manual_seed(seed)
    np.random.seed(seed)

    new_labels = torch.zeros((n_visits,), dtype=torch.bool)

    for p_id, visits in dataset.patient_to_index.items():
        for n_visit, sample_idx in enumerate(visits):
            if sample_idx >= n_visits or n_visit >= seq_len:
                continue
            sample = dataset.samples[sample_idx]
            sample_data = tokenizers.to_data(sample=sample)

            if n_visit:
                new_dataset[sample_idx] = (
                    new_dataset[visits[n_visit - 1]].clone().detach()
                )
            new_dataset[sample_idx][n_visit] = torch.tensor(
                sample_data, dtype=torch.bool
            )
            new_labels[sample_idx] = sample["label"]

    del dataset  # this is holding many gigs of RAM

    new_dataset_train, new_dataset_val, new_dataset_test = split_data(
        new_dataset, .9 - val_ratio, val_ratio, 0.1
    )
    
    train_labels, val_labels, test_labels = split_data(new_labels, .9 - val_ratio, val_ratio, 0.1)


    train_dataset = TensorDataset(new_dataset_train, train_labels)
    del new_dataset_train
    val_dataset = TensorDataset(new_dataset_val, val_labels)
    del new_dataset_val
    test_dataset = TensorDataset(new_dataset_test, test_labels)
    del new_dataset_test
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    del test_dataset

    experiment = Experiment(
        api_key="eQ3INeSsFGUYKahSdEtjhry42", project_name="general", workspace="samdoud"
    )

    clf = NeuralNetworkClassifier(
        SAnD(in_feature, seq_len, n_heads, factor, num_class, num_layers),
        nn.CrossEntropyLoss(),
        optim.Adam,
        optimizer_config={
            "lr": 5e-4,
            "betas": (0.9, 0.98),
            "eps": 4e-09,
            "weight_decay": 5e-4,
        },
        experiment=experiment,
    )

    # training network
    epochs = 10
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=get_weighted_sampler(train_labels))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    clf.fit(
        {"train": train_loader, "val": val_loader},
        epochs=epochs,
        start_epoch=0,
        total_epochs=epochs,
        train_dataset_size=len(train_dataset),
    )

    # evaluating
    clf.evaluate(test_loader)

    # save
    clf.save_to_file("./save_params/")
