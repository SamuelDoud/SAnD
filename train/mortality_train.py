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

from utils.functions import (
    get_pca,
    pca_transform,
    split_data,
    Tokenizers,
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
datasets = [mimiciii_mortality]

clamp_seq = 28
max_visits = 500000
for i in range(len(datasets)):
    dataset = datasets[i]()  # don't load this until need, then purge the old one
    tokenizers = Tokenizers(
        ["drugs", "conditions", "procedures"], dataset=dataset, depth=0
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

    n_heads = 32
    factor = 32
    num_class = 2
    num_layers = 6
    batch_size = 1024
    val_ratio = 0.2
    seed = 1234
    pca_dim = 20
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
        new_dataset, 0.7, 0.1, 0.2
    )
    pca_train, _, _ = split_data(new_dataset, 0.1, 0.90, 0.0)
    train_labels, val_labels, test_labels = split_data(new_labels, 0.7, 0.1, 0.2)

    logging.info(
        f"The shape before PCA, new_dataset_train.shape = {new_dataset_train.shape}, "
        f"new_dataset_test.shape = {new_dataset_test.shape}"
    )
    pca = get_pca(pca_train, pca_dim=pca_dim)
    del pca_train

    train_data = pca_transform(pca=pca, dataset=new_dataset_train)
    del new_dataset_train
    val_data = pca_transform(pca=pca, dataset=new_dataset_val)
    del new_dataset_val
    test_data = pca_transform(pca=pca, dataset=new_dataset_test)
    del new_dataset_test
    logging.info(
        f"The shape after PCA, train_data.shape = {train_data.shape},"
        f" test_data.shape = {test_data.shape}, with pca_component = {pca_dim}"
    )

    train_dataset = TensorDataset(train_data, train_labels)
    del train_data
    val_dataset = TensorDataset(val_data, val_labels)
    del val_data
    test_dataset = TensorDataset(test_data, test_labels)
    del test_data
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
            "lr": 1e-5,
            "betas": (0.9, 0.98),
            "eps": 4e-09,
            "weight_decay": 5e-4,
        },
        experiment=experiment,
    )

    # training network
    epochs = 24
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
