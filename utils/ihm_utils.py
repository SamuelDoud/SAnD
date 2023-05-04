import json
from typing import Optional
import os

import pandas as pd
import numpy as np
import sklearn.utils as sk_utils
import torch
from mimic3_benchmarks.mimic3models.metrics import print_metrics_binary

from utils import common_utils


def load_data(reader, discretizer, normalizer, small_part=False, return_names=False):
    N = reader.get_number_of_examples()
    if small_part:
        N = 1000
    ret = common_utils.read_chunk(reader, N)
    data = ret["X"]
    ts = ret["t"]
    labels = ret["y"]
    names = ret["name"]
    data = [discretizer.transform(X, end=t)[0] for (X, t) in zip(data, ts)]
    if normalizer is not None:
        data = [normalizer.transform(X) for X in data]
    whole_data = (torch.tensor(np.array(data)), torch.tensor(labels))
    if not return_names:
        return whole_data
    return {"data": whole_data, "names": names}


def save_results(names, pred, y_true, path):
    common_utils.create_directory(os.path.dirname(path))
    with open(path, "w") as f:
        f.write("stay,prediction,y_true\n")
        for name, x, y in zip(names, pred, y_true):
            f.write("{},{:.6f},{}\n".format(name, x, y))


def evaluate_predictions(
    pred_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_iters: Optional[int] = 10000,
    save_file: Optional[str] = "results/ihm_results.json",
):
    df = test_df.merge(
        pred_df, left_on="stay", right_on="stay", how="left", suffixes=["_l", "_r"]
    )
    assert df["prediction"].isnull().sum() == 0
    assert df["y_true_l"].equals(df["y_true_r"])

    metrics = [
        ("AUC of ROC", "auroc"),
        ("AUC of PRC", "auprc"),
        ("min(+P, Se)", "minpse"),
    ]

    data = np.zeros((df.shape[0], 2))
    data[:, 0] = np.array(df["prediction"])
    data[:, 1] = np.array(df["y_true_l"])

    results = dict()
    results["n_iters"] = n_iters
    ret = print_metrics_binary(data[:, 1], data[:, 0], verbose=0)
    for m, k in metrics:
        results[m] = dict()
        results[m]["value"] = ret[k]
        results[m]["runs"] = []

    for i in range(n_iters):
        cur_data = sk_utils.resample(data, n_samples=len(data))
        ret = print_metrics_binary(cur_data[:, 1], cur_data[:, 0], verbose=0)
        for m, k in metrics:
            results[m]["runs"].append(ret[k])

    for m, k in metrics:
        runs = results[m]["runs"]
        results[m]["mean"] = np.mean(runs)
        results[m]["median"] = np.median(runs)
        results[m]["std"] = np.std(runs)
        results[m]["2.5% percentile"] = np.percentile(runs, 2.5)
        results[m]["97.5% percentile"] = np.percentile(runs, 97.5)
        del results[m]["runs"]

    print("Saving the results in {} ...".format(save_file))
    with open(save_file, "w") as f:
        json.dump(results, f, indent=4)

    print(results)
