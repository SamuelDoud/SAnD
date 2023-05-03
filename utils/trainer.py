from collections import defaultdict
import os
import time
import tqdm
import pandas as pd
from copy import deepcopy
from typing import Dict, Optional
import warnings
import numpy as np

warnings.simplefilter(action='ignore', category=np.VisibleDeprecationWarning)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    auc,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
)


class NeuralNetworkClassifier:
    """
    | NeuralNetworkClassifier depend on `Comet-ML <https://www.comet.ml/>`_ .
    | You have to create a project on your workspace of Comet, if you use this class.
    |
    | example

    ---------------------
    1st, Write your code.
    ---------------------
    ::

        # code.py
        from comet_ml import Experiment
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from SAnD.utils.trainer import NeuralNetworkClassifier

        class Network(nn.Module):
           def __init__(self):
               super(Network ,self).__init__()
               ...
           def forward(self, x):
               ...

        optimizer_config = {"lr": 0.001, "betas": (0.9, 0.999), "eps": 1e-08}
        comet_config = {}

        train_val_loader = {
           "train": train_loader,
           "val": val_loader
        }
        test_loader = DataLoader(test_ds, batch_size)

        clf = NeuralNetworkClassifier(
                Network(), nn.CrossEntropyLoss(),
                optim.Adam, optimizer_config, Experiment()
            )

        clf.experiment_tag = "experiment_tag"
        clf.num_classes = 3
        clf.fit(train_val_loader, epochs=10)
        clf.evaluate(test_loader)
        lf.confusion_matrix(test_ds)
        clf.save_weights("save_params_test/")

    ----------------------------
    2nd, Run code on your shell.
    ----------------------------
    | You need to define 2 environment variables.
    | :code:`COMET_API_KEY` & :code:`COMET_PROJECT_NAME`

    On Unix-like system, you can define them like this and execute code.
    ::

        export COMET_API_KEY="YOUR-API-KEY"
        export COMET_PROJECT_NAME="YOUR-PROJECT-NAME"
        user@user$ python code.py

    -------------------------------------------
    3rd, check logs on your workspace of comet.
    -------------------------------------------
    Just access your `Comet-ML <https://www.comet.ml/>`_ Project page.

    ^^^^^
    Note,
    ^^^^^

    Execute this command on your shell, ::

        export COMET_DISABLE_AUTO_LOGGING=1

    If the following error occurs. ::

        ImportError: You must import Comet before these modules: torch

    """

    def __init__(
        self, model, criterion, optimizer, optimizer_config: dict, experiment
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer(self.model.parameters(), **optimizer_config)
        self.criterion = criterion
        self.experiment = experiment

        self.hyper_params = optimizer_config
        self._start_epoch = 0
        self.hyper_params["epochs"] = self._start_epoch
        self.__num_classes = None
        self._is_parallel = False

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self._is_parallel = True

            notice = "Running on {} GPUs.".format(torch.cuda.device_count())
            print("\033[33m" + notice + "\033[0m")

    def fit(
        self,
        loader: Dict[str, DataLoader],
        epochs: int,
        checkpoint_path: str = None,
        validation: bool = True,
        start_epoch: Optional[int] = 0,
        total_epochs: Optional[int] = None,
        train_dataset_size: Optional[int] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        """
        | The method of training your PyTorch Model.
        | With the assumption, This method use for training network for classification.

        ::

            train_ds = Subset(train_val_ds, train_index)
            val_ds = Subset(train_val_ds, val_index)

            train_val_loader = {
                "train": DataLoader(train_ds, batch_size),
                "val": DataLoader(val_ds, batch_size)
            }

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )
            clf.fit(train_val_loader, epochs=10)


        :param loader: Dictionary which contains Data Loaders for training and validation.: dict{DataLoader, DataLoader}
        :param epochs: The number of epochs: int
        :param checkpoint_path: str
        :param validation:
        :return: None
        """
        len_of_train_dataset = len(loader["train"].dataset)
        epochs = epochs + self._start_epoch

        self.hyper_params["epochs"] = epochs
        self.hyper_params["batch_size"] = loader["train"].batch_size
        self.hyper_params["train_ds_size"] = len_of_train_dataset

        if validation:
            len_of_val_dataset = len(loader["val"].dataset)
            self.hyper_params["val_ds_size"] = len_of_val_dataset

        self.experiment.log_parameters(self.hyper_params)

        for epoch in range(self._start_epoch + start_epoch, epochs + start_epoch):
            if checkpoint_path is not None and epoch % 100 == 0:
                self.save_to_file(checkpoint_path)
            with self.experiment.train():
                running_y = []
                running_pred = []
                mses = []
                correct = 0.0
                total = 0.0
                running_outputs = []

                self.model.train()
                pbar = tqdm.tqdm(
                    total=len_of_train_dataset
                    if not train_dataset_size
                    else train_dataset_size
                )
                for x, y in loader["train"]:
                    b_size = y.shape[0]
                    total += y.shape[0]
                    x = (
                        x.type(torch.LongTensor).to(self.device)
                        if isinstance(x, torch.Tensor)
                        else [i.to(self.device) for i in x]
                    )
                    y = y.type(torch.LongTensor).to(self.device)

                    pbar.set_description(
                        "\033[36m"
                        + "Training"
                        + "\033[0m"
                        + " - Epochs: {:03d}/{:03d}".format(
                            epoch + 1, total_epochs if total_epochs else epochs
                        )
                    )
                    pbar.update(b_size)

                    self.optimizer.zero_grad()
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y.float())
                    loss.backward()
                    self.optimizer.step()
                    _, predicted = torch.max(outputs, 1)
                    running_y += y.tolist()
                    running_pred += predicted.tolist()
                    running_outputs += outputs.tolist()
                    if not isinstance(self.criterion, nn.BCEWithLogitsLoss):
                        correct += (predicted == y).sum().float().cpu().item()
                        self.experiment.log_metric(
                            "accuracy", float(correct / total), step=epoch
                        )
                        mses.extend(
                            int(y[i] - predicted[i]) ** 2
                            for i in range(predicted.shape[0])
                        )
                        self.experiment.log_metric("MSE", sum(mses) / total, step=epoch)
                        fpr, tpr, _ = roc_curve(
                            y_true=running_y, y_score=running_pred, pos_label=1
                        )
                        self.experiment.log_metric("AUROC", auc(fpr, tpr), step=epoch)
                        precision, recall, _ = precision_recall_curve(
                            y_true=running_y, probas_pred=running_pred, pos_label=1
                        )
                        self.experiment.log_metric(
                            "AUPRC", auc(recall, precision), step=epoch
                        )
                        self.experiment.log_confusion_matrix(
                            running_y,
                            running_pred,
                            title=f"Confusion Matrix, train",
                            file_name=f"confusion-matrix-train-.json",
                            step=epoch,
                        )

                    self.experiment.log_metric("loss", loss.cpu().item(), step=epoch)

            if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                with self.experiment.train():
                    try:
                        self.experiment.log_metric(
                            "accuracy",
                            roc_auc_score(running_y, running_outputs),
                            step=epoch,
                        )
                    except ValueError as e:
                        print(f"error on train {epoch=}")
                        raise e
            if verbose:
                self.print_metrics("Train")
            if validation:
                running_y = []
                running_pred = []
                mses = []
                with self.experiment.validate():
                    with torch.no_grad():
                        val_correct = 0.0
                        val_total = 0.0
                        running_outputs = []
                        running_y = []
                        self.model.eval()
                        for x_val, y_val in loader["val"]:
                            val_total += y_val.shape[0]
                            x_val = (
                                x_val.type(torch.LongTensor).to(self.device)
                                if isinstance(x_val, torch.Tensor)
                                else [i_val.to(self.device) for i_val in x_val]
                            )
                            y_val = y_val.type(torch.LongTensor).to(self.device)

                            val_output = self.model(x_val)
                            val_loss = self.criterion(val_output, y_val.float())
                            _, val_pred = torch.max(val_output, 1)
                            running_y += y_val.tolist()
                            running_outputs += val_output.tolist()
                            running_pred += val_pred.tolist()

                            if not isinstance(self.criterion, nn.BCEWithLogitsLoss):
                                val_correct += (
                                    (val_pred == y_val).sum().float().cpu().item()
                                )
                                self.experiment.log_metric(
                                    "accuracy",
                                    float(val_correct / val_total),
                                    step=epoch,
                                )
                                mses.extend(
                                    int(y_val[i] - val_pred[i]) ** 2
                                    for i in range(val_pred.shape[0])
                                )
                                self.experiment.log_metric(
                                    "MSE", sum(mses) / total, step=epoch
                                )
                                fpr, tpr, _ = roc_curve(
                                    y_true=running_y, y_score=running_pred, pos_label=1
                                )
                                self.experiment.log_metric(
                                    "AUROC", auc(fpr, tpr), step=epoch
                                )
                                precision, recall, _ = precision_recall_curve(
                                    y_true=running_y,
                                    probas_pred=running_pred,
                                    pos_label=1,
                                )
                                self.experiment.log_metric(
                                    "AUPRC", auc(recall, precision), step=epoch
                                )

                                self.experiment.log_confusion_matrix(
                                    running_y,
                                    running_pred,
                                    title=f"Confusion Matrix, val",
                                    file_name=f"confusion-matrix-val.json",
                                    step=epoch,
                                )

                            self.experiment.log_metric(
                                "loss", val_loss.cpu().item(), step=epoch
                            )

                if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                    with self.experiment.validate():
                        try:
                            self.experiment.log_metric(
                                "accuracy",
                                roc_auc_score(running_y, running_outputs),
                                step=epoch,
                            )
                        except ValueError as e:
                            print(f"error on val {epoch=}")
                            raise e
                if verbose:
                    self.print_metrics("Validation")

            pbar.close()

    def evaluate(self, loader: DataLoader, verbose: bool = False) -> None or float:
        """
        The method of evaluating your PyTorch Model.
        With the assumption, This method use for training network for classification.

        ::

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )
            clf.evaluate(test_loader)


        :param loader: DataLoader for Evaluating: torch.utils.data.DataLoader
        :param verbose: bool
        :return: None
        """
        running_loss = 0.0
        running_y = []
        running_pred = []
        mses = []
        pbar = tqdm.tqdm(total=len(loader.dataset))

        self.model.eval()
        self.experiment.log_parameter("test_ds_size", len(loader.dataset))

        with self.experiment.test():
            with torch.no_grad():
                correct = 0.0
                total = 0.0
                running_y = []
                running_outputs = []
                for x, y in loader:
                    b_size = y.shape[0]
                    total += b_size
                    x = (
                        x.type(torch.LongTensor).to(self.device)
                        if isinstance(x, torch.Tensor)
                        else [i.to(self.device) for i in x]
                    )
                    y = y.type(torch.LongTensor).to(self.device)

                    pbar.set_description("\033[32m" + "Evaluating" + "\033[0m")
                    pbar.update(b_size)

                    outputs = self.model(x)
                    loss = self.criterion(outputs, y.float())
                    _, predicted = torch.max(outputs, 1)
                    running_outputs += outputs.tolist()
                    running_y += predicted.tolist()

                    if not isinstance(self.criterion, nn.BCEWithLogitsLoss):
                        correct += (predicted == y).sum().float().cpu().item()
                        self.experiment.log_metric("accuracy", float(correct / total))
                        mses.extend(
                            int(y[i] - predicted[i]) ** 2 for i in range(b_size)
                        )
                        self.experiment.log_metric("MSE", sum(mses) / total)
                        fpr, tpr, _ = roc_curve(
                            y_true=running_y, y_score=running_pred, pos_label=1
                        )
                        self.experiment.log_metric("AUROC", auc(fpr, tpr))
                        precision, recall, _ = precision_recall_curve(
                            y_true=running_y, probas_pred=running_pred, pos_label=1
                        )
                        self.experiment.log_metric("AUPRC", auc(recall, precision))

                    running_loss += loss.cpu().item()
                    running_pred += predicted.tolist()

                    self.experiment.log_metric("loss", running_loss)

                pbar.close()

        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            with self.experiment.test():
                try:
                    self.experiment.log_metric(
                        "accuracy",
                        roc_auc_score(running_y, running_outputs),
                    )
                except ValueError:
                    print("error on test")
        else:
            self.experiment.log_confusion_matrix(
                running_y,
                running_pred,
                title="Confusion Matrix, Test",
                file_name="confusion-matrix-test.json",
            )
        self.print_metrics("Evaluation")

    def safe_get_metric(self, key: str) -> float:
        try:
            metric = self.experiment.get_metric(key)
        except KeyError:
            metric = 0.0
        finally:
            return metric

    def print_metrics(self, name: str) -> None:
        acc = self.safe_get_metric("accuracy")
        mse = self.safe_get_metric("MSE")
        auroc = self.safe_get_metric("AUROC")
        auprc = self.safe_get_metric("AUPRC")
        color = defaultdict(
            lambda: "35", {"test": "33", "validation": "31", "train": "32"}
        )
        print(
            f"\033[{color[name.lower()]}m"
            + f"{name} finished. "
            + "\033[0m"
            + "Accuracy: {:.4f} MSE: {:.4f} AUROC: {:.4f} AUPRC: {:.4f}".format(
                acc, mse, auroc, auprc
            )
        )

    def save_checkpoint(self) -> dict:
        """
        The method of saving trained PyTorch model.

        Note,  return value contains
            - the number of last epoch as `epochs`
            - optimizer state as `optimizer_state_dict`
            - model state as `model_state_dict`

        ::

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )

            clf.fit(train_loader, epochs=10)
            checkpoints = clf.save_checkpoint()

        :return: dict {'epoch', 'optimizer_state_dict', 'model_state_dict'}
        """

        checkpoints = {
            "epoch": deepcopy(self.hyper_params["epochs"]),
            "optimizer_state_dict": deepcopy(self.optimizer.state_dict()),
        }

        if self._is_parallel:
            checkpoints["model_state_dict"] = deepcopy(self.model.module.state_dict())
        else:
            checkpoints["model_state_dict"] = deepcopy(self.model.state_dict())

        return checkpoints

    def save_to_file(self, path: str) -> str:
        """
        | The method of saving trained PyTorch model to file.
        | Those weights are uploaded to comet.ml as backup.
        | check "Asserts".

        Note, .pth file contains
            - the number of last epoch as `epochs`
            - optimizer state as `optimizer_state_dict`
            - model state as `model_state_dict`

        ::

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )

            clf.fit(train_loader, epochs=10)
            filename = clf.save_to_file('path/to/save/dir/')

        :param path: path to saving directory. : string
        :return: path to file : string
        """
        if not os.path.isdir(path):
            os.mkdir(path)

        file_name = "model_params-epochs_{}-{}.pth".format(
            self.hyper_params["epochs"], time.time()
        )
        path = os.path.join(path, file_name)

        checkpoints = self.save_checkpoint()

        torch.save(checkpoints, path)
        self.experiment.log_asset(path, file_name=file_name)

        return path

    def restore_checkpoint(self, checkpoints: dict) -> None:
        """
        The method of loading trained PyTorch model.

        :param checkpoints: dictionary which contains {'epoch', 'optimizer_state_dict', 'model_state_dict'}
        :return: None
        """
        self._start_epoch = checkpoints["epoch"]
        if not isinstance(self._start_epoch, int):
            raise TypeError

        if self._is_parallel:
            self.model.module.load_state_dict(checkpoints["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoints["model_state_dict"])

        self.optimizer.load_state_dict(checkpoints["optimizer_state_dict"])

    def restore_from_file(self, path: str, map_location: str = "cpu") -> None:
        """
        The method of loading trained PyTorch model from file.

        ::

            clf = NeuralNetworkClassifier(
                    Network(), nn.CrossEntropyLoss(),
                    optim.Adam, optimizer_config, experiment
                )
            clf.restore_from_file('path/to/trained/weights.pth')

        :param path: path to saved directory. : str
        :param map_location: default cpu: str
        :return: None
        """
        checkpoints = torch.load(path, map_location=map_location)
        self.restore_checkpoint(checkpoints)

    @property
    def experiment_tag(self) -> list:
        return self.experiment.get_tags()

    @experiment_tag.setter
    def experiment_tag(self, tag: str) -> None:
        """
        ::

            clf = NeuralNetworkClassifier(...)
            clf.experiment_tag = "tag"

        :param tag: str
        :return: None
        """
        if not isinstance(tag, str):
            raise TypeError

        self.experiment.add_tag(tag)

    @property
    def num_class(self) -> int or None:
        return self.__num_classes

    @num_class.setter
    def num_class(self, num_class: int) -> None:
        if not (isinstance(num_class, int) and num_class > 0):
            raise Exception("the number of class must be greater than 0.")

        self.__num_classes = num_class
        self.experiment.log_parameter("classes", self.__num_classes)

    def confusion_matrix(
        self, dataset: torch.utils.data.Dataset, labels=None, sample_weight=None
    ) -> None:
        """
        | Generate confusion matrix.
        | result save on comet.ml.

        :param dataset: dataset for generating confusion matrix.
        :param labels: array, shape = [n_samples]
        :param sample_weight: array-lie of shape = [n_samples], optional
        :return: None
        """
        targets = []
        predicts = []
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        pbar = tqdm.tqdm(total=len(loader.dataset))

        self.model.eval()

        with torch.no_grad():
            for step, (x, y) in enumerate(loader):
                x = x.to(self.device)

                pbar.set_description(
                    "\033[31m" + "Calculating confusion matrix" + "\033[0m"
                )
                pbar.update(step)

                outputs = self.model(x)
                _, predicted = torch.max(outputs, 1)

                predicts.append(predicted.cpu().numpy())
                targets.append(y.numpy())
            pbar.close()

        cm = pd.DataFrame(confusion_matrix(targets, predicts, labels, sample_weight))
        self.experiment.log_asset_data(
            cm.to_csv(),
            "ConfusionMatrix-epochs-{}-{}.csv".format(
                self.hyper_params["epochs"], time.ctime().replace(" ", "_")
            ),
        )
