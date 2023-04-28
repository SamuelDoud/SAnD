from typing import List, Optional, Tuple
from torch import Tensor, float32, tensor, from_numpy
import numpy as np
from pyhealth.tokenizer import Tokenizer
from sklearn.decomposition import PCA


def positional_encoding(n_positions: int, hidden_dim: int) -> Tensor:
    def calc_angles(pos, i):
        rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(hidden_dim))
        return pos * rates

    rads = calc_angles(
        np.arange(n_positions)[:, np.newaxis], np.arange(hidden_dim)[np.newaxis, :]
    )

    rads[:, 0::2] = np.sin(rads[:, 0::2])
    rads[:, 1::2] = np.cos(rads[:, 1::2])

    pos_enc = rads[np.newaxis, ...]
    pos_enc = tensor(pos_enc, dtype=float32, requires_grad=False)
    return pos_enc


def dense_interpolation(batch_size: int, seq_len: int, factor: int) -> Tensor:
    W = np.zeros((factor, seq_len), dtype=np.float32)
    for t in range(seq_len):
        s = np.array((factor * (t + 1)) / seq_len, dtype=np.float32)
        for m in range(factor):
            tmp = np.array(1 - (np.abs(s - (1 + m)) / factor), dtype=np.float32)
            w = np.power(tmp, 2, dtype=np.float32)
            W[m, t] = w

    W = tensor(W, requires_grad=False).float().unsqueeze(0)
    return W.repeat(batch_size, 1, 1)


def subsequent_mask(size: int) -> Tensor:
    """
    from Harvard NLP
    The Annotated Transformer

    http://nlp.seas.harvard.edu/2018/04/03/attention.html#batches-and-masking

    :param size: int
    :return: torch.Tensor
    """
    attn_shape = (size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype("float32")
    mask = from_numpy(mask) == 0
    return mask.float()


class ScheduledOptimizer:
    """
    Reference: `jadore801120/attention-is-all-you-need-pytorch \
    <https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Optim.py>`_
    """

    def __init__(self, optimizer, d_model: int, warm_up: int) -> None:
        self._optimizer = optimizer
        self.warm_up = warm_up
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    def step(self) -> None:
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self) -> None:
        self._optimizer.zero_grad()

    def _get_lr_scale(self) -> np.array:
        return np.min(
            [
                np.power(self.n_current_steps, -0.5),
                np.power(self.warm_up, -1.5) * self.n_current_steps,
            ]
        )

    def get_lr(self):
        lr = self.init_lr * self._get_lr_scale()
        return lr

    def _update_learning_rate(self):
        self.n_current_steps += 1
        lr = self.get_lr()

        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr

    def state_dict(self):
        return self._optimizer.state_dict()


def strings_to_tensor(list_of_strs: List[str]) -> Tensor:
    return Tensor([int(v) for v in list_of_strs])


def split_data(data, train: float, val: float, test: float):
    """
    :param data: the data to be split,
    :param train: the training ratio
    :param val: the val ratio.
    :param test: the test ratio.
    """
    err = 1e-5
    if not 1 - err < (train + val + test) < 1 + err:
        raise Exception(
            f"{train=} + {val=} + {test=} = {train + val + test}. Needs to be 1."
        )
    length = len(data)
    end_train = int(length * train)
    end_val = int(length * val) + end_train
    return data[:end_train], data[end_train:end_val], data[end_val:]


tokenizers = {}


def tokenizer_helper(sample, key: str, dataset) -> np.array:
    if key not in tokenizers:
        alls = {s for l in [sample[key][0] for sample in dataset.samples] for s in l}
        tokenizers[key] = Tokenizer(list(alls))
    tokenizer = tokenizers[key]
    items = sample[key][0]
    item_table = np.zeros(shape=(tokenizer.get_vocabulary_size()))
    item_indicies = tokenizer.convert_tokens_to_indices(items)
    item_table[item_indicies] = True
    return item_table


def pca_data(
    dataset_train: Tensor, dataset_test: Tensor, pca_dim=20
) -> Tuple[Tensor, Tensor]:
    # convert data from tensor to numpy
    dataset_train_np = dataset_train.numpy()
    dataset_test_np = dataset_test.numpy()

    # reshape alo
    dataset_train_np_flatten = dataset_train_np.reshape(
        dataset_train_np.shape[0] * dataset_train_np.shape[1], dataset_train_np.shape[2]
    )
    dataset_test_np_flatten = dataset_test_np.reshape(
        dataset_test_np.shape[0] * dataset_test_np.shape[1], dataset_test_np.shape[2]
    )

    pca = PCA(n_components=pca_dim)
    dataset_train_np_flatten_pca = pca.fit_transform(dataset_train_np_flatten)
    dataset_test_np_flatten_pca = pca.transform(dataset_test_np_flatten)

    dataset_train_np_pca = dataset_train_np_flatten_pca.reshape(
        dataset_train_np.shape[0], dataset_train_np.shape[1], pca_dim
    )
    dataset_test_np_pca = dataset_test_np_flatten_pca.reshape(
        dataset_test_np.shape[0], dataset_test_np.shape[1], pca_dim
    )
    return Tensor(dataset_train_np_pca), Tensor(dataset_test_np_pca)
