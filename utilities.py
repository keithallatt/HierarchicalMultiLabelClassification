"""
utilities.py

A variety of utility functions such as loading and saving to pickle files, and calculating accuracies and such.

Author(s): Keith Allatt,
"""
import datetime
from typing import Union

import numpy as np
import os.path
from pathlib import Path
import pickle
import time
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def make_progressbar(length: int, progress: float, naive: bool = False, time_start: float = None) -> str:
    """
    Create a progress bar of a specified length with a percentage filled.

    :param length: The length of the progress bar in characters.
    :param progress: Percent completion expressed as $0 <= p <= 1$. $p=0$ is an empty progress bar, and $p=1$ is full.
    :param naive: Don't use the fractional block characters. This trades off some granularity for computations.
    :param time_start: If provided, then the time elapsed is displayed in HH:MM:SS format.
    """
    l2 = progress * length
    bar = "█"

    t_str = ""
    if time_start is not None:
        t_elapsed = (time.time() - time_start)
        t_str = str(datetime.timedelta(seconds=int(t_elapsed)))
        t_str = t_str.rjust(10)

    if naive:
        return (bar * round(progress * length)).ljust(length)[:length] + t_str

    num_full, fraction = divmod(l2, 1)

    bar_ord = ord(bar)
    bar *= num_full
    eighths = round(fraction * 8)
    chars = " " + ''.join([chr(bar_ord + i) for i in range(8)][::-1])
    return (bar + chars[eighths]).ljust(length)[:length] + f"{chars[1]}{progress * 100:.2f}% {t_str}"


def dump_data(outfile: str, obj: object, make_dirs: bool = False) -> None:
    """
    Dump an object into a pickle file.

    :param outfile: Filepath of the pickle file.
    :param obj: The object to dump.
    :param make_dirs: If True, makes the parent folder, if the parent folder does not exist.
    """
    if make_dirs:
        parent_folder = Path(outfile).parent.absolute()
        if not os.path.exists(parent_folder):
            os.makedirs(parent_folder)
    with open(outfile, 'wb') as file:
        pickle.dump(obj, file)


def load_data(infile: str) -> object:
    """
    Load a pickle file into an object.

    :param infile: Filepath of the pickle file.]
    :returns The un-pickled file.
    """
    with open(infile, 'rb') as file:
        return pickle.load(file)


def to_one_hot(z: Union[int, list], d: int) -> np.ndarray:
    """
    Convert z('s) into one hot vectors of size d.

    :param z: The list of indices.
    :param d: The dimensionality.
    :returns A NumPy array of one hot vectors.
    """
    if isinstance(z, int):
        z = [z]
    identity = np.eye(d, dtype=np.float32)
    assert set(z).issubset(set(range(d))), "Invalid data for one hot vector"
    return np.array(list(map(identity.__getitem__, z)))


def get_accuracy(model: nn.Module, data: Dataset, str_repr: bool = False, max_total: int = None) -> Union[str, float]:
    """
    Calculate accuracy for a given model and dataset.

    :param model: The PyTorch Model being evaluated.
    :param data: The data set to be evaluated by the model.
    :param str_repr: Represent the accuracy as a string if true, else a ratio of correct to total elements.
    :param max_total: The maximum number of data points to evaluate (used for estimation).
    """
    dataloader = DataLoader(data, batch_size=500)

    model.eval()  # annotation for evaluation; sets dropout and batch norm for evaluation.

    correct = 0
    total = 0

    for xs, ts in dataloader:
        output = model(xs)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(ts.view_as(pred)).sum().item()
        total += xs.shape[0]
        if max_total is not None and total > max_total:
            break

    return f"{correct} / {total} ({correct/total:.4f})" if str_repr else (correct / total)


def estimate_accuracy(model: nn.Module, data: Dataset) -> float:
    """
    Estimate accuracy using 2000 data points.

    :param model: The PyTorch Model being evaluated.
    :param data: The data set to be evaluated by the model.
    """
    return get_accuracy(model, data, max_total=2000)


def top_n_error_rate(model: nn.Module, data: Dataset, n: int) -> float:
    """
    Rate at which correct label is not in the top n predictions.

    :param model: The model to evaluate
    :param data: The test data set
    :param n: number of top predictions to verify label is in
    :return: the top n error rate
    """
    not_in_top_n = 0
    total = 0

    model.eval()  # annotation for evaluation; sets dropout and batch norm (not necessary yet)

    dl = DataLoader(data, batch_size=128)

    for xs, ts in dl:
        predictions = model(xs)
        top_n = torch.topk(predictions, k=n, dim=1)
        top_n_i = top_n.indices
        ts_r = np.repeat(ts.reshape((ts.shape[0], 1)), n, axis=1)

        not_in_top_n += (ts_r != top_n_i).all(axis=1).sum().item()
        total += xs.shape[0]

    # safe divide by zero
    return 0. if total == 0 else not_in_top_n / total


def make_layer_mult_mlp(input_size: int, output_size: int, layer_multiples: tuple) -> nn.Sequential:
    """Make MLP with each layer defined as a multiple of the previous.
    """
    layers = list()
    prev_size = input_size
    for mult in layer_multiples:
        layers.append(nn.Linear(prev_size, prev_size*mult))
        prev_size *= mult
    layers.append(nn.Linear(prev_size, output_size))
    return nn.Sequential(*layers)


if __name__ == "__main__":
    pass
