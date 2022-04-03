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

from tqdm import tqdm

import pickle
import time
from datetime import datetime, timedelta
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import matplotlib.pyplot as plt


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
    bar *= int(num_full)
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


# def to_one_hot(z: Union[int, list], d: int) -> np.ndarray:
#     """
#     Convert z('s) into one hot vectors of size d.
#
#     :param z: The list of indices.
#     :param d: The dimensionality.
#     :returns A NumPy array of one hot vectors.
#     """
#     if isinstance(z, int):
#         z = [z]
#     identity = np.eye(d, dtype=np.float32)
#     assert set(z).issubset(set(range(d))), "Invalid data for one hot vector"
#     return np.array(list(map(identity.__getitem__, z)))

def to_one_hot(z: torch.tensor, d: int) -> torch.tensor:
    """
    Convert z('s) into one hot vectors of size d.

    :param z: The list of indices.
    :param d: The dimensionality.
    :returns A NumPy array of one hot vectors.
    """
    return torch.eye(d)[z]


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
        for i in range(len(model.output_sizes)):
            pred = output[i].max(1, keepdim=True)[1]
            correct += pred.eq(ts[:,i].view_as(pred)).sum().item()
            total += xs.shape[0]

        if max_total is not None and total > max_total*len(model.output_sizes):
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
        layers.append(nn.ReLU())
        prev_size *= mult
    layers.append(nn.Linear(prev_size, output_size))
    return nn.Sequential(*layers)


# MODEL TRAINING FUNCTIONS @ KEITH.ALLATT, RENATO.ZIMMERMANN
def train(model, train_data, valid_data, batch_size=64, learning_rate=0.001, num_epochs=7,
          calc_acc_every=0, max_iterations=100_000, shuffle=True, train_until=1.0,
          device="cpu", checkpoint_path=None):
    """
    Train a model.
    """

    # can add a train_lock attribute to a model to prevent the train function from making changes.
    if hasattr(model, "train_lock"):
        if model.train_lock:
            print("Model is locked. Please unlock the model to train.")

    check_prefix = datetime.now().strftime("%H%M%S")
    if checkpoint_path is not None and not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    tot_train = len(train_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)

    # TODO: be able to chooose loss / optimizer with keyword arguments
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    its, its_sub, losses, train_acc, val_acc = [], [], [], [], []

    n_batch = 0
    # for pickled models, if training_iterations is an attribute, then
    if hasattr(model, "training_iterations"):
        n_batch = model.training_iterations

    done = False
    start_time = time.time()
    loss = 0
    try:

        for epoch in range(num_epochs):

            n_train = 0
            for xs, ts in train_loader:

                model.train()

                xs, ts = xs.to(device), ts.to(device)
                preds = model(xs)

                loss = 0
                for i, d in enumerate(model.output_sizes):
                    loss += criterion(preds[i], to_one_hot(ts[:,i], d))

                xs.detach(), ts.detach()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                its.append(n_batch)
                losses.append(float(loss)/batch_size)

                n_batch += 1
                n_train += xs.shape[0]

                if calc_acc_every != 0 and n_batch % calc_acc_every == 0:
                    its_sub.append(n_batch)
                    ta = get_accuracy(model, train_data)
                    va = get_accuracy(model, valid_data)
                    train_acc.append(ta)
                    val_acc.append(va)

                    if all(x >= train_until for x in set(train_acc[-2:])) or n_batch >= max_iterations:
                        done = True
                        break

                if not (n_batch % 10):
                    it_progress = tqdm.format_meter(n_train, tot_train, time.time()-start_time,
                                                    prefix=f"E:[{epoch+1}/{num_epochs}] I")
                    ta_progress = make_progressbar(8, estimate_accuracy(model, train_data))
                    va_progress = make_progressbar(8, estimate_accuracy(model, valid_data))
                    print(f"\r{it_progress} [TA:{ta_progress}] [VA:{va_progress}] ", end='')

            if done:
                break

        if checkpoint_path is not None:
            torch.save(model.state_dict(),
                        checkpoint_path + f"model_{check_prefix}_{epoch}")

    except KeyboardInterrupt:
        n = len(val_acc)
        its = its[:n]
        its_sub = its_sub[:n]
        losses = losses[:n]
        train_acc = train_acc[:n]

    if not train_acc:
        train_acc = [get_accuracy(model, train_data)]
    if not losses:
        losses = [float(loss) / batch_size]
    if not val_acc:
        val_acc = [get_accuracy(model, valid_data)]
    if not its_sub:
        its_sub = [n_batch]

    print()

    if hasattr(model, "training_iterations"):
        model.training_iterations = n_batch

    return its, its_sub, losses, train_acc, val_acc


def train_model(model, train_data, valid_data, test_data=None, data_loader=lambda x: x,
                outfile="model.pickle", train_opts=None, **kwargs):

    # can specify data_loader, by default, the identify function x -> x. Acts like a preprocessor.
    training_dataset = data_loader(train_data)
    validation_dataset = data_loader(valid_data)
    testing_dataset = data_loader(test_data)

    if train_opts is None:
        train_opts = dict()
    its, its_sub, losses, train_acc, val_acc = train(model, training_dataset, validation_dataset,
                                                     **train_opts)

    show = kwargs.get("show_plts", False)
    ask = kwargs.get("ask", False)

    loss_fig, train_fig = None, None
    if show:
        if len(its) > 1:
            try:
                loss_fig, ax = plt.subplots()
                ax.set_title("Loss Curve")
                ax.plot(its, losses, label="Train")
                ax.set_xlabel("Iterations")
                ax.set_ylabel("Loss")
                loss_fig.show()
            except ValueError:
                print("Loss curve unavailable")
                print(len(its), len(losses))

        if len(its_sub) > 1:
            try:
                train_fig, ax = plt.subplots()
                ax.set_title("Learning Curve")
                ax.plot(its_sub, train_acc, label="Train")
                ax.plot(its_sub, val_acc, label="Validation")
                ax.set_xlabel("Iterations")
                ax.set_ylabel("Training Accuracy")
                ax.legend(loc='best')
                train_fig.show()
            except ValueError:
                print("Learning curve unavailable")
                print(len(its_sub), len(train_acc), len(val_acc))

        input("Showing plots. Type anything to continue.")

    if train_acc:
        print("Final Training Accuracy: {}".format(train_acc[-1]))
    if val_acc:
        print("Final Validation Accuracy: {}".format(val_acc[-1]))
    if test_data is not None:
        print("-" * 30)
        str_repr_test_acc = get_accuracy(model, testing_dataset, str_repr=True)
        print("Final Test Accuracy: {}".format(str_repr_test_acc))

    if ask and input("Save to file? [y/n] > ").lower().strip() == "y":
        dump_data(outfile, model)
        if loss_fig is not None:
            loss_fig.savefig(kwargs.get("loss_fig_out", "loss_curve.png"))
        if train_fig is not None:
            train_fig.savefig(kwargs.get("train_fig_out", "train_curve.png"))


if __name__ == "__main__":
    pass
