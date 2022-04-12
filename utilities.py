"""
utilities.py

A variety of utility functions such as loading and saving to pickle files, and calculating accuracies and such.

Author(s): Keith Allatt,
"""
import datetime
from typing import Union
from unicodedata import category
from data_cleaning import csv_pt_pairs

import os.path
from pathlib import Path

from tqdm import tqdm

import pickle
import time
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import matplotlib.pyplot as plt

import numpy as np

# number of labels used in multi-label classification
NUM_LABELS = 3 # l1,l2,l3


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

    num_full, fraction = int(l2), l2 % 1

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

def to_one_hot(z: torch.tensor, d: int, device="cpu") -> torch.tensor:
    """
    Convert z('s) into one hot vectors of size d.

    :param z: The list of indices.
    :param d: The dimensionality.
    :returns A NumPy array of one hot vectors.
    """
    return torch.eye(d)[z].to(device)


def get_accuracy(model: nn.Module, data: Dataset, str_repr: bool = False, max_total: int = None, device="cpu") -> Union[str, float]:
    """
    Calculate accuracy for a given model and dataset.

    :param model: The PyTorch Model being evaluated.
    :param data: The data set to be evaluated by the model.
    :param str_repr: Represent the accuracy as a string if true, else a ratio of correct to total elements.
    :param max_total: The maximum number of data points to evaluate (used for estimation).

    TODO: modify to track l1,l2,l3 accuracy individually

    Total : (correct / total)
    L1 : (correct / total)
    L2 : (correct / total)
    L3 : (correct / total)

    """
    dataloader = DataLoader(data, batch_size=500)

    model.eval()  # annotation for evaluation; sets dropout and batch norm for evaluation.

    num_points = 0 

    category_correct = np.asarray([0,0,0,0]) # number of l1,l2,l3,total correct predictions
   

    for xs, ts in dataloader: # grab a batch of 500 (or less if data does not divide evenly into 500)

        xs, ts = xs.to(device), ts.to(device)
        num_points += xs.shape[0]

        output = model(xs)  # returns per category predictions.  3 x N x Li where Li is the number of classes in Li
        for i in range(len(model.output_sizes)):
            pred = output[i].max(1, keepdim=True)[1]
            num_correct = pred.eq(ts[:,i].view_as(pred)).sum().item()
            category_correct[i] += num_correct # update per category correct predictions
            category_correct[NUM_LABELS] += num_correct # update total correct predictions across categories
            #total += xs.shape[0]

        xs.detach(), ts.detach()

        # if max_total is not None and total > max_total*len(model.output_sizes):
        #     break

    num_points_across_categories = num_points * NUM_LABELS
    accuracies = category_correct[0:NUM_LABELS] / num_points
    accuracies = np.append(accuracies, category_correct[-1] / num_points_across_categories) # add avg accuracy across all categories
  

    acc_str = (f"Total : ({category_correct[-1]} / {num_points_across_categories}) {accuracies[-1]}%\n" 
               f"L1 : ({category_correct[0]} / {num_points}) {accuracies[0]}\n"  
               f"L2 : ({category_correct[1]} / {num_points}) {accuracies[1]}\n"   
               f"L3 : ({category_correct[2]} / {num_points}) {accuracies[2]}")   
    
           
    return acc_str if str_repr else accuracies

    #return f"{correct} / {total} ({correct/total:.4f})" if str_repr else (correct / total)


def estimate_accuracy(model: nn.Module, data: Dataset, device="cpu") -> float:
    """
    Estimate accuracy using 2000 data points.

    :param model: The PyTorch Model being evaluated.
    :param data: The data set to be evaluated by the model.
    """
    return get_accuracy(model, data, max_total=2000, device=device)[-1] # return total accuracy across all categories


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

    its, its_sub = [], []
  
    # overall, l1, l2 and l3 losses/accuracies
    # losses[i] = loss for Li category. losses[NUM_LABELS] = loss across all categories
    losses = [[], [], [], []]
    train_accs = [[], [], [], []]
    val_accs = [[], [], [], []]


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
                #print("batch size {}".format(xs.shape[0]))
                model.train()

                xs, ts = xs.to(device), ts.to(device)
                preds = model(xs)

                loss_sum = 0 # sum of l1+l2+l3 losses 

                for i, d in enumerate(model.output_sizes):
                    # per category mini-batch loss
                    loss = criterion(preds[i], to_one_hot(ts[:,i], d, device=device))
                   
                    loss_sum += loss
                    losses[i].append(float(loss))  # avg loss per category

                
                losses[NUM_LABELS].append(float(loss_sum) / NUM_LABELS) # avg loss across all categories
            
                xs.detach(), ts.detach()

                loss_sum.backward()
                optimizer.step()
                optimizer.zero_grad()

                its.append(n_batch)
              
                n_batch += 1
                n_train += xs.shape[0]

                if calc_acc_every != 0 and n_batch % calc_acc_every == 0:
                    its_sub.append(n_batch)
                   
                    # returns an array of per category + total accuracies
                    batch_train_accs = get_accuracy(model, train_data, device=device)
                    batch_val_accs = get_accuracy(model, valid_data, device=device)
                 
            
                    for i in range(NUM_LABELS+1):
                        train_accs[i].append(batch_train_accs[i])
                        val_accs[i].append(batch_val_accs[i])

                    # if all(x >= train_until for x in set(train_acc[-2:])) or n_batch >= max_iterations:
                    #     done = True
                    #     break

                if not (n_batch % 10):
                    it_progress = tqdm.format_meter(n_train, tot_train, time.time()-start_time,
                                                    prefix=f"E:[{epoch+1}/{num_epochs}] I")
                    ta_progress = make_progressbar(8, estimate_accuracy(model, train_data, device=device))
                    va_progress = make_progressbar(8, estimate_accuracy(model, valid_data, device=device))
                    print(f"\r{it_progress} [TA:{ta_progress}] [VA:{va_progress}] ", end='')

            if done:
                break

        if checkpoint_path is not None:
            torch.save(model.state_dict(),
                        checkpoint_path + f"model_{check_prefix}_{epoch}")

    
    except KeyboardInterrupt:
        n = len(val_accs[0])
        its = its[:n]
        its_sub = its_sub[:n]
        losses = [category[:n] for category in losses]
        #losses = losses[:n]
        train_accs = [category[:n] for category in train_accs]
        #train_acc = train_acc[:n]

    # what is this for?
    if not train_accs:
        assert(1 == 0)
        train_accs = get_accuracy(model, train_data)
    if not losses:
        assert(1 == 0)
        losses = [float(loss_sum) / batch_size]
    if not val_accs:
        assert(1 == 0)
        val_accs = get_accuracy(model, valid_data, device=device)
    if not its_sub:
        assert(1 == 0)
        its_sub = [n_batch]

    print()

    if hasattr(model, "training_iterations"):
        model.training_iterations = n_batch

    return its, its_sub, losses, train_accs, val_accs
    #return its, its_sub, losses, train_acc, val_acc



def gen_category_loss_plots(its, losses):
    for i in range(NUM_LABELS):
        loss_fig, ax = plt.subplots()
        ax.set_title(f"L{i+1} Loss Curve")
        ax.plot(its, losses[i], label="Train")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Loss")
        loss_fig.show()

def gen_category_acc_plots(its_sub, train_accs, val_accs):
    print("------------------")
    print(train_accs[0])
    for i in range(NUM_LABELS):
        train_fig, ax = plt.subplots()
        ax.set_title(f"L{i+1} Learning Curve")
        ax.plot(its_sub, train_accs[i], label="Train")
        ax.plot(its_sub, val_accs[i], label="Validation")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Training Accuracy")
        ax.legend(loc='best')
        train_fig.show()


def train_model(model, train_data, valid_data, test_data=None, data_loader=lambda x: x,
                outfile="model.pickle", train_opts=None, device="cpu", show_category_stats=True, **kwargs):

    # can specify data_loader, by default, the identify function x -> x. Acts like a preprocessor.
    training_dataset = data_loader(train_data)
    validation_dataset = data_loader(valid_data)
    testing_dataset = data_loader(test_data)

    if train_opts is None:
        train_opts = dict()
    its, its_sub, losses, train_accs, val_accs = train(model, training_dataset, validation_dataset,
                                                     device=device, **train_opts)

    show = kwargs.get("show_plts", False)
    ask = kwargs.get("ask", False)

    
    loss_fig, train_fig = None, None
    if show:
        if len(its) > 1:
            try:
                loss_fig, ax = plt.subplots()
                ax.set_title("Loss Curve Across Categories")
                ax.plot(its, losses[-1], label="Train")
                ax.set_xlabel("Iterations")
                ax.set_ylabel("Loss")
                loss_fig.show()

                if show_category_stats:
                    gen_category_loss_plots(its, losses)
                    
            except ValueError:
                print("Loss curve unavailable")
                print(len(its), len(losses))

    # generate accuracies

        if len(its_sub) > 1:
            try:
                train_fig, ax = plt.subplots()
                ax.set_title("Learning Curve Across All Categories")
                ax.plot(its_sub, train_accs[-1], label="Train")
                ax.plot(its_sub, val_accs[-1], label="Validation")
                ax.set_xlabel("Iterations")
                ax.set_ylabel("Training Accuracy")
                ax.legend(loc='best')
                train_fig.show()

                if show_category_stats:
                    gen_category_acc_plots(its_sub, train_accs, val_accs)
            except ValueError:
                print("Learning curve unavailable")
                print(len(its_sub), len(train_accs[-1]), len(val_accs[-1]))

    input("Showing plots. Type anything to continue.")

    if train_accs:
        str_repr = "Final Training Accuracy Across All Categories: {}\n".format(train_accs[-1][-1]) + \
                   "Final L1 Training Accuracy: {}\n".format(train_accs[0][-1]) + \
                   "Final L2 Training Accuracy: {}\n".format(train_accs[1][-1]) + \
                   "Final L3 Training Accuracy: {}".format(train_accs[2][-1])    
            
        print(str_repr)
        print("-" * 30)
    if val_accs:
         str_repr = "Final Validation Accuracy Across All Categories: {}\n".format(val_accs[-1][-1]) + \
                    "Final L1 Validation Accuracy: {}\n".format(val_accs[0][-1]) + \
                    "Final L2 Validation Accuracy: {}\n".format(val_accs[1][-1]) + \
                    "Final L3 Validation Accuracy: {}".format(val_accs[2][-1])  
         print(str_repr)
    if test_data is not None:
        print("-" * 30)
        str_repr_test_acc = get_accuracy(model, testing_dataset, str_repr=True, device=device)
        print("Final Test Accuracy: {}".format(str_repr_test_acc))

    if ask and input("Save to file? [y/n] > ").lower().strip() == "y":
        dump_data(outfile, model)
        if loss_fig is not None:
            loss_fig.savefig(kwargs.get("loss_fig_out", "loss_curve.png"))
        if train_fig is not None:
            train_fig.savefig(kwargs.get("train_fig_out", "train_curve.png"))


def _tot_params_helper(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_param_sizes(model):
    ret = dict()
    ret["RNN"] = _tot_params_helper(model.rnn)
    for i in range(len(model.embedding_fcs)):
        ret[f"emb_fc_{i+1}"] = _tot_params_helper(model.embedding_fcs[i])
        ret[f"clf_fc_{i+1}"] = _tot_params_helper(model.classifier_fcs[i])
    ret["total"] = sum(n_params for n_params in ret.values())
    return ret


def find_example(model, l1=True, l2=True, l3=True,  matches=True, dataset="test"):
    """
    Finds an example, looking for either a correct classification or an incorrect classification.

    :param model: The Hierarchical RNN thats been trained.
    :param l1: Find a correct/incorrect classification at the l1 level.
    :param l2: Find a correct/incorrect classification at the l2 level.
    :param l3: Find a correct/incorrect classification at the l3 level.
    :param matches: True if looking for a correct classification, False for an incorrect classification.
    :param dataset: The data set to look for examples in; by default, the testing set.
    :return: A wiki summary, the predicted labels, the actual labels.
    """

    for docs, labs in csv_pt_pairs(dataset):
        summary, doc_embedding = docs
        label_text, label_emb = labs

        output = model(doc_embedding)

        corrects = []
        for i in range(len(model.output_sizes)):
            pred = output[i].max(1, keepdim=True)[1]
            corrects.append(pred.eq(label_emb[i].view_as(pred)))

        is_example = True

        for i in range(3):
            # want not (l_i => corrects[i] == matches),
            # which is equivalent to l_i and corrects[i] != matches
            if [l1, l2, l3][i] and corrects[i] != matches:
                is_example = False

        if is_example:
            return summary, label_emb, output

if __name__ == "__main__":
    # print(find_example(None))
    pass
