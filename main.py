"""
main.py

:)
"""
import torch

from utilities import train_model, get_param_sizes
from model import DBPedia, HierarchicalRNN


if __name__ == "__main__":

    file_fmt = "processed_data/DBPEDIA_{split}_{var}.pt"
    obs = 4000 # set this to some lower number when testing
    train_obs= 6000
    val_obs = 3000
    test_obs = 3000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train = DBPedia(file_fmt.format(split="train", var="embeddings"),
                    file_fmt.format(split="train", var="labels"),
                    obs=train_obs)
    val = DBPedia(file_fmt.format(split="val", var="embeddings"),
                    file_fmt.format(split="val", var="labels"),
                    obs=val_obs)
    test = DBPedia(file_fmt.format(split="test", var="embeddings"),
                    file_fmt.format(split="test", var="labels"),
                    obs=test_obs)

    model = HierarchicalRNN(
        input_size=768, emb_size=100, output_sizes=(9, 70, 219)
    ).to(device)

    train_opts = {
        "calc_acc_every": 5,
        "num_epochs": 30,
    }

    

    param_sizes = get_param_sizes(model)

    train_model(model, train, val, test, show_plts=True,
                device=device, train_opts=train_opts)
