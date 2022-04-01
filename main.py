"""
main.py

:)
"""
from pathlib import Path

from utilities import train_model
from model import DBPedia, HierarchicalRNN


if __name__ == "__main__":

    file_fmt = "processed_data/DBPEDIA_{split}_{var}.pt"
    train = DBPedia(file_fmt.format(split="train", var="embeddings"),
                    file_fmt.format(split="train", var="labels"))
    val = DBPedia(file_fmt.format(split="val", var="embeddings"),
                    file_fmt.format(split="val", var="labels"))
    test = DBPedia(file_fmt.format(split="test", var="embeddings"),
                    file_fmt.format(split="test", var="labels"))

    model = HierarchicalRNN(
        input_size=768, emb_size=100, output_sizes=(9, 70, 219)
    )

    config = {
    }

    train_model(model, train, val, test, **config)
