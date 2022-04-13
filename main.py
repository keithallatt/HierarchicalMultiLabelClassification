"""
main.py

:)
"""
from unicodedata import category
import torch

from utilities import train_model, get_param_sizes
from model import DBPedia, HierarchicalRNN




if __name__ == "__main__":

  

    file_fmt = "processed_data/DBPEDIA_{split}_{var}.pt"
    small_file_fmt = "processed_data/DBPEDIA_{split}_small_{var}.pt"
    l2_l1_file_fmt = "processed_data/DBPEDIA_l2_l1_Agent_{var}.pt"


    obs = 4000 # set this to some lower number when testing
    train_obs= 15000
    val_obs = 20000
    test_obs = 20000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Using device {}".format(device))

    # Uncomment train based on what training set you want to use

    # train = DBPedia(small_file_fmt.format(split="train", var="embeddings", category="l2"),
    #                 small_file_fmt.format(split="train", var="labels", category="l2"),
    #                 obs=train_obs)

    # train = DBPedia(l2_l1_file_fmt.format(var="embeddings"),
    #                 l2_l1_file_fmt.format(var="labels"),
    #                 obs=train_obs)
       
    
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
        "calc_acc_every": 4,
        "num_epochs": 12,
        "checkpoint_path" : "./checkpoint/",
       
    }

    '''
     "load_checkpoint" : True,
     "load_checkpoint_path": "./checkpoint/model_233939_0"
    '''

    

    param_sizes = get_param_sizes(model)

    train_model(model, train, val, test, show_plts=True,
                device=device, train_opts=train_opts)
