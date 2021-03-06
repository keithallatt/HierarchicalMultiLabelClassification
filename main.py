"""
main.py

This is where model training is done. Note that train_model() in utilities.py is where our main
training function is defined.
"""
from unicodedata import category
import torch
import pickle

from utilities import train_model, get_param_sizes, generate_hyperparameters, \
    find_best_parameters, find_correct_classifications
from model import DBPedia, HierarchicalRNN, BaselineMLP


if __name__ == "__main__":
    file_fmt = "processed_data/DBPEDIA_{split}_{var}.pt"
    small_file_fmt = "processed_data/DBPEDIA_{split}_small_{var}.pt"
    l2_l1_file_fmt = "processed_data/DBPEDIA_l2_l1_Agent_{var}.pt"

    # how much data to load
    train_obs = 40000
    val_obs = 36003
    test_obs = 60794
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

    # uncomment model to use

    model = HierarchicalRNN(
        input_size=768, emb_size=100, output_sizes=(9, 70, 219)
    ).to(device)

    # model = BaselineMLP(
    #     input_size=768, output_sizes=(9, 70, 219)
    # ).to(device)

    '''
    Model Checkpointing Notes
    
    For checkpointing/saving a model, set checkpoint_path to the directory for 
    model checkpoint to be saved i.e. "./checkpoint/".

    To load a checkpointed model and train on it set: load_checkpoint: True and 
    load_checkpoint_path = <path-to-checkpointed-model> ex. "./checkpoint/model_233939_0"

    You can also set the checkpoint frequency by setting checkpoint_frequency = <x epochs>
    By default it is set to 4 epochs.

    Model checkpoints will be saved in a directory within your checkpoint path
    '''
    train_opts = {
        "calc_acc_every": 4,
        "num_epochs": 100,
        "checkpoint_path": './checkpoints/',
        # "load_checkpoint": True,
        # "load_checkpoint_path": './checkpoints/18-04-2022 11:19:48/model_18-04-2022 11:19:48_20',
        "optimizer": "adam",
        "tf_init": 0,
        "tf_decay": 0.5
    }

    # param_sizes = get_param_sizes(model)

    '''
    Toggle save_imgs to True to save imgs to an imgs directory which will be created if it doesn't exist: imgs/
    '''
    # hp = find_best_parameters(20, model, train, val, test, device)
    ho = {'calc_acc_every': 4, 'batch_size': 64, 'learning_rate': 0.001020977066089074,
          'weight_decay': 0.000, 'momentum': 0.000, 'num_epochs': 14}
    train_opts.update(ho)

    # train_model(model, train, val, test,
    #             device=device, train_opts=train_opts, show_plts=False, save_imgs=False)

    # needed to load pickle file.

    # from data_cleaning import WordIdMapping
    # options = {
    #     'load_checkpoint': True,
    #     'load_checkpoint_path': "./checkpoints/18-04-2022 16:20:40/model_18-04-2022 16:20:40_14"
    # }
    # data_mapping = pickle.load(open('./processed_data/mapping.pkl', 'rb'))
    # res = find_correct_classifications(model, opts=options, device=device, word_mapping=data_mapping)
    # print(res)
