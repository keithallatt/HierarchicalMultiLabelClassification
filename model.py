import os

from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from pathlib import Path

from utilities import make_layer_mult_mlp, estimate_accuracy


class DBPedia(Dataset):

    def __init__(self, emb_file, lab_file, load_func=torch.load, obs=None):
        
        super(Dataset, self).__init__()
        self.embs = load_func(emb_file)[:obs] # array of document embeddings
        self.labs = load_func(lab_file)[:obs] # array of corresponding labels: (l1,l2,l3) for each doc

    def __len__(self):
        return self.embs.shape[0]

    def __getitem__(self, idx):
        return self.embs[idx], self.labs[idx]


class HierarchicalRNN(nn.Module):

    def __init__(self, input_size: int, emb_size: int, output_sizes: tuple,
                 emb_size_mults: tuple = (1,), clf_size_mults: tuple = (1,),
                 rnn_size_mult: int = 1, rnn_n_hidden: int = 1):

        super(HierarchicalRNN, self).__init__()

        self.output_sizes = output_sizes

        self.embedding_fcs = nn.ModuleList()
        for fc_in_size in (input_size,) + output_sizes[:-1]:  # (768, 9, 70, 219)
            self.embedding_fcs.append(
                make_layer_mult_mlp(fc_in_size, emb_size, emb_size_mults)
            )

        self.rnn_hidden_size = emb_size*rnn_size_mult
        self.rnn = nn.RNN(emb_size,
                          self.rnn_hidden_size,
                          rnn_n_hidden)

        self.classifier_fcs = nn.ModuleList()
        for fc_out_size in output_sizes:
            self.classifier_fcs.append(
                make_layer_mult_mlp(emb_size, fc_out_size, clf_size_mults)
            )

    # doc = 1x748
    def forward(self, doc_emb: torch.tensor):

        in_data = doc_emb
        last_hidden = None
        preds = list()
        for emb_fc, clf_fc in zip(self.embedding_fcs, self.classifier_fcs):

            emb = emb_fc(in_data)
            
            # add sequence length dimension
            emb = torch.unsqueeze(emb, dim=0)
            # [batch_size, seq_len, repr_dim]
            out, hid = self.rnn(emb, last_hidden) # gets passed a single document embedding, not a sequence?
            # remove sequence length dimension
            clf = clf_fc(out)
            clf = torch.squeeze(clf, dim=0)

            preds.append(clf) 
            in_data = clf
            last_hidden = hid

        return preds
        

def train(model, train_data, valid_data, batch_size=32, num_epochs=7,  
          weight_decay=0.0, learning_rate=0.01, momentum=0,
          device="cpu", checkpoint_path="checkpoints"):

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=momentum,
                          weight_decay=weight_decay)

    loss_log = {
        "iter": [],
        "loss": []
    }

    acc_log = {
        "epoch": [],
        "train_acc": [],
        "valid_acc": []
    }

    start_time = datetime.now().strftime("%H%M%S")
    if checkpoint_path is not None and not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    it = 0
    for epoch in range(num_epochs):
    
        for doc_embs, labels in train_loader:

            it += len(labels)

            doc_embs, labels = doc_embs.to(device), labels.to(device)
            preds = model(doc_embs)
            loss = torch.sum((criterion(pred, label) 
                              for pred, label in zip(preds, labels)))
            doc_embs.detach(), labels.detach()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_log["iter"].append(it)
            loss_log["loss"].append(loss.item())

        train_acc = estimate_accuracy(model, train_data)
        valid_acc = estimate_accuracy(model, valid_data)

        print(f"Epoch: {epoch} | "
              f"Train Acc: {train_acc} | "
              f"Val Acc: {valid_acc}")

        acc_log["epoch"].append(epoch)
        acc_log["train_acc"].append(train_acc)        
        acc_log["valid_acc"].append(valid_acc)

        if checkpoint_path is not None:
            torch.save(model.state_dict(), 
                       checkpoint_path + f"model_{start_time}_{epoch}")

    _, axs = plt.subplots(nrows=2, ncols=1, figsize=(10,10))

    axs[0].plot("iter", "loss", data=loss_log)
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Loss")

    axs[1].plot("epoch", "train_acc", data=acc_log, 
                label="Train Accuracy")
    axs[1].plot("epoch", "valid_acc", data=acc_log,
                label="Validation Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")



if __name__ == "__main__":

    import numpy as np
    from transformers import BertTokenizer, BertModel

    emb_file = "processed_data/DBPEDIA_test_embeddings.pt"
    labels_file = "processed_data/DBPEDIA_test_labels.pt"

    train_test = DBPedia(emb_file,
                    labels_file,
                    obs=3000)

    # input size = BERT embedding size
    # 
    pred_model = HierarchicalRNN(
        input_size=768, emb_size=100, output_sizes=(9, 70, 219)
    )

    print(pred_model)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    emb_model = BertModel.from_pretrained("bert-base-uncased")

    raw_docs = [("We had given AM sentience. Inadvertently, of course, but "
                 "sentience nonetheless. But it had been trapped. AM wasn't "
                 "God, he was a machine. We had created him to think, but "
                 "there was nothing it could do with that creativity.")
                ]

    doc = tokenizer(raw_docs, return_tensors='pt', padding=True)
    doc_emb = emb_model(**doc).pooler_output

    preds = pred_model(doc_emb)
    # for raw in raw_docs:
    #     raw_sents = raw.split('. ')
    #     sent = tokenizer(raw_sents, return_tensors='pt', padding=True)
    #     sent_emb = emb_model(**sent).pooler_output
    #     preds = pred_model(sent_emb)
    
    print("DONE")
