"""
model.py

Defines:

dataset object: DBPedia
decoder implementations: HierarchicalRNN, BaselineMLP
encoder implementation: EncoderRNN

Note that our primary/final model uses the HierarchicalRNN decoder paired with a BERT encoder used
in process_documents() in data_cleaning.py.

Author(s): Renato Zimmermann, Brandon Jaipersaud 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchtext
from utilities import make_layer_mult_mlp, to_one_hot

glove = torchtext.vocab.GloVe(name="6B", dim=50)


class DBPedia(Dataset):
    """Dataset object for retrieving document embeddings and processed labels
    from previously-processed matrices.
    """

    def __init__(self, emb_file, lab_file, load_func=torch.load, obs=None):
        """Initialize DBPedia dataset object.

        Arguments:
            emb_file: path to the embedding matrix.
            lab_file: path to the label matrix.
        """

        super(Dataset, self).__init__()
        self.embs = load_func(emb_file)[:obs]
        self.labs = load_func(lab_file)[:obs]

    def __len__(self):
        """Length as the number of embedded points."""
        return self.embs.shape[0]

    def __getitem__(self, idx):
        """Get data from an embedded document and label."""
        return self.embs[idx], self.labs[idx]



class HierarchicalRNN(nn.Module):
    """Proposed hierarchical prediction RNN."""

    def __init__(self, input_size: int, emb_size: int, output_sizes: tuple,
                 emb_size_mults: tuple = (1,), clf_size_mults: tuple = (1,),
                 rnn_size_mult: int = 1, rnn_n_hidden: int = 1):
        """Initialize hierarchical RNN.

        Arguments:
            input_size (int): size of document embedding inputs.
            emb_size (int): constant embedding size. This is the size of the
                inputs to the RNN.
            output_sizes (tuple): size of each output category.
            emb_size_mults (tuple): size of the the embedding layers as a
                multiple of the size of the last layer.
            clf_size_mults (tuple): size of the the classifier layers as a
                multiple of the size of the last layer.
            rnn_size_mult (int): size of the RNN hidden layer as a multiple of
                the embedding size.
            rnn_n_hidden (int): number of RNN hidden layers.
        """

        super(HierarchicalRNN, self).__init__()

        self.output_sizes = output_sizes

        self.embedding_fcs = nn.ModuleList()
        for fc_in_size in (input_size,) + output_sizes[:-1]:
            self.embedding_fcs.append(
                make_layer_mult_mlp(fc_in_size, emb_size, emb_size_mults)
            )

        self.rnn_hidden_size = emb_size*rnn_size_mult
        self.rnn = nn.GRU(emb_size,
                          self.rnn_hidden_size,
                          rnn_n_hidden)

        self.classifier_fcs = nn.ModuleList()
        for fc_out_size in output_sizes:
            self.classifier_fcs.append(
                make_layer_mult_mlp(emb_size, fc_out_size, clf_size_mults)
            )

    def forward(self, doc_emb: torch.tensor, true_labs=None):
        """Forward pass on data.

        Arguments:
            doc_emb (torch.tensor): document embeddings.
            true_labs [optional]: True labels of the document embeddings, used
                for teacher forcing. If specified, the true labels are fed to
                each stage instead of the last stage's predictions. If None,
                the label prediction from the last layer is fed to the next
                stage.
        """

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if true_labs is not None:
            assert true_labs.shape[1] == len(self.embedding_fcs)

        in_data = doc_emb
        last_hidden = None
        preds = list()
        for i in range(len(self.output_sizes)):
            emb = self.embedding_fcs[i](in_data)
            # add sequence length dimension
            emb = torch.unsqueeze(emb, dim=0)
            out, hid = self.rnn(emb, last_hidden)
            # remove sequence length dimension
            clf = self.classifier_fcs[i](out)
            clf = torch.squeeze(clf, dim=0)

            preds.append(clf)
            in_data = torch.argmax(clf, axis=1) if true_labs is None \
                      else true_labs[:,i]
            in_data = to_one_hot(in_data, d=self.output_sizes[i], device=device)
            last_hidden = hid

        return preds



'''
Below are 2 models (an encoder and a decoder) used for comparing against our main encoder-decoder model. 
They are discussed under the Benchmarking section of the README.md
'''


class BaselineMLP(nn.Module):
    """Baseline MLP decoder to compare against our main decoder:HierarchicalRNN ."""

    def __init__(self, input_size: int, output_sizes: tuple,
                 clf_size_mults: tuple = (1,)):
        """Initialize baseline model.

        input_size: size of input
        output_size: size of output
        clf_size_mults: size of hidden layers as a multiple of the size of the
            last layer.
        """

        super(BaselineMLP, self).__init__()

        self.output_sizes = output_sizes
        self.classifier_fcs = nn.ModuleList()

        for fc_out_size in output_sizes:
            self.classifier_fcs.append(
                make_layer_mult_mlp(input_size, fc_out_size, clf_size_mults)
            )

    def forward(self, doc_emb: torch.tensor):

        in_data = doc_emb
        preds = list()
        for clf_fc in self.classifier_fcs:

            clf = clf_fc(in_data)
            clf = torch.squeeze(clf, dim=0)

            preds.append(clf)

        return preds


class EncoderRNN(nn.Module):
    """Simple RNN encoder which uses GloVe embeddings + GRU for encoding articles.
       Used for comparison with BERT encoder.

       Inspired by CSC413 RNN Google Colab notebook: rnn_notebook.ipynb
    """

    def __init__(self, input_size=50, hidden_size=768):
        """Initialize model.
        """
        super(EncoderRNN, self).__init__()
        self.emb = nn.Embedding.from_pretrained(glove.vectors)
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        # Look up the embedding
        x = self.emb(x)
        # Forward propagate the RNN
        out, last_hidden = self.rnn(x)
        return last_hidden






if __name__ == "__main__":
    """Perform model debugging and testing."""

    import numpy as np
    from transformers import BertTokenizer, BertModel

    # pred_model = HierarchicalRNN(
    #     input_size=768, emb_size=100, output_sizes=(9, 70, 219)
    # )

    #print(pred_model)

    baseline_model = BaselineMLP(input_size=768, output_sizes=(9, 70, 219))

    print(baseline_model)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    emb_model = BertModel.from_pretrained("bert-base-uncased")

    raw_docs = [("We had given AM sentience. Inadvertently, of course, but "
                 "sentience nonetheless. But it had been trapped. AM wasn't "
                 "God, he was a machine. We had created him to think, but "
                 "there was nothing it could do with that creativity."),
                ("Without those weapons, often though he had used them "
                 "against himself, Man would never have conquered his world. "
                 "Into them he had put his heart and soul, and for ages they "
                 "had served him well. But now, as long as they existed, he "
                 "was living on borrowed time.")]

    doc = tokenizer(raw_docs, return_tensors='pt', padding=True)
    doc_emb = emb_model(**doc).pooler_output
    print(doc_emb.shape)

    # preds = pred_model(doc_emb)

    baseline_preds = baseline_model(doc_emb)
    #print(baseline_preds[2].shape)

    # for raw in raw_docs:
    #     raw_sents = raw.split('. ')
    #     sent = tokenizer(raw_sents, return_tensors='pt', padding=True)
    #     sent_emb = emb_model(**sent).pooler_output
    #     preds = pred_model(sent_emb)

    print("DONE")
