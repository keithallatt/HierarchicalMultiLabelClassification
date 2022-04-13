import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from utilities import make_layer_mult_mlp, to_one_hot


class DBPedia(Dataset):

    def __init__(self, emb_file, lab_file, load_func=torch.load, obs=None):

        super(Dataset, self).__init__()
        self.embs = load_func(emb_file)[:obs]
        self.labs = load_func(lab_file)[:obs]

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
            in_data = to_one_hot(in_data, d=self.output_sizes[i], device=clf.get_device())
            last_hidden = hid

        return preds


if __name__ == "__main__":

    import numpy as np
    from transformers import BertTokenizer, BertModel

    pred_model = HierarchicalRNN(
        input_size=768, emb_size=100, output_sizes=(9, 70, 219)
    )

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

    preds = pred_model(doc_emb)

    for raw in raw_docs:
        raw_sents = raw.split('. ')
        sent = tokenizer(raw_sents, return_tensors='pt', padding=True)
        sent_emb = emb_model(**sent).pooler_output
        preds = pred_model(sent_emb)

    print("DONE")
