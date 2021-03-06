"""
data_cleaning.py

Load the data, extract the relevant information from the csv files to create a more easily parsable file.

The encoder portion of our model is in process_documents(). This is where the article gets transformed to a
BERT embedding.

Author(s): Keith Allatt, Renato Zimmermann
"""
import csv
import pickle

from typing import Union, Generator
from string import punctuation

from pathlib import Path
from tqdm import tqdm

import torch
from transformers import BertTokenizer, BertModel


# filepaths, cross-platform compatible,
all_data = Path("./dbpedia_data/DBP_wiki_data.csv")
training_data = Path("./dbpedia_data/DBPEDIA_train.csv")
validation_data = Path("./dbpedia_data/DBPEDIA_val.csv")
testing_data = Path("./dbpedia_data/DBPEDIA_test.csv")


training_data_small = Path("./dbpedia_data/DBPEDIA_train_small_l1.csv")
validation_data_small = Path("./dbpedia_data/DBPEDIA_train_small_l2.csv")
testing_data_small = Path("./dbpedia_data/DBPEDIA_train_small_l3.csv")


class WordIdMapping:
    """Class that creates multi-level word-id mappings used to create
    numeric ids for word labels in different levels.
    """

    def __init__(self, n_levels):
        """Initialize object."""
        self.n_levels = n_levels
        self.word_to_id = [dict() for _ in range(n_levels)]
        self.id_to_word = [list() for _ in range(n_levels)]

    def add_word(self, level, word):
        """Add a word to a given level. Do nothing if word is already
        present.
        """
        if word not in self.word_to_id[level]:
            self.word_to_id[level][word] = len(self.id_to_word[level])
            self.id_to_word[level].append(word)

    def add_set(self, words):
        """Add a set of words to each available level.
        """
        assert len(words) == self.n_levels
        for i, word in enumerate(words):
            self.add_word(i, word)

    def get_word_set(self, words):
        """Get the ids for a set of words."""
        return tuple([self.word_to_id[i][word]
                      for i, word in enumerate(words)])

    def add_and_get(self, words):
        """Add a set of words and get back their ids.
        """
        self.add_set(words)
        return self.get_word_set(words)

    def get_word_from_id(self, level, _id):
        """Get word represented by a given ID."""
        return self.id_to_word[level][_id]


def csv_file_generator(infile: Union[str, Path]) -> Generator[tuple, None, None]:
    """ Clean csv file contents and yield a text accompanied by its labels
    row by row.

    Used in the aggregate DBPedia dataset.
    """

    l1i, l2i, l3i = 1, 2, 3  # indices of l1, l2, and l3 category in all data file.

    # was useful for testing that the categories were being extracted properly from the csv file.
    # l1_category, l2_category, l3_category = [], [], []  # extract labels for each

    with open(infile, mode='r') as file:
        # reads file in, line by line.
        csv_file = csv.reader(file)
        for line in csv_file:
            # first row has the header, can disregard.
            if line[0] == "text":
                continue

            # remove unnecessary empty strings from end of list
            line = list(filter(lambda x: x, line))

            # how many of the first cells are actually the summary split by semicolons?
            num_groups = len(line) - 5  # if no semi's, only one, and thus it is good.
            doc = ";".join(line[:num_groups])
            line = [doc] + line[num_groups:]

            # get l1, l2, and l3 category
            l1 = line[l1i]
            l2 = line[l2i]
            l3 = line[l3i]

            # in all examples, these are the categories that are misplaced in l1
            # when semicolons appear in the article name. This will revert the
            # change made when the article name gets split in two and thus shifts the categories.
            if l1 in ["MusicalWork", "Comic", "Cartoon", "AmusementParkAttraction", "Software"]:
                # shift l1-l2 to l2-l3,
                l2, l3 = l1, l2
                # get the l1 cat from summary
                l1 = doc.split(";")[-1]
                # and remove it from summary
                doc = ";".join(doc.split(";")[:-1])

            # for ll, lc in [(l1, l1_category), (l2, l2_category), (l3, l3_category)]:
            #     if ll not in lc:
            #         lc.append(ll)

            yield doc, (l1, l2, l3)


def gen_from_data(infile: Union[str, Path]) -> Generator[tuple, None, None]:
    """Get contents of train, validateion or test data in the same format as
    the aggregate dataset.

    This function performs all necessary text cleaning.
    """

    with open(infile, mode='r') as file:
        # reads file in, line by line.
        csv_file = iter(file)
        next(csv_file)  # skip header
        for line in csv_file:
            line = line.split(',')
            # get l1, l2, and l3 category
            l3 = line[-1].strip()
            l2 = line[-2].strip()
            l1 = line[-3].strip()
            doc = ','.join(line[:-3]).strip()

            yield doc, (l1, l2, l3)


def process_documents(data_files,
                      emb_suffix="_embeddings.pt",
                      lab_suffix="_labels.pt",
                      map_file="mapping.pkl"):
    """Get text and labels from raw documents downloaded from Kaggle and save
    three objects from them:
    1. An embedding matrix whose rows are the BERT embedding of each document.
    2. A label matrix whose rows are the numeric ID combination for each word
        label.
    3. A word-id mapping object for translating ids used in (2).
    """
    rm_punct = str.maketrans('', '', punctuation)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    map_data = WordIdMapping(3)

    for file_name in data_files:
        print(f"PROCESSING: {file_name}")

        file_name = Path(file_name)
        doc_gen = gen_from_data(file_name)
        emb_list, lab_list = [], []

        for doc, labs in tqdm(doc_gen):
            doc = doc.translate(rm_punct)[:510]
            tok_in = tokenizer(doc, return_tensors="pt", padding=True)
            emb = model(**tok_in).pooler_output.squeeze().detach()

            lab_ids = map_data.add_and_get(labs)

            emb_list.append(emb.squeeze())
            lab_list.append(torch.tensor(lab_ids))

        torch.save(torch.stack(emb_list), "processed_data/" + file_name.stem+emb_suffix)
        torch.save(torch.stack(lab_list), "processed_data/" + file_name.stem+lab_suffix)

    # "processed_data/" +
    with open(map_file, "wb") as f:
        pickle.dump(map_data, f)


def csv_pt_pairs(dataset, assert_tests=False):
    """
    Generate document-label pairs with their associated embeddings. Matches the embeddings so
    future analytics does not rely on re-encoding data, as BERT Encodings take a while to perform.

    Yields (document summary, document embedding) and (label text, label encoding)
    """
    processed_embeddings = torch.load(f"./processed_data/DBPEDIA_{dataset}_embeddings.pt")
    processed_labels = torch.load(f"./processed_data/DBPEDIA_{dataset}_labels.pt")
    doc_gen = gen_from_data(f"./dbpedia_data/DBPEDIA_{dataset}.csv")

    if assert_tests:
        rm_punct = str.maketrans('', '', punctuation)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")

        i = 0
        for doc, labels in tqdm(doc_gen):
            # do an integrity check on all the documents and labels.
            doc = doc.translate(rm_punct)[:510]
            tok_in = tokenizer(doc, return_tensors="pt", padding=True)
            emb = model(**tok_in).pooler_output.squeeze().detach()
            emb_squeeze = emb.squeeze()
            sum_diff = float(torch.sum(torch.abs(emb_squeeze - processed_embeddings[i])).detach())
            assert sum_diff <= 0.015, f"Sum difference too high. {i=}"  # test set had max diff of 0.008
            i += 1

        return

    i = 0
    for doc, labels in doc_gen:
        doc_emb = processed_embeddings[i]
        lab_emb = processed_labels[i]

        yield (doc, doc_emb), (labels, lab_emb)
        i += 1


if __name__ == '__main__':
    # csv_pt_pairs("test", assert_tests=True)
    process_documents([f"./dbpedia_data/DBPEDIA_{name}.csv"
                      for name in ["train_small_l1"]])

    # process_documents([f"./dbpedia_data/DBPEDIA_{name}.csv"
    #                    for name in ["l2_l1_Agent_dep"]])
    # process_documents([f"./dbpedia_data/DBPEDIA_{name}.csv"
    #                    for name in ["train", "val", "test"]])

    # for d, ls in g:
    #     print(d[:min(len(d), 100)], ls)
