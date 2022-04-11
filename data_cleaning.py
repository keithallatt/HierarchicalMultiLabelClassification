"""
data_cleaning.py

Load the data, extract the relevant information from the csv files to create a more easily parsable file.

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
debug_data = Path("./dbpedia_data/DBPEDIA_debug.csv")

'''
Assigning every li class label an integer
{w1:25 w2 w3  }  { w4 w5  }  { w6 }
[]             []           []
'''
class WordIdMapping:

    def __init__(self, n_levels):

        self.n_levels = n_levels
        # maps word in a given level to its integer id
        self.word_to_id = [dict() for _ in range(n_levels)]
        # array[0] = array of l1 labels 
        self.id_to_word = [list() for _ in range(n_levels)]

    def add_word(self, level, word):
        if word not in self.word_to_id[level]:
            self.word_to_id[level][word] = len(self.id_to_word[level]) # assign unique id to label b/w 0 and Li_size
            self.id_to_word[level].append(word)

    def add_set(self, words):
        assert len(words) == self.n_levels
        for i, word in enumerate(words):
            self.add_word(i, word)

    def get_word_set(self, words):
        return tuple([self.word_to_id[i][word]
                      for i, word in enumerate(words)])

    # labels (l1, l2, l3)
    def add_and_get(self, words):
        self.add_set(words) # update level, labels mapping
        return self.get_word_set(words) # return tuple of label ids (25, 15, 48)


def csv_file_generator(infile: Union[str, Path]) -> Generator[tuple, None, None]:
    """ Clean csv file contents and yield row by row. """

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
    """Get contents of train, val or test data
        Generator that steps through data
    """

    with open(infile, mode='r') as file:
        # reads file in, line by line.
        csv_file = iter(file)
        next(csv_file) # skip header
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

    rm_punct = str.maketrans('', '', punctuation)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    
    map_data = WordIdMapping(3)

    # with open(data_files[0], mode='r') as file:
    #     csv_file = csv.reader(file)
    #     i = 0
    #     for line in csv_file:
    #         print(line)
    #         i+=1
    #         if i == 2:
    #             break


    

    for file_name in data_files:

        print(f"PROCESSING: {file_name}")

        file_name = Path(file_name)
        doc_gen = gen_from_data(file_name) # iterator through data
        emb_list, lab_list = [], []  # array of embeddings, array of label tuples (ground truth) ex. (25, 48, 18)

        for doc, labs in tqdm(doc_gen):

            doc = doc.translate(rm_punct)[:510]
            #print(doc)
           
            
            tok_in = tokenizer(doc, return_tensors="pt", padding=True)
            #print(tok_in)
            
            emb = model(**tok_in).pooler_output.squeeze().detach()
            print(emb.shape)  # 768
            
            lab_ids = map_data.add_and_get(labs)
           

            emb_list.append(emb.squeeze())
            lab_list.append(torch.tensor(lab_ids))
        
        
        torch.save(torch.stack(emb_list), file_name.stem+emb_suffix)
        torch.save(torch.stack(lab_list), file_name.stem+lab_suffix)
       

    # save label mappings to a file
    with open(map_file, "wb") as f:
       pickle.dump(map_data, f)

if __name__ == '__main__':
    process_documents([f"./dbpedia_data/DBPEDIA_{name}.csv"
                       for name in ["debug", "train", "val", "test"]])
                       
    # for d, ls in g:
    #     print(d[:min(len(d), 100)], ls)
