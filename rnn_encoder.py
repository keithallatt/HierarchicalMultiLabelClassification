'''
An alternate encoder used for benchmarking against the main BERT encoder in data_cleaning.py

Author: Brandon Jaipersaud
'''

from data_cleaning import gen_from_data, WordIdMapping
import csv
import pickle

from typing import Union, Generator
from string import punctuation
from tqdm import tqdm

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext

from model import EncoderRNN

# filepaths, cross-platform compatible,
all_data = Path("./dbpedia_data/DBP_wiki_data.csv")
training_data = Path("./dbpedia_data/DBPEDIA_train.csv")
validation_data = Path("./dbpedia_data/DBPEDIA_val.csv")
testing_data = Path("./dbpedia_data/DBPEDIA_test.csv")


glove = torchtext.vocab.GloVe(name="6B", dim=50)
glove_emb = nn.Embedding.from_pretrained(glove.vectors)

DOC_LENGTH = 100
MAX_MEMORY = 10000

NUM_POINTS = 5000



'''
Truncate document to DOC_LENGTH words and use GloVe embeddings to encode each word
'''
def parse_doc(doc):
    
    parsed_doc = []
  
    # separate punctuations
    doc = doc.replace(".", " . ") \
                 .replace(",", " , ") \
                 .replace(";", " ; ") \
                 .replace("?", " ? ") \
                 .replace("\"", "")

   
    doc = doc.lower().split()
    parsed_doc_len = 0

    for w in doc:
        if w in glove.stoi:
            parsed_doc.append(glove.stoi[w])
            parsed_doc_len += 1
            if parsed_doc_len == DOC_LENGTH:
                break
   
    if len(parsed_doc) != DOC_LENGTH:
        return False 
    else:
        return parsed_doc


'''
Encodes the document using GloVe embeddings + GRU
'''
def convert_to_embedding(doc):
    doc = parse_doc(doc)
   
    if not doc:
        return False
    else:
        doc = torch.IntTensor(doc)
        emb = EncoderRNN()
        emb = emb(doc)
        return emb
        
       
'''
Analogous to process_documents in data_cleaning.py. The only difference is convert_to_embedding()
which encodes the document using GloVe embeddings + GRU
'''
def process_documents(data_files,
                      emb_suffix="_alt_embeddings_gru.pt",
                      lab_suffix="_alt_labels_gru.pt",
                      map_file="alt_mapping_gru.pkl"):

    
    map_data = WordIdMapping(3)

    
    for file_name in data_files:
        print(f"PROCESSING: {file_name}")

        file_name = Path(file_name)
        doc_gen = gen_from_data(file_name)
        emb_list, lab_list = [], []

        try:
            num_saved_points = 0
            emb_list_size = 0
            for doc, labs in tqdm(doc_gen):
                lab_ids = map_data.add_and_get(labs)
                emb = convert_to_embedding(doc) # shape = 1x768
                
                if torch.is_tensor(emb):
                    num_saved_points += 1
                    emb_list.append(emb.squeeze())
                    lab_list.append(torch.tensor(lab_ids))
                    emb_list_size += 1

                if emb_list_size == NUM_POINTS:
                    raise Exception
                    

        finally:
             print("Number of saved articles in {} is {}".format(file_name, num_saved_points))
             torch.save(torch.stack(emb_list), "processed_data/" + file_name.stem+emb_suffix)
             torch.save(torch.stack(lab_list), "processed_data/" + file_name.stem+lab_suffix)
             with open("processed_data/" + map_file, "wb") as f:
                pickle.dump(map_data, f)



if __name__ == '__main__':
   
    process_documents([f"./dbpedia_data/DBPEDIA_{name}.csv"
                      for name in ["train"]])
