"""
data_cleaning.py

Load the data, extract the relevant information from the csv files to create a more easily parsable file.

Author(s): Keith Allatt,
"""
from typing import Union, Tuple

import utilities
from pathlib import Path
import csv

# filepaths, cross-platform compatible,
all_data = Path("./dbpedia_data/DBP_wiki_data.csv")
training_data = Path("./dbpedia_data/DBPEDIA_train.csv")
validation_data = Path("./dbpedia_data/DBPEDIA_val.csv")
testing_data = Path("./dbpedia_data/DBPEDIA_test.csv")


def clean_csv_into_pickle(infile: Union[str, Path], outfile: Union[str, Path] = None) -> Tuple[list, list, list]:
    """ Clean csv file into a pickle file for future use """
    if outfile is None:
        # infer outfile path
        outfile = infile.with_name(infile.stem + "_cleaned.pickle")

    l1i, l2i, l3i = 1, 2, 3  # indices of l1, l2, and l3 category in all data file.
    l1_category, l2_category, l3_category = [], [], []  # extract labels for each
    cleaned_set = []

    with open(infile, mode='r') as file:
        csv_file = csv.reader(file)
        for line in csv_file:
            if line[0] == "text":
                continue
            # remove unnecessary empty strings from end of list
            line = list(filter(lambda x: x, line))
            # how many of the first cells are actually the summary split by semicolons?
            num_groups = len(line) - 5  # if no semi's, only one, and thus it is good.
            summary = ";".join(line[:num_groups])
            line = [summary] + line[num_groups:]

            # get l1, l2, and l3 category
            l1 = line[l1i]
            l2 = line[l2i]
            l3 = line[l3i]

            # in all examples, these are the categories that are misplaced in l1
            # when semicolons appear in the article name.
            if l1 in ["MusicalWork", "Comic", "Cartoon", "AmusementParkAttraction", "Software"]:
                l3 = l2
                l2 = l1
                # get the l1 cat from summary
                l1 = summary.split(";")[-1]
                # and remove it from summary
                summary = ";".join(summary.split(";")[:-1])

            for l, lc in [(l1, l1_category), (l2, l2_category), (l3, l3_category)]:
                if l not in lc:
                    lc.append(l)

            cleaned_set.append((summary, l1, l2, l3))

    utilities.dump_data(outfile, cleaned_set)
    return l1_category, l2_category, l3_category


if __name__ == '__main__':
    l1c, l2c, l3c = clean_csv_into_pickle(all_data)
    assert list(map(len, [l1c, l2c, l3c])) == [9, 70, 219], "Categories have wrong number of elements."

