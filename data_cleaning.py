"""
data_cleaning.py

Load the data, extract the relevant information from the csv files to create a more easily parsable file.

Author(s): Keith Allatt,
"""
from typing import Union, Generator

from pathlib import Path
import csv

# filepaths, cross-platform compatible,
all_data = Path("./dbpedia_data/DBP_wiki_data.csv")
training_data = Path("./dbpedia_data/DBPEDIA_train.csv")
validation_data = Path("./dbpedia_data/DBPEDIA_val.csv")
testing_data = Path("./dbpedia_data/DBPEDIA_test.csv")


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


if __name__ == '__main__':
    g = csv_file_generator(all_data)
    for d, ls in g:
        print(d[:min(len(d), 100)], ls)
