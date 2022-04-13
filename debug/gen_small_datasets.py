"""
gen_small_datasets.py

Generates three datasets to test the ability for the model to predict l1,l2,l3 classes.
Does it learn the l1-l2-l3 dependencies correctly?

l1: can the model predict l1 classes

given l1 class (i.e agent), can the model predict l2 classes?

given l1,l2 classes can the model predict l3 classes?



Author(s): Brandon Jaipersaud
"""

import csv
import pickle



'''
Agent -> 

check how well model performs with predicting different types of agents

filter train, val, test to only contain agents
'''
def gen_l2(file_path, num_instances=1000, l1_label="Agent"):
    small_l2_path = "../DBPEDIA_train_small_l2.csv"
    l2_labels = {}

    with open(small_l2_path, mode='w') as small_l1_file:
        small_l1_file.write("text,l1,l2,l3")

        with open(file_path, mode='r') as file:
            # reads file in, line by line.
            csv_file = iter(file)
            next(csv_file) # skip header
            for line in csv_file:
                parse_line = line.split(',')
                # get l1, l2, and l3 category
                l3 = parse_line[-1].strip()
                l2 = parse_line[-2].strip()
                l1 = parse_line[-3].strip()
                #print(l1)
                if (l1 == l1_label):
                    small_l1_file.write(line)
                    num_instances -= 1
                if num_instances == 0:
                    return

                    

def check_file_statistics(file_path):

    labels = [{},{},{}]
    category_counts = [0, 0, 0]
    size = 0

    with open(file_path, mode='r') as file:
        # reads file in, line by line.
        csv_file = iter(file)
        next(csv_file) # skip header
        for line in csv_file:
            size += 1
            parse_line = line.split(',')
            # get l1, l2, and l3 category
            l3 = parse_line[-1].strip()
            l2 = parse_line[-2].strip()
            l1 = parse_line[-3].strip()

            for count, l in enumerate([l1, l2, l3]):
                if l not in labels[count]:
                    category_counts[count] += 1
                    labels[count][l] = True
                

    print("L1 distinct categories: {}".format(category_counts[0]))
    print("L2 distinct categories: {}".format(category_counts[1]))
    print("L3 distinct categories: {}".format(category_counts[2]))

    print("Number of data points {}".format(size))




'''
Generate 3 small datasets, each of which is tailored to overfit on L1,L2,L3 classification respectively.

Ex. small_l2.csv will contain 1-2 points for each L2 category

:param num_instances: How many class repetitions to include. Ex. if num_instances=2, then 2 of 
each class will be used.

'''
def gen_small_datasets(file_path, num_instances=2):
    small_l1_path = "../dbpedia_data/DBPEDIA_train_small_l1.csv"
    small_l2_path = "../dbpedia_data/DBPEDIA_train_small_l2.csv"
    small_l3_path = "../dbpedia_data/DBPEDIA_train_small_l3.csv"
    
    label_counts = {}
    with open(small_l1_path, mode='w') as small_l1_file:
        with open(small_l2_path, mode='w') as small_l2_file:
            with open(small_l3_path, mode='w') as small_l3_file:
                files = [small_l1_file, small_l2_file, small_l3_file]
                for f in files:
                    f.write("text,l1,l2,l3\n")
            
                with open(file_path, mode='r') as file:
                    # reads file in, line by line.
                    csv_file = iter(file)
                    next(csv_file) # skip header
                    for line in csv_file:
                        parse_line = line.split(',')
                        # get l1, l2, and l3 category
                        l3 = parse_line[-1].strip()
                        l2 = parse_line[-2].strip()
                        l1 = parse_line[-3].strip()

                        labels = [l1, l2, l3]
                        for count,l in enumerate(labels):
                            if l not in label_counts:
                                label_counts[l] = 0
                            
                            if label_counts[l] < num_instances:
                                label_counts[l] += 1 
                                files[count].write(line)





if __name__ == '__main__':
    train_path = "../dbpedia_data/DBPEDIA_train.csv"
    gen_small_datasets(train_path)
    check_file_statistics(train_path)
    check_file_statistics("../dbpedia_data/DBPEDIA_train_small_l1.csv")
    check_file_statistics("../dbpedia_data/DBPEDIA_train_small_l2.csv")
    check_file_statistics("../dbpedia_data/DBPEDIA_train_small_l3.csv")