# CSC413H5 Final Project

## Introduction

The goal of this project is to create a deep learning model to categorize topics in the form of Wikipedia
article summaries into 3 progressively finer sets of categories.

## Model

We tackle the issue of hierarchical classification using an encoder-decoder
architecture composed of a transformer-based encoder and RNN-based decoder.
An illustration of the model can be seen below.

![model](readme_assets/model_diagram.png)

### Encoder

Our model used a pre-trained BERT transformer to encode input text into a
single dense tensor. The chosen implementation is the `bert-base-uncased` by
Hugging Face, which was pretrained on the BookCorpus and English Wikipedia
datasets. The whole model contains 109,482,240 pre-trained parameters, none of
which are further tuned as part of our application.

### Decoder

Our decoder model takes in a text embeddings and generates three separate
predictions, each corresponding to one level of class specificity. Each level
has its own MLP to used to project the last level's output onto the input
dimension of an RNN; we call these embedding MLPs. Much like other sequence
generation tasks, the first level takes in the encoder output (the text embedding).
Each level also has its own MLP used to project the RNN output onto a class
prediction for that specific level; these are called classifier MLPs. In
between these MLPs, we utilize a common RNN body composed of Gated Recurrent
Units to enable weight sharing and information passing from one level to the next.
We chose an RNN instead of a Transformer body for our decoder as the former
proved to be more flexible for generating sequences with different input and output
dimensions. Additionally, we would like for encourage a certain degree of
information decay over the network to embody how each level should mosty use
information of the previous level.

The parameter distribution of each decoder component is a follows:

## Data

We used the [DBPedia Classes Kaggle dataset](https://www.kaggle.com/datasets/danofer/dbpedia-classes) for training
and evaluating our model. It contains 342780 data points, consisting of a Wikipedia article's summary, and it's
classification in 3 levels of categories, ranging from broad categories such as 'Agent' or 'Place', to specific
categories such as 'Earthquake' or 'SolarEclipse'.

## Training

*To be filled in*

## Results

*To be filled in*


#### Authored by: Keith Allatt, Brandon Jaipersaud, David Chen, Renato Zimmermann


