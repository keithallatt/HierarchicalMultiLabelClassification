# CSC413H5 Final Project

## Introduction

The goal of this project is to create a deep learning model to categorize topics in the form of Wikipedia
article summaries into 3 progressively finer sets of categories. 

## Model

Our model uses pretrained BERT encoders to encode chunks plain text into a more information dense tensor that the 
RNN (composed of Gated Recurrent Units) uses to transform variable length sequence into a fixed size. At this point,
a set of multilayer perceptrons use this fixed size input to predict the classification of each category level.


![model](readme_assets/model_diagram.png)

The various parameters are found in the Recurrent Neural Network and in the final Multilayer Perceptron. [](TODO_add_in_numbers)

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


