# CSC413H5 Final Project

## Introduction


Our deep learning model will perform hierarchical, multi-label classification to articles. Note that this is different from multi-class classification which assigns a *single* label/class to an input which is chosen from a group of multiple labels/classes. In particular, given an article as input, our model will assign 3 class labels to it, each of which come from a set of labels: L1, L2 and L3 respectively.  Li contains broader labels than Lj where i < j.  To illustrate, our model may assign labels: {Place, Building, HistoricalBuilding} to an article where Place ∈ L1,  Building ∈ L2 and HistoricalBuilding ∈ L3.  Note that Building is a type of Place while HistoricalBuilding is a type of Building.  Tagging an unstructured article with these 3 labels serves to categorize and add some structure to it.



## Model

### Overview

We tackle the hierarchical, multi-label classification task using an encoder-decoder 
architecture composed of a transformer-based encoder and a GRU-based decoder. 



### Design Decisions



#### Using BERT as our Encoder

Our choice of encoder is a pre-trained BERT transformer as outlined in [1]. Our chosen implementation is the `bert-base-uncased` by
Hugging Face, which was pretrained on the BookCorpus and English Wikipedia datasets. BERT is a state-of-the-art encoder used for NLP based tasks. One of its distinguishing features is its bidirectional language modelling. Most other transformers are unidirectional. Recall the 4-gram language model developed in CSC413 A1. This model was trained to predict the next word given the previous three words. Thus, it is unidirectional. In contrast, BERT uses both left and right context when encoding sequences which will make it much better at summarizing and learning the "gist" of an article.


The input to our model are articles which are very long sequences. Using a traditional RNN would be a poor choice of encoder since it would fail to effectively summarize and extract the important parts of the article. The attention mechanism of a transformer was designed to solve this issue. Furthermore, the articles in our training set are Wikipedia articles. Since `bert-base-uncased` was also trained on Wikipedia articles, it makes this model an even better candidate to use as an encoder. 



### Model Diagram and Forward Pass

The computation graph of our model can be seen below:

<img src="readme_assets/computation_graph.png" alt="computation_graph"/>



#### Encoder

Our encoder model takes in a single article, $x$ as input. The article must go through a pre-processing stage which does two things: First, punctuation is stripped from the article and the first 510 article characters are taken. Second, the truncated article is passed to a BERT Tokenizer which converts the article to a format acceptable by the BERT transformer. This is outlined in more detail in the Data Transformation section of the report. The tokenized article is then passed to the BERT Transformer which outputs an article embedding of shape 1x768. Intuitively, this represents a summarized version of the article.


#### Decoder

The decoder portion of our model takes in the encoded article of shape 1x768. It consists of three layers: One for predicting L1, L2 and L3 labels respectively. Each layer: Layer i begins with an MLP which we call the *embedding MLP*. MLP1 takes the encoded article as input. MLP2 and MLP3 take in *either* the output of the previous layer (i.e. y1 or y2) with probability $p$ or the true labels of the previous layer (i.e. t1 or t2) with probability (1-$p$). This implements *teacher forcing* since we feed the ground truth labels of Layer i to Layer i+1. Each embedding MLP outputs a vector of shape 1x100 which gets fed into a GRU. The GRU aims to remember sequential information between the layers. This is important since the L1, L2 and L3 labels have dependencies between them. For instance, Place ∈ L1, Building ∈ L2 and HistoricalBuilding ∈ L3. The L3 labels are a specific type of L2 label which are in turn a specific type of L1 label. Thus, for example, when predicting an L2 label, we would like to remember the information in our prediction of the L1 label to aid in our L2 prediction. To accomplish this, the inputs to GRUi is the output of GRUi-1 and the output of MLPi with the exception of GRU1 which takes in just the output of MLP1 since it is the first layer. The output of each GRU is a 1x100 vector which gets fed into a *classifer MLP*: CLFi. This outputs a vector of class scores: yi for Layer i and gets fed into the MLP at the next layer. Thus, y1, y2 and y3 have shapes: 1x9, 1x70 and 1x219 respectively which are the number of labels in L1, L2 and L3. Lastly, we compute the cross entropy loss: Lce by averaging the L1,L2 and L3 cross entropies.


<!-- ### Decoder

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
information of the previous level. -->

<!-- The parameter distribution of each decoder component is a follows: -->



<!-- ### Encoder

Our model used a pre-trained BERT transformer to encode input text into a
single dense tensor. The chosen implementation is the `bert-base-uncased` by
Hugging Face, which was pretrained on the BookCorpus and English Wikipedia
datasets. The whole model contains 109,482,240 pre-trained parameters, none of
which are further tuned as part of our application. -->


### Model Parameters


The pre-trained `bert-base-uncased` model which we use as our encoder, contains 110 million parameters [2]. 



*To be filled in*










## Data

### Summary Statistics and Split

The dataset we are using is the [Kaggle DBPedia Classes dataset](https://www.kaggle.com/datasets/danofer/dbpedia-classes). There are 337739 data points in the dataset. 240942 (71%) of these points are in the training set, 36003 (11%) are in the validation set and 60794 (18%) are in the test set. This is the default split that came with the Kaggle dataset. The ratio between training+validation points and test points is roughly 80:20 which seems like a common ratio chosen by many machine learning practicioners. However, the optimal data split also depends on the volume of data avaliable and an 80:20 split may not be an optimal split for all datasets. In addition, the data points chosen for the training data should give a good representation of the entire dataset while leaving enough test points to determine whether the model generalizes well. We believe the collectors of the data in the Kaggle DBPedia dataset conducted careful experiments to determine the optimal dataset split. 
<!-- 
#which should ideally reduce performance variance in the training set  -->

Each data point in the data set is a 4-tuple with shape (4,).  It can be represented as: (article, L1, L2, L3) where article is the input Wikipedia article and L1, L2 and L3 are the 3 ground truth labels as outlined in the Introduction. Each of the values in the tuple are represented as strings. There are 9, 70 and 219 labels in L1, L2 and L3 respectively. The 3 most common L1, L2 and L3 labels along with their frequency percentages in the training set are:  L1:  (’Agent’,  51.80%),  (’Place’, 19.04%), (’Species’, 8.91%), L2:  (’Athlete’, 12.91%), (’Person’, 8.09%), (’Animal’, 6.09%) and L3:  (’AcademicJournal’, 0.80%), (’Manga’, 0.80%), (’FigureSkater’, 0.80%).  Note that slightly over half of the articles in the training set have an L1 classification of ’Agent’.  In the L3 classifications, there is no dominant label since the 3 most common labels all appear in 0.80% of the articles.  The minimum, maximum and average article length in the training set are:  11, 499 and 102.80 tokens respectively.

<!-- We used the [DBPedia Classes Kaggle dataset](https://www.kaggle.com/datasets/danofer/dbpedia-classes) for training
and evaluating our model. It contains 342780 data points, consisting of a Wikipedia article's summary, and it's
classification in 3 levels of categories, ranging from broad categories such as 'Agent' or 'Place', to specific
categories such as 'Earthquake' or 'SolarEclipse'. -->

### Data Transformation



Recall that in our model, articles get fed as input to a pre-trained BERT transformer which encodes the article to a fixed-length vector embedding of shape: 1x768. However, as outlined in the Model section, BERT expects the input sequence to be in a special format. To aid with this, we pass our articles to BERTTokenizer which does the work of tokenizing the articles. In particular, this involves converting words to tokens, adding a [CLS] token to the beginning of each article and a [SEP] token to the end of each article. Note that we treat each article as a sequence. We could have been more granular and treated each sentence within the article as a sequence. This would involve adding a [CLS] token to denote the start of the sentence and a [SEP] token to separate sentences. However, from the BERT paper [1]: "A “sentence” can be an arbitrary span of contiguous text, rather than an actual linguistic sentence". Thus, we are treating each article as a "sentence". [Mention about attention masks]

Next, the maximum token length that BERT supports is 512. To accomodate this, we remove punctuation from each article and use the first 510 words from each article. This allows all *tokenized* articles to meet the 512 maximum length requirement for the BERT transformer. *tokenized* is italizicized since although an article may be 510 words, it's tokenized representation may be a different length. Other strategies for article truncation were considered such as keeping all punctuation but using a smaller region of the article as input. However, after some experimentation, the strategy we used gives the best performance on the validation set. 


Lastly, recall that a data point looks something like: (article, L1, L2, L3) where article is the model input and (L1,L2,L3) are the ground truth labels. Each parameter is a string. Articles are converted to tokens which are then mapped to integers. This is handled by the BERT Tokenizer. However, we must manually map the ground truth labels to integers since this is not handled by BERT. This is what the WordIdMapping class is for. It contains an array of 3 dictionaries: word_to_id where word_to_id[i] is a dictionary mapping an Li label to a unique integer. id_to_word is an array of 3 arrays where id_to_word[i] is an array containing the Li labels as strings. This is used for giving each label a unique integer id. As we iterate through the training set to encode each article, we extract the article label strings and call *map_data.add_and_get(labs)* which converts them to integer ids. This label representation will serve useful when performing multi-label classification. Namely, given a single article embedding, the output of the model forward pass is an array of 3 prediction arrays: *preds*. preds[i] is a vector shape: Li where Li is the number of labels in Li. For instance, preds[0] has shape 9 representing the un-normalized scores for each of the 9 L1 labels. Thus, it is useful to assign each L1 label a unique integer id between 0-8 which is what word_to_id is for! Similarly, we assign integer ids to L2 and L3 labels which have unique ids between 0-69 and 0-218 respectively.


<!-- After the data is processed, we return two arrays:  
There is also the option to pad each article so they are the same length but we decided to omit this.  -->
 



## Training

*To be filled in*

## Results

*To be filled in*


## References

[1] https://arxiv.org/pdf/1810.04805.pdf
[2] https://huggingface.co/transformers/v3.3.1/pretrained_models.html


#### Authored by: Keith Allatt, Brandon Jaipersaud, David Chen, Renato Zimmermann


---

### TODOs

- update teacher forcing probability in model


