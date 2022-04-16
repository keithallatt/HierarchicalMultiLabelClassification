# CSC413H5 Final Project - Multi-Label Classification on DBPedia

## Introduction


Our deep learning model will perform hierarchical, multi-label classification to articles. Note that this is different from multi-class classification which assigns a *single* label/class to an input which is chosen from a group of multiple labels/classes. In particular, given an article as input, our model will assign 3 class labels to it, each of which come from a set of labels: L1, L2 and L3 respectively.  Li contains broader labels than Lj where i < j.  To illustrate, our model may assign labels: {Place, Building, HistoricalBuilding} to an article where Place ∈ L1,  Building ∈ L2 and HistoricalBuilding ∈ L3.  Note that Building is a type of Place while HistoricalBuilding is a type of Building.  Tagging an unstructured article with these 3 labels serves to categorize and add some structure to it.



## Model

### Overview

We tackle the hierarchical, multi-label classification task using an encoder-decoder 
architecture composed of a transformer-based encoder and a GRU-based decoder. process_documents() in data_cleaning.py 
defines the implementation of our encoder. The HierarchicalRNN class in model.py defines the implementation of our decoder.


### Design Decisions



#### Using BERT as our Encoder

Our choice of encoder is a pre-trained BERT transformer as outlined in [1]. Our chosen implementation is the `bert-base-uncased` by
Hugging Face, which was pretrained on the BookCorpus and English Wikipedia datasets. BERT is a state-of-the-art encoder used for NLP-based tasks. One of its distinguishing features is its bidirectional language modelling. Most other transformers and language models are unidirectional. For instance, recall the 4-gram language model developed in CSC413 A1. This model was trained to predict the next word given the previous three words in a sentence which makes it a unidirectional language model. In contrast, BERT uses both left and right context when encoding sequences which will make it much better at summarizing and learning the "gist" of an article.


The input to our model are articles which are very long sequences. Using a traditional RNN would be a poor choice of encoder since it would fail to effectively summarize and extract the important parts of the article. The attention mechanism of a transformer was designed to solve this issue. Furthermore, the articles in our training set are Wikipedia articles. Since `bert-base-uncased` was also trained on Wikipedia articles, it makes this model an even better candidate to use as an encoder. 



### Model Diagram and Forward Pass

The computation graph of our model is shown below:

<img src="readme_assets/computation_graph.png" alt="computation_graph"/>



#### Encoder

Our encoder model takes in a single article, x as input. The article must go through a pre-processing stage which does two things: First, punctuation is stripped from the article and the first 510 article characters are taken. Second, the truncated article is passed to a BERT Tokenizer which converts the article to a format acceptable by the BERT transformer. These ideas are outlined in more detail in the Data Transformation section of the report. The tokenized article is then passed to the BERT Transformer which outputs an article embedding of shape 1x768. Intuitively, this represents a summarized version of the article.


#### Decoder

The decoder portion of our model takes in the encoded article of shape 1x768. It consists of three layers: One for predicting L1, L2 and L3 labels respectively. Each Layer i begins with an MLP which we call the *embedding MLP*. MLP1 takes the encoded article as input. MLP2 and MLP3 take in *either* the output of the previous layer (i.e. y1 or y2) with probability p or the true labels of the previous layer (i.e. t1 or t2) with probability 1-p. This implements *teacher forcing* since we feed the ground truth labels of Layer i to Layer i+1. Each embedding MLP contains several layers with a ReLU activation function applied between each layer. It outputs a vector of shape 1x100 which then gets fed into a GRU. The number of hidden states in the GRU is a hyperparameter which is set to 100 by default. The GRU aims to remember sequential information between the layers. This is important since the L1, L2 and L3 labels have dependencies between them. For instance, Place ∈ L1, Building ∈ L2 and HistoricalBuilding ∈ L3. The L3 labels are a specific type of L2 label which are in turn a specific type of L1 label. Thus, for example, when predicting an L2 label, we would like to remember the information in our prediction of the L1 label to aid in our L2 prediction. To accomplish this, the inputs to GRUi is the output of GRUi-1 and the output of MLPi with the exception of GRU1 which takes in just the output of MLP1 since it is the first layer. The output of each GRU is a 1x100 vector which gets fed into a *classifer MLP*: CLFi. Similar to the embedding MLP, the classifier MLP contains several layers with ReLU applied between each layer. However, it outputs a vector of class scores: yi for Layer i and gets fed into the embedding MLP in the next layer. Thus, y1, y2 and y3 have shapes: 1x9, 1x70 and 1x219 respectively which are the number of labels in L1, L2 and L3. Lastly, we compute the cross entropy loss: Lce by averaging the L1,L2 and L3 cross entropies.


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


### Encoder

The pre-trained `bert-base-uncased` model which we use as our encoder, contains 110 million parameters [2]. However, these parameters have already been set by the pre-trained model so we don't include them as part of our tuneable parameter count.


### Decoder 

#### Understanding the Shapes of the Decoder Components



To count the number of parameters in the decoder portion of our model we must consider the embedding and classifier
MLPs at each layer as well as the GRUs. The number of layers in each MLP and the output shape of the embedding MLPs is a hyperparameter. In our model, this is
configured by specifying three parameters: embedding_size and [emb|clf]_size_mults. The former specifies the output shape of the embedding MLP. The latter is an n-tuple which defines the number of layers in the embedding/classifier MLPs and the number of hidden units in the current MLP layer as a multiple of the hidden units in the previous MLP layer. Note that the output shape of the classifier MLPs are fixed since they represent the un-normalized class scores for the L1,L2,L3 labels respectively. To illustrate, suppose embedding_size=100 and emb_size_mults = (1,2,3). The input/output shapes of MLP1 from the diagram looks like: (in_features=768, out_features=768) -> (in_features=768, out_features=1536) -> (in_features=1536, out_features=4608) -> (in_features=4608, out_features=100). The input/output shapes of MLP2 looks like:  (in_features=9, out_features=9) -> (in_features=9, out_features=18) -> (in_features=18, out_features=54) -> (in_features=54, out_features=100). Notice that the input to MLP1 is a 1x768 vector which represents the encoded article. The input to MLP2 is a 1x9 vector which represents the output of Layer 1. Also notice that the output features/number of hidden units in each layer is determined by the emb_multiplier: (1,2,3). Both MLPs output a 1x100 vector as determined by embedding_size=100. This is useful since the inputs to the GRU at each layer must be the same shape. 

The input/output of the classifier MLPs are determined in a similar fashion. clf_multiplier is used instead of emb_multiplier to define the number of layers and hidden units in the classifier MLPs. The outputs of CLF1, CLF2 and CLF3 are 1x9, 1x70 and 1x219 respectively which represent the un-normalized class scores for L1, L2 and L3 respectively. 

Lastly, the number of features of the GRUs in each layer is a hyperparameter which is calculated as a multiple of the emb_size. For instance, if emb_size = 100 and rnn_size_mult = 1, each GRU will contain 100 features. The number of GRU layers can also be configured via. rnn_n_hidden but we leave this as 1 when tuning our hyperparameters. 

#### Number of Parameters in the Decoder


Our final model uses the default parameters specified in the HierarchicalRNN class in model.py. A summary of the number of trainable decoder parameters is below:

<table>
  <tr>
    <th>Component</th>
    <th>Trainable Parameters</th>
  
  </tr>
  <tr>
    <td>MLP1</td>
    <td>768*768 + 768*100 + 768+100 = 667492 </td>
  
  </tr>
  <tr>
    <td>MLP2</td>
    <td>9*9 + 9*100 + 9+100 = 1090 </td>
  </tr>
  <tr>
    <td>MLP3</td>
    <td>70*70 + 70*100 + 70+100 = 12070 </td>
  
  </tr>
  <tr>
    <td>CLF1</td>
    <td>100*100 + 100*9 + 100+9 = 11009 </td>
   
  </tr>
  <tr>
    <td>CLF2</td>
    <td>100*100 + 100*70 + 100+70 = 17170</td>
    
  </tr>
  <tr>
    <td>CLF3</td>
    <td>100*100 + 100*219 + 100+219 = 32219</td>
  </tr>
  <tr>
    <td>GRU</td>
    <td>60600</td>
  </tr>
  <tr>
    <td>Total</td>
    <td>801650</td>
  </tr>
</table>


Therefore, the total number of trainable parameters used by our final model is 801,650.







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


Note that no data augmentation was applied since we have an adequate volume of data in our dataset.

<!-- After the data is processed, we return two arrays:  
There is also the option to pad each article so they are the same length but we decided to omit this.  -->
 



## Training

*To be filled in*

## Results

### Quantitative Measures

- L1,L2,L3, total loss and accuracy
- compare results with BaselineMLP
- compare with other dbpedia benchmarks
- 

*To be filled in*


## References

[1] https://arxiv.org/pdf/1810.04805.pdf

[2] https://huggingface.co/transformers/v3.3.1/pretrained_models.html


#### Authored by: Keith Allatt, Brandon Jaipersaud, David Chen, Renato Zimmermann


---

### TODOs

- update teacher forcing probability in model
- rescale model diagram so it looks good on main Github page
- can add Latex support to model explanations (i.e. L subscript 1 instead of L1)
- maybe provide detailed GRU parameter breakdown for model parameter section
- change multi-label classification to hierarchical multi-class classification


