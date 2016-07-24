# Recurrent versus Recursive Approaches Towards Compositionality in Semantic Vector Spaces
This is code I wrote for a course project for CS 224U: Natural Language Understanding, taught by Bill MacCartney and Chris Potts in Spring Quarter 2016.

Project Abstract: Semantic vector spaces have long been useful for representing word tokens; however, they cannot express the meaning of longer phrases without some notion of compositionality. Recursive neural models explicitly encode syntactic properties that combine word representations into phrases; whereas recurrent neural models attain compositionality by processing word representations sequentially. A natural question that arises is whether recursive models are strictly necessary for attaining meaningful compositionality or are recurrent models sufficient? In this paper, we demonstrate that for the task of fine-grained sentiment analysis, recurrent models augmented with neural attention can outperform a recursive model. Specifically, we introduce a new type of recurrent attention mechanism that allows us to achieve 47.4% accuracy for the root-level sentiment analysis task on the Stanford Sentiment Treebank, which outperforms the Recursive Neural Tensor Network's (RNTN) previous 45.7% accuracy on the same dataset.

For the paper, see: https://sites.google.com/site/anayebihomepage/cs224ufinalproject

# Libraries Needed:

1. Numpy
2. Theano
3. Keras, a deep learning library for Theano and TensorFlow (I patched Keras 1.0 for this project for the additional attention layers to work, and this is located in `keras.zip`)
4. GloVe vectors: http://nlp.stanford.edu/projects/glove/, preferably the 840B 300-dimensional vectors from: http://nlp.stanford.edu/data/glove.840B.300d.zip

# How to Train

First, install Keras by unzipping `keras.zip`, and download the GloVe vectors to a directory of your choice. After doing so, you are ready to begin!

`train_models.py` can be used to directly train any of the four models (LSTM, Bidirectional LSTM, LSTM + global attention, and LSTM + recurrent attention), by uncommenting one of the functions at the end of the file. The `glove_home` global variable must be defined to include a valid path to your GloVe vectors. The LSTM + recurrent attention is our own new attentional model which performed the best in the fine-grained sentiment task. This can be trained by uncommenting the `train_lstm_fusion` function. `softattention.py`, `utils.py`, `preprocess_sentiment.py`, and `/trees` should all be in the same directory as `train_models.py`.

# Code Explanation

`softattention.py` contains all the additional Keras layers. `utils.py` (written by Chris Potts for the CS 224U course) and `preprocess_sentiment.py` (adapted from code written for Richard Socher's CS 224D course), are used to create the train and test data sets from the `/trees` directory for the sentiment analysis task as well as to aid in the creation of GloVe vector word embeddings used in `train_models.py`. The `/trees` directory is the Stanford Sentiment Treebank dataset (more information can be found in our paper or here: http://nlp.stanford.edu/sentiment/treebank.html).
