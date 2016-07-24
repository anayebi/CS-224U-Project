# Recurrent versus Recursive Approaches Towards Compositionality in Semantic Vector Spaces
This is code I wrote for a course project for CS 224U: Natural Language Understanding, taught by Bill MacCartney and Chris Potts in Spring Quarter 2016.

Project Abstract: Semantic vector spaces have long been useful for representing word tokens; however, they cannot express the meaning of longer phrases without some notion of compositionality. Recursive neural models explicitly encode syntactic properties that combine word representations into phrases; whereas recurrent neural models attain compositionality by processing word representations sequentially. A natural question that arises is whether recursive models are strictly necessary for attaining meaningful compositionality or are recurrent models sufficient? In this paper, we demonstrate that for the task of fine-grained sentiment analysis, recurrent models augmented with neural attention can outperform a recursive model. Specifically, we introduce a new type of recurrent attention mechanism that allows us to achieve 47.4% accuracy for the root-level sentiment analysis task on the Stanford Sentiment Treebank, which outperforms the Recursive Neural Tensor Network's (RNTN) previous 45.7% accuracy on the same dataset.

# Libraries Needed:

1. Numpy
2. Theano
3. Keras (I patched Keras 1.0 for this project for the additional attention layers to work, and this is located in the `/keras` directory)


For the paper, see: https://sites.google.com/site/anayebihomepage/cs224ufinalproject
