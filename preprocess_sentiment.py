import numpy as np
import collections
import os
from keras.preprocessing.sequence import pad_sequences

UNK = 'UNK'

# This file contains the dataset in a useful way. We populate a list of Trees to train/test our Neural Nets such that each Tree contains any number of Node objects.

# The best way to get a feel for how these objects are used in the program is to drop pdb.set_trace() in a few places throughout the codebase
# to see how the trees are used.. look where loadtrees() is called etc..


class Node: # a node in the tree
    def __init__(self,label,word=None):
        self.label = label 
        self.word = word # NOT a word vector, but index into L.. i.e. wvec = L[:,node.word]
        self.parent = None # reference to parent
        self.left = None # reference to left child
        self.right = None # reference to right child
        self.isLeaf = False # true if I am a leaf (could have probably derived this from if I have a word)
        self.fprop = False # true if we are currently performing fprop
        self.hActs1 = None # h1 from the handout
        self.hActs2 = None # h2 from the handout (only used for RNN2)
        self.probs = None # yhat

class Tree:

    def __init__(self,treeString,openChar='(',closeChar=')'):
        tokens = []
        self.open = '('
        self.close = ')'
        for toks in treeString.strip().split():
            tokens += list(toks)
        self.root = self.parse(tokens)

    def parse(self, tokens, parent=None):
        assert tokens[0] == self.open, "Malformed tree"
        assert tokens[-1] == self.close, "Malformed tree"

        split = 2 # position after open and label
        countOpen = countClose = 0

        if tokens[split] == self.open: 
            countOpen += 1
            split += 1
        # Find where left child and right child split
        while countOpen != countClose:
            if tokens[split] == self.open:
                countOpen += 1
            if tokens[split] == self.close:
                countClose += 1
            split += 1

        # New node
        node = Node(int(tokens[1])-1) # zero index labels
        node.parent = parent 

        # leaf Node
        if countOpen == 0:
            node.word = ''.join(tokens[2:-1]).lower() # lower case?
            node.isLeaf = True
            return node

        node.left = self.parse(tokens[2:split],parent=node)
        node.right = self.parse(tokens[split:-1],parent=node)

        return node

        

def leftTraverse(root,nodeFn=None,args=None):
    """
    Recursive function traverses tree
    from left to right. 
    Calls nodeFn at each node
    """
    nodeFn(root,args)
    if root.left is not None:
        leftTraverse(root.left,nodeFn,args)
    if root.right is not None:
        leftTraverse(root.right,nodeFn,args)


def countWords(node,words):
    if node.isLeaf:
        words[node.word] += 1

def mapWords(node,wordMap):
    if node.isLeaf:
        if node.word not in wordMap:
            node.word = wordMap[UNK]
        else:
            node.word = wordMap[node.word]
    

def loadWordMap():
    import cPickle as pickle
    
    with open('wordMap.bin','r') as fid:
        return pickle.load(fid)

def buildWordMap():
    """
    Builds map of all words in training set
    to integer values.
    """

    import cPickle as pickle
    print "Reading trees to build word map.."
    file='trees/train.txt'
    with open(file,'r') as fid:
        trees = [Tree(l) for l in fid.readlines()]

    print "Counting words to give each word an index.."

    words = collections.defaultdict(int)
    for tree in trees:
        leftTraverse(tree.root,nodeFn=countWords,args=words)

    wordMap = dict(zip(words.iterkeys(),xrange(1,len(words)+1))) #ensures no word has zero so that we can zero-pad LSTM in Keras
    wordMap[UNK] = len(words)+1 # Add unknown as word

    print "Saving wordMap to wordMap.bin"
    with open('wordMap.bin','w') as fid:
        pickle.dump(wordMap,fid)


def loadTrees(dataSet='train'):
    """
    Loads training trees. Maps leaf node words to word ids.
    """
    wordMap = loadWordMap()
    file = 'trees/%s.txt'%dataSet
    print "Loading %sing trees.."%dataSet
    with open(file,'r') as fid:
        trees = [Tree(l) for l in fid.readlines()]
    for tree in trees:
        leftTraverse(tree.root,nodeFn=mapWords,args=wordMap)
    return trees
 
def getLeaves(root, leaves):
    """
    Recursive function traverses tree
    from left to right. 
    Calls nodeFn at each node
    """
    if root.isLeaf:
        leaves.append(root.word)
    if root.left is not None:
        getLeaves(root.left, leaves)
    if root.right is not None:
        getLeaves(root.right, leaves)

def get_max_sen_len(): # compute max sentence length across all the sets
    train = loadTrees()
    dev = loadTrees(dataSet='dev')
    test = loadTrees(dataSet='test')
    max_sen_length = 0
    for tree in train:
        leaves = []
        getLeaves(tree.root, leaves)
        if len(leaves) > max_sen_length:
            max_sen_length = len(leaves)

    for tree in dev:
        leaves = []
        getLeaves(tree.root, leaves)
        if len(leaves) > max_sen_length:
            max_sen_length = len(leaves)

    for tree in test:
        leaves = []
        getLeaves(tree.root, leaves)
        if len(leaves) > max_sen_length:
            max_sen_length = len(leaves)

    return max_sen_length

def build_dataset(argset, max_sen_length):
    X = [] #sentences but with their word indices
    y = [] #root labels
    datasetTrees = loadTrees(dataSet=argset) #maps each word to its index
    for tree in datasetTrees:
        leaves = []
        getLeaves(tree.root, leaves)
        X.append(leaves)
        pred = np.zeros((5,))
        pred[tree.root.label] = 1 #make them one hot vectors
        y.append(pred)
    y = np.array(y)
    pad_X = pad_sequences(X, maxlen=max_sen_length) #pad input with zeros to deal with variable length, can try padding='post' to put zeros at the end but default is 'pre' so go with that
    return pad_X, y

def build_dataset_nopad(argset):
    X = [] #sentences but with their word indices
    y = [] #root labels
    import cPickle as pickle
    print "Reading trees to build word map.."
    file='trees/test.txt'
    with open(file,'r') as fid:
        trees = [Tree(l) for l in fid.readlines()]

    print "Counting words to give each word an index.."

    words = collections.defaultdict(int)
    for tree in trees:
        leftTraverse(tree.root,nodeFn=countWords,args=words)

    wordMap = dict(zip(words.iterkeys(),xrange(1,len(words)+1))) #ensures no word has zero so that we can zero-pad LSTM in Keras
    wordMap[UNK] = len(words)+1 # Add unknown as word

    print "Saving wordMap to wordMap.bin"
    with open('wordMap.bin','w') as fid:
        pickle.dump(wordMap,fid)
    datasetTrees = loadTrees(dataSet=argset) #maps each word to its index
    for tree in datasetTrees:
        leaves = []
        getLeaves(tree.root, leaves)
        X.append(leaves)
        pred = np.zeros((5,))
        pred[tree.root.label] = 1 #make them one hot vectors
        y.append(pred)
    X = np.array(X)
    wordmap = loadWordMap()
    for i in range(X.shape[0]):
        for idx, word_rep in enumerate(X[i]):
            for key in wordmap.iterkeys():
                if wordmap[key] == X[i][idx]:
                    X[i][idx] = key
    y = np.array(y)
    #pad_X = pad_sequences(X, maxlen=max_sen_length) #pad input with zeros to deal with variable length, can try padding='post' to put zeros at the end but default is 'pre' so go with that
    return X, y
#if __name__=='__main__':





