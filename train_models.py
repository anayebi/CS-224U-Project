import numpy as np
import h5py
import pickle
import utils
import keras
import keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Input, merge
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation, Dense, Highway, Dropout, Merge, Lambda, TimeDistributedDense
from keras.layers.wrappers import TimeDistributed
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adagrad
from keras.callbacks import ModelCheckpoint, EarlyStopping
from preprocess_sentiment import *
from softattention import SoftAttentionConcat, TDistSoftAttention, LSTMMem, AttnFusion, AttnFusionOnes

nb_classes = 5 # 5 classes for fine-grained sentiment
buildWordMap()
vocab_dim = 300
glove_home = 'glove_dir/glove.6B'
wordMap = loadWordMap()
vocab_size = len(wordMap) + 1 # adding 1 to account for 0th index (for masking)
#vocab_size = 16583
max_sen_length = get_max_sen_len() # can use this or if you know just give it the number
#max_sen_length = 56

early_stop_val = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='auto')


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

class ValLossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('val_loss'))

class AccHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('acc'))

class ValAccHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('val_acc'))

def load_data_embedding(glove6B=False):
    X_train, y_train = build_dataset('train', max_sen_length)
    X_dev, y_dev = build_dataset('dev', max_sen_length)
    embedding_weights = np.zeros((vocab_size, vocab_dim))
    if vocab_dim == 50:
        GLOVE = utils.glove2dict(os.path.join(glove_home, 'glove.6B.50d.txt'))
    elif vocab_dim == 100:
        GLOVE = utils.glove2dict(os.path.join(glove_home, 'glove.6B.100d.txt'))
    elif vocab_dim == 200:
        GLOVE = utils.glove2dict(os.path.join(glove_home, 'glove.6B.200d.txt'))
    elif vocab_dim == 300:
        if glove6B:
            GLOVE = utils.glove2dict(os.path.join(glove_home, 'glove.6B.300d.txt'))
        else:
            glove_home = 'glove_dir/glove.840B'
            GLOVE = utils.glove2dict(os.path.join(glove_home, 'glove.840B.300d.txt'))
    for word, index in wordMap.items():
        if word in GLOVE:
            embedding_weights[index, :] = GLOVE[word]
        else:
            embedding_weights[index, :] = utils.randvec(vocab_dim)
    return X_train, y_train, X_dev, y_dev, embedding_weights


def train_lstm_fusion(X_train, y_train, X_dev, y_dev, embedding_weights, reg=0.0, embed_glove=False):

    '''Trains an lstm network, using my recurrent attention layer,
    which is based on Cheng et al. deep attention fusion and ideas from Section 3.1 of Luong et al. 2015 (http://arxiv.org/pdf/1508.04025v5.pdf)'''

    checkpointer = ModelCheckpoint(filepath="lstm_memfusion_best.hdf5", monitor='val_acc', verbose=1, save_best_only=True) #saves best val loss weights
    input_sentences = Input(shape=(max_sen_length,), dtype='int32')
    if embed_glove: # embed glove vectors
        x = Embedding(input_dim=vocab_size, output_dim=vocab_dim, input_length=max_sen_length, mask_zero=True, weights=[embedding_weights])(input_sentences)
    else: # or use random embedding
        x = Embedding(input_dim=vocab_size, output_dim=vocab_dim, input_length=max_sen_length, mask_zero=True)(input_sentences)
    dropout_x = Dropout(0.15)(x)
    lstm_out = LSTM(vocab_dim, dropout_U=0.25, return_sequences=True)(dropout_x)
    context = TDistSoftAttention(LSTMMem(vocab_dim/2, dropout_U=0.25, return_mem=True))(lstm_out)
    # NOTE: attention needs to be twice that of LSTMem for r*cell_in operation to be valid
    attentional_hs = AttnFusion(vocab_dim, dropout_U=0.3, W_regularizer=l2(0.0), U_regularizer=l2(0.0), return_sequences=False)(context)
    attentional_hs = Highway(activity_regularizer=activity_l2(reg))(attentional_hs)
    prediction = Dense(nb_classes, activation='softmax', activity_regularizer=activity_l2(reg))(attentional_hs)
    history = LossHistory()
    val_history = ValLossHistory()
    acc = AccHistory()
    val_acc = ValAccHistory()
    model = Model(input=input_sentences, output=prediction)
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, nb_epoch=40, batch_size=300, validation_data=(X_dev, y_dev), callbacks=[checkpointer, early_stop_val, history, val_history, acc, val_acc])
    pickle.dump(history.losses, open("lstm_memfusion_trainloss.p", "wb"))
    pickle.dump(val_history.losses, open("lstm_memfusion_devloss.p", "wb"))
    pickle.dump(acc.losses, open("lstm_memfusion_trainacc.p", "wb"))
    pickle.dump(val_acc.losses, open("lstm_memfusion_devacc.p", "wb"))

def train_lstm(X_train, y_train, X_dev, y_dev, embedding_weights, reg=0.0, embed_glove=False):

    '''Trains a vanilla lstm network '''

    checkpointer = ModelCheckpoint(filepath="lstm_best.hdf5", monitor='val_acc', verbose=1, save_best_only=True) #saves best val loss weights
    model = Sequential()
    if embed_glove: # embed glove vectors
        model.add(Embedding(input_dim=vocab_size, output_dim=vocab_dim, input_length=max_sen_length, mask_zero=True, weights=[embedding_weights]))
    else: # or use random embedding
        model.add(Embedding(input_dim=vocab_size, output_dim=vocab_dim, input_length=max_sen_length, mask_zero=True))
    model.add(Dropout(0.2))
    model.add(LSTM(150, return_sequences=False))
    model.add(Dense(nb_classes, W_regularizer=l2(reg)))
    model.add(Activation('softmax'))
    history = LossHistory()
    val_history = ValLossHistory()
    acc = AccHistory()
    val_acc = ValAccHistory()
    model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, nb_epoch=20, validation_data=(X_dev, y_dev), callbacks=[checkpointer, early_stop_val, history, val_history, acc, val_acc])
    pickle.dump(history.losses, open("lstm_trainloss.p", "wb"))
    pickle.dump(val_history.losses, open("lstm_devloss.p", "wb"))
    pickle.dump(acc.losses, open("lstm_trainacc.p", "wb"))
    pickle.dump(val_acc.losses, open("lstm_devacc.p", "wb"))

def train_bilstm(X_train, y_train, X_dev, y_dev, embedding_weights, reg=0.0, embed_glove=False):

    '''Trains a vanilla bidirectional lstm network '''

    checkpointer = ModelCheckpoint(filepath="bilstm_best.hdf5", monitor='val_acc', verbose=1, save_best_only=True) #saves best val loss weights
    input_sentences = Input(shape=(max_sen_length,), dtype='int32')
    if embed_glove: # embed glove vectors
        x = Embedding(input_dim=vocab_size, output_dim=vocab_dim, input_length=max_sen_length, mask_zero=True, weights=[embedding_weights])(input_sentences)
    else: # or use random embedding
        x = Embedding(input_dim=vocab_size, output_dim=vocab_dim, input_length=max_sen_length, mask_zero=True)(input_sentences)
    d = Dropout(0.3)(x)
    lstm_1 = LSTM(300, return_sequences=False, dropout_W=0.0, dropout_U=0.3)(d)
    lstm_2 = LSTM(300, return_sequences=False, go_backwards=True, dropout_W=0.0, dropout_U=0.3)(d)
    concat = merge([lstm_1, lstm_2], mode='concat')
    prediction = Dense(nb_classes, activation='softmax', activity_regularizer=activity_l2(reg))(concat)
    history = LossHistory()
    val_history = ValLossHistory()
    acc = AccHistory()
    val_acc = ValAccHistory()
    model = Model(input=input_sentences, output=prediction)
    model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, nb_epoch=20, batch_size=300, validation_data=(X_dev, y_dev), callbacks=[checkpointer, early_stop_val, history, val_history, acc, val_acc])
    pickle.dump(history.losses, open("bilstm_trainloss.p", "wb"))
    pickle.dump(val_history.losses, open("bilstm_devloss.p", "wb"))
    pickle.dump(acc.losses, open("bilstm_trainacc.p", "wb"))
    pickle.dump(val_acc.losses, open("bilstm_devacc.p", "wb"))

def train_lstm_mem(X_train, y_train, X_dev, y_dev, embedding_weights, reg=0.0, embed_glove=False):

    '''Trains an lstm network with simple attention, using ideas from Section 3.1 of Luong et al. 2015 (http://arxiv.org/pdf/1508.04025v5.pdf) '''

    checkpointer = ModelCheckpoint(filepath="lstm_mem_best.hdf5", monitor='val_acc', verbose=1, save_best_only=True) #saves best val loss weights
    input_sentences = Input(shape=(max_sen_length,), dtype='int32')
    if embed_glove: # embed glove vectors
        x = Embedding(input_dim=vocab_size, output_dim=vocab_dim, input_length=max_sen_length, mask_zero=True, weights=[embedding_weights])(input_sentences)
    else: # or use random embedding
        x = Embedding(input_dim=vocab_size, output_dim=vocab_dim, input_length=max_sen_length, mask_zero=True)(input_sentences)
    new_x = Dropout(0.3)(x)
    lstm_comp = LSTM(vocab_dim, dropout_U=0.3, return_sequences=True)
    contextconcat = SoftAttentionConcat(lstm_comp)(new_x)
    attentional_hs = Dense(25, activation='tanh', activity_regularizer=activity_l2(reg))(contextconcat)
    prediction = Dense(nb_classes, activation='softmax', activity_regularizer=activity_l2(reg))(attentional_hs)
    history = LossHistory()
    val_history = ValLossHistory()
    acc = AccHistory()
    val_acc = ValAccHistory()
    model = Model(input=input_sentences, output=prediction)
    model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, nb_epoch=20, batch_size=300, validation_data=(X_dev, y_dev), callbacks=[checkpointer, early_stop_val, history, val_history, acc, val_acc])
    pickle.dump(history.losses, open("lstm_mem_trainloss.p", "wb"))
    pickle.dump(val_history.losses, open("lstm_mem_devloss.p", "wb"))
    pickle.dump(acc.losses, open("lstm_mem_trainacc.p", "wb"))
    pickle.dump(val_acc.losses, open("lstm_mem_devacc.p", "wb"))

X_train, y_train, X_dev, y_dev, embedding_weights = load_data_embedding(glove6B=False)

'''Uncomment one of these to train a given model'''
#train_bilstm(X_train, y_train, X_dev, y_dev, embedding_weights, reg=0.1, embed_glove=True)
#train_lstm_mem(X_train, y_train, X_dev, y_dev, embedding_weights, reg=0.1, embed_glove=True)
#train_lstm_fusion(X_train, y_train, X_dev, y_dev, embedding_weights, reg=0.1, embed_glove=True)
#train_lstm(X_train, y_train, X_dev, y_dev, embedding_weights, reg=0.01, embed_glove=True)
