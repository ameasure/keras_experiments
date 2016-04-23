# -*- coding: utf-8 -*-
"""
This is an attempt to use a single word embedding for multiple filters by combining
the Graph and Sequential containers.

@author: ameasure
"""

import datetime

import pandas as pd
import numpy as np
np.random.seed(42)

from sklearn.preprocessing import LabelEncoder

import theano
import keras
from keras.models import Sequential, Graph
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2

from embed_preprocessing import get_initial_embeddings
from msha_extractor import get_data




def accuracy(model, x_test, y_test):
    prediction_probs = model.predict({'input': x_test})['output']
    predictions = prediction_probs.argmax(axis=1)
    comparison = y_test.argmax(axis=1)
    matches = predictions==comparison
    return float(matches.sum())/len(matches)
    
class AccuracyCB(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.accuracy = []
        
    def on_batch_end(self, batch, logs={}):
        print logs
        acc = accuracy(self.model, x_test=batch, y_test=self.model.validation_data[1])
        self.accuracy.append(acc)
        
n_filters = 400
max_len = 100
embedding_size = 300
batch_size = 128

code_type = 'ACTIVITY_CD'
raw_train, raw_valid, raw_test = get_data(n_train=47500, n_valid=2500, n_test=10000)
raw_train_labels = raw_train[code_type]
raw_valid_labels = raw_valid[code_type]
raw_test_labels = raw_test[code_type]

labeler = LabelEncoder()
labels = set(raw_train[code_type].tolist() + raw_test[code_type].tolist())
labeler.fit(list(labels))
nb_classes = len(set(labels))
print('nb_classes = %s' % nb_classes)

y_train = labeler.transform(raw_train_labels)
Y_train = np_utils.to_categorical(y_train, nb_classes)

y_valid = labeler.transform(raw_valid_labels)
Y_valid = np_utils.to_categorical(y_valid, nb_classes)

y_test = labeler.transform(raw_test_labels)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print 'Tokenizing X_train'
tokenizer = Tokenizer()
tokenizer.fit_on_texts(raw_train['NARRATIVE'])
tokenizer.word_index['BLANK_EMBEDDING'] = 0
X_train = tokenizer.texts_to_sequences(raw_train['NARRATIVE'])
X_valid = tokenizer.texts_to_sequences(raw_valid['NARRATIVE'])
X_test = tokenizer.texts_to_sequences(raw_test['NARRATIVE'])
X_train = pad_sequences(X_train, maxlen=max_len)
X_valid = pad_sequences(X_valid, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)
print('X_train shape:', X_train.shape)
print('X_valid shape:', X_valid.shape)
print('X_test shape:', X_test.shape)

initial_embeddings = get_initial_embeddings(tokenizer.word_index)
vocab_size = len(tokenizer.word_index)
ngram_filters = [2, 3, 4, 5, 6]
filter_lengths = [300, 300, 300, 300, 300]

convolution_outputs = []

model = Graph()
model.add_input(name='input', input_shape=(vocab_size,), dtype='int')
model.add_node(layer=Embedding(vocab_size, embedding_size, 
                               weights=[initial_embeddings],
                               input_length=max_len, dropout=0.5),
               name='embedding', input='input')
model.add_node(layer=Reshape((1, max_len, embedding_size)), name='reshape', input='embedding')
for n, n_gram in enumerate(ngram_filters):
    sequential = Sequential()
    sequential.add(Convolution2D(nb_filter=filter_lengths[n], 
                                 nb_row=n_gram, 
                                 nb_col=embedding_size,
                                 input_shape=(1, max_len, embedding_size)))
    sequential.add(Activation("relu"))
    sequential.add(MaxPooling2D(pool_size=(max_len - n_gram + 1, 1)))
    sequential.add(Flatten())
    model.add_node(sequential, name='unit_%s' % n_gram, input='reshape')
model.add_node(Dropout(0.5), name='dropout', 
               inputs=['unit_%s' % n for n in ngram_filters],
               merge_mode='concat')
fc = Sequential()
fan_in = sum(filter_lengths)
fc.add(Dense(output_dim=nb_classes, input_shape=(fan_in,)))
fc.add(Activation('softmax'))
model.add_node(fc, name='fully_connected', input='dropout')
model.add_output(name='output', input='fully_connected')

print 'compiling model'
model.compile(loss={'output':'categorical_crossentropy'}, optimizer='adam')
print 'fitting model'

#accuracy_cb = AccuracyCB()
for epoch in range(10):
    print 'Epoch: %s' % epoch
    model.fit({'input':X_train, 'output': Y_train}, nb_epoch=1, 
              batch_size=batch_size, verbose=1, validation_split=0.05)
    valid_accuracy = accuracy(model=model, x_test=X_valid, y_test=Y_valid)
    print 'val_accuracy: %s' % valid_accuracy

score = model.evaluate({'input':X_test, 'output':Y_test}, batch_size=batch_size)
print accuracy(model=model, x_test=X_test, y_test=Y_test)
