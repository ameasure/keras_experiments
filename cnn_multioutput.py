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
import random

import theano
import keras
from keras.models import Sequential, Graph
from keras.layers.convolutional import Convolution1D, MaxPooling1D
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


def accuracy(model, x_test, y_test, output_name):
    prediction_probs = model.predict({'input': x_test})[output_name]
    predictions = prediction_probs.argmax(axis=1)
    comparison = y_test.argmax(axis=1)
    matches = predictions==comparison
    return float(matches.sum())/len(matches)
    
        
max_len = 100
embedding_size = 300
batch_size = 128

code_types = ['ACTIVITY_CD', 'CLASSIFICATION_CD', 'ACCIDENT_TYPE_CD', 'INJURY_SOURCE_CD',
              'OCCUPATION_CD', 'NATURE_INJURY_CD', 'INJ_BODY_PART_CD']
raw_train, raw_valid, raw_test = get_data(n_train=47500, n_valid=2500, n_test=10000)

Y = {'train': {}, 'valid': {}, 'test': {}}
nb_classes = {}
for code_type in code_types:
    raw_train_labels = raw_train[code_type]
    raw_valid_labels = raw_valid[code_type]
    raw_test_labels = raw_test[code_type]

    labeler = LabelEncoder()
    labels = set(raw_train[code_type].tolist() + raw_valid[code_type].tolist() + raw_test[code_type].tolist())
    labeler.fit(list(labels))
    nb_classes[code_type] = len(set(labels))
    
    y_train = labeler.transform(raw_train_labels)
    Y['train'][code_type] = np_utils.to_categorical(y_train, nb_classes[code_type])
    
    y_valid = labeler.transform(raw_valid_labels)
    Y['valid'][code_type] = np_utils.to_categorical(y_valid, nb_classes[code_type])
    
    y_test = labeler.transform(raw_test_labels)
    Y['test'][code_type] = np_utils.to_categorical(y_test, nb_classes[code_type])
print('nb_classes[%s] = %s' % (code_type, nb_classes))

print 'Tokenizing X_train'
X = {'train': {}, 'valid': {}, 'test': {}}
tokenizer = Tokenizer()
tokenizer.fit_on_texts(raw_train['NARRATIVE'])
tokenizer.word_index['BLANK_EMBEDDING'] = 0
X_train = tokenizer.texts_to_sequences(raw_train['NARRATIVE'])
X_valid = tokenizer.texts_to_sequences(raw_valid['NARRATIVE'])
X_test = tokenizer.texts_to_sequences(raw_test['NARRATIVE'])
X['train'] = pad_sequences(X_train, maxlen=max_len)
X['valid'] = pad_sequences(X_valid, maxlen=max_len)
X['test'] = pad_sequences(X_test, maxlen=max_len)
print('X_train shape:', X['train'].shape)
print('X_valid shape:', X['valid'].shape)
print('X_test shape:', X['test'].shape)

initial_embeddings = get_initial_embeddings(tokenizer.word_index)
vocab_size = len(tokenizer.word_index)
ngram_filters = [2, 3, 4, 5, 6]
filter_lengths = [900, 900, 900, 900, 900]

convolution_outputs = []

model = Graph()
model.add_input(name='input', input_shape=(vocab_size,), dtype='int')
model.add_node(layer=Embedding(vocab_size, embedding_size, 
                               weights=[initial_embeddings],
                               input_length=max_len,
                               dropout=0.5),
               name='embedding', input='input')
for n, n_gram in enumerate(ngram_filters):
    sequential = Sequential()
    sequential.add(Convolution1D(nb_filter=filter_lengths[n], 
                                 filter_length=ngram_filters[n],
                                 border_mode='valid',
                                 activation='relu',
                                 subsample_length=1,
                                 input_shape=(max_len, embedding_size)
                                 ))
    sequential.add(MaxPooling1D(pool_length=max_len - n_gram + 1))
    sequential.add(Flatten())
    model.add_node(sequential, name='unit_%s' % n_gram, input='embedding')
model.add_node(Dropout(0.5), name='dropout', 
               inputs=['unit_%s' % n for n in ngram_filters],
               merge_mode='concat')
losses = {}
for code_type in code_types:
    output_name = 'output_%s' % code_type
    model.add_node(Dense(output_dim=nb_classes[code_type], activation='softmax'),
                   name='dense_%s' % code_type, input='dropout')
    model.add_output(name=output_name, input='dense_%s' % code_type)
    losses[output_name] = 'categorical_crossentropy'

print 'compiling model'  
model.compile(loss=losses, optimizer='adam')
print 'fitting model'

def get_inputs_and_outputs_dict(dataset):
    inputs_and_outputs = {'input': X[dataset]}
    for code_type in code_types:
        inputs_and_outputs['output_%s' % code_type] = Y[dataset][code_type]
    return inputs_and_outputs

for epoch in range(15):
    print 'Epoch: %s' % epoch
    model.fit(get_inputs_and_outputs_dict('train'), nb_epoch=1, 
              batch_size=batch_size, verbose=1, validation_split=0.05)
    train_accuracy = accuracy(model=model, x_test=random.sample(X['train']['ACTIVITY_CD'], 2500), 
                              y_test=Y['train']['ACTIVITY_CD'], 
                              output_name='output_ACTIVITY_CD')
    print 'train_accuracy: %s' % train_accuracy
    valid_accuracy = accuracy(model=model, x_test=X['valid'], 
                              y_test=Y['valid']['ACTIVITY_CD'], 
                              output_name='output_ACTIVITY_CD')
    print 'val_accuracy: %s' % valid_accuracy

score = model.evaluate(get_inputs_and_outputs_dict('test'), batch_size=batch_size)
print accuracy(model=model, x_test=X['test'], y_test=Y['test']['ACTIVITY_CD'], 
               output_name='output_ACTIVITY_CD')
