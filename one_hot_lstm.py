# -*- coding: utf-8 -*-
"""
Work in progress to use graph for bidirectional LSTM

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
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Merge
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2

from embed_preprocessing import get_initial_embeddings
from msha_extractor import get_data

#output needs to be (nb_samples, timesteps, input_dim)
def generate_seq_to_one_hot(X, Y, vocab_size, batch_size):
    n_samples = len(X)
    seq_len = len(X[0])
    start = 0
    while 1:
        stop = start + batch_size
        chunk = X[start: stop]
        slices = []
        for i, seq_indexes in enumerate(chunk):
            x_slice = np.zeros([seq_len, vocab_size])
            x_slice[np.arange(seq_len), seq_indexes] = 1
            slices.append(x_slice)
        x_out = np.stack(slices, axis=0)
        y_out = Y[start: stop]
        start += batch_size
        if (start + batch_size) > n_samples:
            print 'reshuffling, %s + %s > %s' % (start, batch_size, n_samples)
            remaining_X = X[start: start + batch_size]
            remaining_Y = Y[start: start + batch_size]
            random_index = np.random.permutation(n_samples)
            X = np.vstack((remaining_X, X[random_index, :]))
            Y = np.vstack((remaining_Y, Y[random_index, :]))
            start = 0
            n_samples = len(X)
        yield x_out, y_out

max_len = 100
gen_batch_size = 10

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
tokenizer = Tokenizer(nb_words=10000)
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


vocab_size = len(tokenizer.word_index)

model = Graph()
model.add_input(name='input', input_shape=(vocab_size,), dtype='int')
model.add_node(layer=LSTM(100, dropout_W=0.5, dropout_U=0.5, 
                          return_sequences=True, 
                          input_shape=(max_len, vocab_size)),
               name='lstm_left', input='input')
model.add_node(layer=LSTM(100, dropout_W=0.5, dropout_U=0.5, 
                          return_sequences=True, go_backwards=True,
                          input_shape=(max_len, vocab_size)),
               name='lstm_right', input='input')
model.add_node(layer=MaxPooling1D(pool_length=max_len), name='maxpooling', inputs=['lstm_left', 'lstm_right'])
model.add_node(layer=Flatten(), name='flatten', input='maxpooling')
model.add_node(layer=Dropout(0.5), name='dropout', input='flatten')
model.add_node(layer=Dense(output_dim=nb_classes, activation='softmax'), 
               name='dense', input='flatten')
model.add_output(name='output', input='dense')
model.compile(loss='categorical_crossentropy', optimizer='adam')
              

train_generator = generate_seq_to_one_hot(X_train, Y_train, vocab_size, batch_size=gen_batch_size)
valid_generator = generate_seq_to_one_hot(X_valid, Y_valid, vocab_size, batch_size=gen_batch_size)
model.fit_generator(generator=train_generator, samples_per_epoch=len(X_train), nb_epoch=30,
                    show_accuracy=True, validation_data=valid_generator, nb_val_samples=len(X_valid))

test_generator = generate_seq_to_one_hot(X_test, Y_test, vocab_size, batch_size=gen_batch_size)
score = model.evaluate_generator(generator=test_generator, val_samples=len(X_test), 
                                 show_accuracy=True)
print('Test score:', score[0])
print('Test accuracy:', score[1])