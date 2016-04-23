# -*- coding: utf-8 -*-
"""
Created on Fri Jul 03 21:30:26 2015

@author: ameasure
"""

import datetime

import pandas as pd
import numpy as np
np.random.seed(42)

from sklearn.preprocessing import LabelEncoder

import theano
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Merge
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2

from embed_preprocessing import get_initial_embeddings
from msha_extractor import get_data


n_filters = 400
max_len = 100
embedding_size = 300
batch_size = 128

code_type = 'ACTIVITY_CD'
raw_train, raw_test = get_data(n_train=50000, n_test=10000)
raw_train_labels = raw_train[code_type]

labeler = LabelEncoder()
labels = set(raw_train[code_type].tolist() + raw_test[code_type].tolist())
labeler.fit(list(labels))
nb_classes = len(set(labels))
print('nb_classes = %s' % nb_classes)
y_train = labeler.transform(raw_train_labels)
print 'y_train shape is:', y_train.shape
print 'Vectorizing labels'
Y_train = np_utils.to_categorical(y_train, nb_classes)
print 'y_train shape is:', Y_train.shape

raw_test_labels = raw_test[code_type]
y_test = labeler.transform(raw_test_labels)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print 'Tokenizing X_train'
tokenizer = Tokenizer()
tokenizer.fit_on_texts(raw_train['NARRATIVE'])
tokenizer.word_index['BLANK_EMBEDDING'] = 0
X_train = tokenizer.texts_to_sequences(raw_train['NARRATIVE'])
X_test = tokenizer.texts_to_sequences(raw_test['NARRATIVE'])
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

initial_embeddings = get_initial_embeddings(tokenizer.word_index)
vocab_size = len(tokenizer.word_index)

model = Sequential()
model.add(Embedding(vocab_size, embedding_size, weights=[initial_embeddings]))
model.add(GRU(embedding_size, 600, activation='sigmoid', inner_activation='hard_sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(600, nb_classes))
model.add(Activation("softmax"))

print 'compiling model'
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
print 'fitting model'


history = model.fit(X_train, Y_train, nb_epoch=200, batch_size=batch_size, 
                    verbose=1, show_accuracy=True, validation_split=0.025)


score = model.evaluate(X_test, Y_test, batch_size=batch_size, show_accuracy=True)
print('Test score:', score[0])
print('Test accuracy:', score[1])