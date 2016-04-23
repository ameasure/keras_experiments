# -*- coding: utf-8 -*-
"""
Created on Sun Jul 05 17:39:01 2015

@author: ameasure
"""

from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.datasets import reuters
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Merge
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer

import cnn_utils


vocab_size = 20000
batch_size = 128
embedding_size = 500
max_len = 100
n_filters = 300
n_gram = 6

print("Loading data...")
(X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=vocab_size, test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

nb_classes = np.max(y_train)+1
print(nb_classes, 'classes')

X_train = cnn_utils.prepare_sequence(X_train, length=max_len)
X_train = np.array(X_train)
X_test = cnn_utils.prepare_sequence(X_test, length=max_len)
X_test = np.array(X_test)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print("Convert class vector to binary class matrix (for use with categorical_crossentropy)")
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

print("Building model...")
model = Sequential()
model.add(Embedding(vocab_size + 1, embedding_size))
model.add(Reshape(1, max_len, embedding_size))
model.add(Convolution2D(nb_filter=n_filters, stack_size=1, nb_row=n_gram,
                        nb_col=embedding_size))
model.add(Activation("relu"))
model.add(MaxPooling2D(poolsize=(max_len - n_gram + 1, 1)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(n_filters, nb_classes, init='glorot_normal'))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')


history = model.fit(X_train, Y_train, nb_epoch=200, batch_size=batch_size, verbose=1, show_accuracy=True, validation_split=0.1)
score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1, show_accuracy=True)
print('Test score:', score[0])
print('Test accuracy:', score[1])