# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 23:36:34 2015

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
from keras.utils import np_utils
from keras.preprocessing.sequence import pad_sequences


vocab_size = 20000
batch_size = 128
embedding_size = 100
maxlen = 75
nb_feature_maps = 100

print("Loading data...")
(X_train, y_train), (X_test, y_test) = reuters.load_data(nb_words=vocab_size, test_split=0.2)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

nb_classes = np.max(y_train) + 1
print(nb_classes, 'classes')

X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print("Convert class vector to binary class matrix (for use with categorical_crossentropy)")
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

ngram_filters = [2, 4, 6]
conv_filters = []

for n_gram in ngram_filters:
    sequential = Sequential()
    conv_filters.append(sequential)

    sequential.add(Embedding(vocab_size + 1, embedding_size))
    sequential.add(Reshape(1, maxlen, embedding_size))
    sequential.add(Convolution2D(nb_feature_maps, 1, n_gram, embedding_size))
    sequential.add(Activation("relu"))
    sequential.add(MaxPooling2D(poolsize=(maxlen - n_gram + 1, 1)))
    sequential.add(Flatten())

model = Sequential()
model.add(Merge(conv_filters, mode='concat'))
model.add(Dropout(0.5))
model.add(Dense(nb_feature_maps * len(conv_filters), nb_classes, init='glorot_normal'))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='adam')

X_train_concat = []
X_test_concat = []
for ngram in ngram_filters:
    X_train_concat.append(X_train)
    X_test_concat.append(X_test)
model.fit(X=X_train_concat, y=Y_train, batch_size=batch_size, nb_epoch=20, verbose=1, show_accuracy=True, validation_split=0.1)

score = model.evaluate(X_test_concat, Y_test, batch_size=batch_size, verbose=1, show_accuracy=True)
print('Test score:', score[0])
print('Test accuracy:', score[1])