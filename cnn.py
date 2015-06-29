# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 23:49:23 2015

@author: ameasure

To do:
Map Documents to Embeddings
1) Figure out max narrative size
2) Download embeddings
3) Load embeddings into gensim model
4) Embed documents and save to disk
5) Iteratively load documents, train keras conv net

"""
import datetime

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

import theano
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, Convolution1D, MaxPooling1D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils

from embed_preprocessing import vectorize, get_simple_model
from msha_extractor import get_data

#theano.config_device = 'gpu0'

def reduce_df(df):
    return df[(df['ACTIVITY_CD'] == '028') | (df['ACTIVITY_CD'] == '039')]


def classify():
    code_type = 'INJ_BODY_PART_CD'
    raw_train, raw_test = get_data(n_train=10000, n_test=1000)
#    raw_train = reduce_df(raw_train)
#    raw_test = reduce_df(raw_test)
    raw_train_labels = raw_train[code_type]

    labeler = LabelEncoder()
    labels = set(raw_train[code_type].tolist() + raw_test[code_type].tolist())
    labeler.fit(list(labels))
    y_train = labeler.transform(raw_train_labels)
    print y_train.shape
    nb_classes = len(set(labels))
    print('nb_classes = %s' % nb_classes)
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    
    simple_model = get_simple_model()
    X_train = vectorize(raw_train, simple_model)
    #X_train = np.random.randn(X_train.shape[0], X_train.shape[1])
    print X_train.shape
    n_features = int(X_train.shape[1])
#    newshape = (X_train.shape[0], 1, 1, X_train.shape[1])
#    X_train = np.reshape(X_train, newshape)
    X_train = make_4d(X_train)
    print X_train.shape

    BATCH_SIZE = 64
    INPUT_SIZE = n_features
    FIELD_SIZE = 3 * 300
    STRIDE = 300
    N_FILTERS = 100

#    model = Sequential()
#    model.add(Convolution1D(nb_filter=10, stack_size=1, filter_length=4))
#    model.add(Activation('relu'))
#    model.add(MaxPooling1D(pool_length=n_features - 4 + 1))
#    model.add(Dropout(0.5))
#    model.add(Flatten())
#    model.add(Dense(10, nb_classes, init='normal'))
#    model.add(Activation('softmax'))

    model = Sequential()
    model.add(Convolution2D(nb_filter=N_FILTERS, stack_size=1, nb_row=FIELD_SIZE, 
                            nb_col=1, subsample=(STRIDE, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(((INPUT_SIZE - FIELD_SIZE)/STRIDE) + 1, 1)))
    model.add(Dropout(0.5))
    model.add(Flatten())
#    model.add(BatchNormalization(N_FILTERS))
    model.add(Dense(N_FILTERS, nb_classes))
    model.add(Activation('softmax'))

#    model = Sequential()
#    model.add(Dense(n_features, 1500))
#    model.add(Activation('relu'))
#    model.add(BatchNormalization((1500)))
#    model.add(Dropout(0.5))
#    model.add(Dense(1500, nb_classes))
#    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')
    model.fit(X_train, Y_train, nb_epoch=20, batch_size=BATCH_SIZE, verbose=1, show_accuracy=True, validation_split=0.1)

    raw_test_labels = raw_test[code_type]
    X_test = vectorize(raw_test, simple_model)
    X_test = make_4d(X_test)
    y_test = labeler.transform(raw_test_labels)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    score = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE, show_accuracy=True)
    print score
    return model, X_test


def make_4d(array):
    array_rows = array.shape[0]
    array_cols = array.shape[1]
    newshape = (array_rows, 1, array_cols, 1)
    return np.reshape(array, newshape)
        
