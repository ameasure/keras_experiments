# -*- coding: utf-8 -*-
"""
This is an attempt to use a single word embedding for multiple filters by combining
the Graph and Sequential containers.

@author: ameasure
1626s/epoch gpu
4480s/epoch cpu (estimated)
"""

import datetime
import os
import numpy as np
np.random.seed(42)

from sklearn.preprocessing import LabelEncoder
import keras
from keras.models import Model
from keras.engine.topology import Merge
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Reshape, merge
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from msha_extractor import get_data

#output needs to be (nb_samples, timesteps, input_dim)
def generate_one_hot(X, Y, vocab_size, batch_size):
    """
    Inputs:
    X: [n_samples, timesteps] each value is the index of a token
    Y: [n_samples, n_categories] already one hot
    
    Returns: dictionary with 'input': [n_samples, n_timesteps, vocab_size] and 'output': [n_samples, n_categories]
    """
    n_samples = len(X)
    seq_len = len(X[0])
    start = 0
    while 1:
        stop = start + batch_size
        X_subset = X[start: stop]
        X_out = np.zeros([batch_size, seq_len, vocab_size])
        index_1 = np.repeat(np.arange(batch_size), seq_len).reshape(batch_size, seq_len)
        index_2 = np.arange(seq_len)
        X_out[index_1, index_2, X_subset] = 1
        Y_out = Y[start: stop]
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
        yield X_out, Y_out


max_len = 100
gen_batch_size = 16
checkpoint_dir = r'C:\Users\ameasure\Desktop\Programming Projects\cnn\checkpoints'

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
tokenizer = Tokenizer(nb_words=5000)
tokenizer.fit_on_texts(raw_train['NARRATIVE'])
tokenizer.word_index['BLANK_WORD'] = 0
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

print 'specifying model'

input_layer = Input(shape=(max_len, vocab_size), dtype='float32') # [n_samples, n_steps, n_features]
lstm_left = LSTM(output_dim=10, dropout_W=0.5, dropout_U=0.5, 
                 return_sequences=True, consume_less='mem')(input_layer) # [n_samples, n_steps, n_units]
lstm_right = LSTM(output_dim=10, dropout_W=0.5, dropout_U=0.5, 
                  return_sequences=True, consume_less='mem', go_backwards=True)(input_layer)
merged_layer = merge([lstm_left, lstm_right], mode='concat', concat_axis=2) # [n_samples, ]
pooling = MaxPooling1D(pool_length=max_len)(merged_layer)
flatten = Flatten()(pooling)
batch_norm = BatchNormalization()(flatten)
dropout = Dropout(0.5)(flatten)
output = Dense(output_dim=nb_classes, activation='softmax')(dropout)
print 'instantiate model'
model = Model(input=input_layer, output=output)
print 'compiling model'
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
         
checkpointer = keras.callbacks.ModelCheckpoint(filepath=os.path.join(checkpoint_dir, 'weights.{epoch:02d}--{val_acc:.3f}.hdf5'), 
                                               monitor='val_acc', 
                                               verbose=1,
                                               save_best_only=False)
train_generator = generate_one_hot(X_train, Y_train, vocab_size, batch_size=gen_batch_size)
valid_generator = generate_one_hot(X_valid, Y_valid, vocab_size, batch_size=gen_batch_size)
print 'fitting model'
model.fit_generator(generator=train_generator, 
                    samples_per_epoch=len(X_train), 
                    nb_epoch=60,
                    validation_data=valid_generator, 
                    nb_val_samples=len(X_valid))

test_generator = generate_one_hot(X_test, Y_test, vocab_size, batch_size=gen_batch_size)
score = model.evaluate_generator(test_generator, len(X_test), verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])