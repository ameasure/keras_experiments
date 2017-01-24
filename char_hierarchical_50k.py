# -*- coding: utf-8 -*-
'''
Goal: use character generated word embeddings

Results:
    batch=16
    char_seq = 15
    word_seq = 15
    sent_seq = 4
    hidden = 128
    n_train = 10k
    maxed at ~47% val acc after many epochs

Next experiment:
    -expand n_train to 50k and batch_size to 48
    -keep original model weights 
    -@ epoch 36 reached 63.37 with GRU before computer froze

Up batch size to 60


'''
import os
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, Embedding, Input, TimeDistributed
from keras.layers import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.layers.pooling import GlobalMaxPooling1D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import LabelEncoder
import nltk

from msha_extractor import get_data
from attention import Attention

max_char_seq = 15
max_word_seq = 15  # cut texts after this number of words (among top max_features most common words)
max_sent_seq = 4
word_lstm_hidden = 128
sent_lstm_hidden = 128
doc_lstm_hidden = 128
embedding_dim = 16
batch_size = 80

print('Loading data...')
raw_train, raw_valid, _ = get_data(n_train=50000, n_valid=1000, n_test=10000)

#def remove_new_labels(df, label_field, new_labels):
#    print('df has %s rows' % len(df))
#    df = df[~df[label_field].isin(new_labels)]
#    print('reduced to %s rows' % len(df))
#    return df
#
#new_labels = ['005', '011', '045', '068']
#raw_train = remove_new_labels(raw_train, 'ACTIVITY_CD', new_labels)
#raw_valid = remove_new_labels(raw_valid, 'ACTIVITY_CD', new_labels)
    
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(raw_train['NARRATIVE'])
max_features = len(tokenizer.word_index)
print('max_features', max_features)

def preprocess(texts, tokenizer):
    n_texts = len(texts)
    output = np.zeros(shape=(n_texts, max_sent_seq, max_word_seq, max_char_seq), dtype=np.int)
    for text_idx, text in enumerate(texts):
        sentences = nltk.sent_tokenize(text)
        for sentence_idx, sentence in enumerate(sentences):
            words = nltk.word_tokenize(sentence)
            for word_idx, word in enumerate(words):
                chars = tokenizer.texts_to_sequences([word])[0]
                for char_idx, char in enumerate(chars):
                    sentence_criteria = (sentence_idx < max_sent_seq)
                    word_criteria = (word_idx < max_word_seq)
                    char_criteria = (char_idx < max_char_seq)
                    if sentence_criteria and word_criteria and char_criteria:
                        output[text_idx, sentence_idx, word_idx, char_idx] = char
    return output

X_train = preprocess(raw_train['NARRATIVE'], tokenizer)
X_test = preprocess(raw_valid['NARRATIVE'], tokenizer)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

code_type='ACTIVITY_CD'
labeler = LabelEncoder()
labels = set(raw_train[code_type].tolist() + raw_valid[code_type].tolist())
labeler.fit(list(labels))
nb_classes = len(set(labels))
print('nb_classes = %s' % nb_classes)

y_train = labeler.transform(raw_train[code_type])
Y_train = np_utils.to_categorical(y_train, nb_classes)

y_valid = labeler.transform(raw_valid[code_type])
Y_valid = np_utils.to_categorical(y_valid, nb_classes)


print('Build model...')
# char embedding layer
input_layer = Input(shape=(max_char_seq,), dtype='int32', name='input_layer')
embedding = Embedding(input_dim=max_features + 1, output_dim=embedding_dim, 
                      input_length=max_char_seq, name='char_embedding')(input_layer)

# word encoder
word_lstm = Bidirectional(GRU(word_lstm_hidden, dropout_W=0.5, dropout_U=0.5,
                               return_sequences=False, name='word_lstm'), 
                          merge_mode='concat')(embedding)
word_encoder = Model(input=input_layer, output=word_lstm)

# sentence encoder
sentence_input = Input(shape=(max_word_seq, max_char_seq), dtype='int32')
encoded_words = TimeDistributed(word_encoder, name='encoded_words')(sentence_input)
lstm = Bidirectional(GRU(sent_lstm_hidden, dropout_W=0.5, dropout_U=0.5, 
                          return_sequences=True, name='sentence_lstm'), merge_mode='concat')(encoded_words)
attention = Attention(2*sent_lstm_hidden, name='attention')(lstm)
sentence_encoder = Model(input=sentence_input, output=attention)

# document encoder
document_input = Input(shape=(max_sent_seq, max_word_seq, max_char_seq), dtype='int32')
encoded_sentences = TimeDistributed(sentence_encoder, name='encoded_sentences')(document_input)
doc_lstm = Bidirectional(GRU(doc_lstm_hidden, dropout_W=0.5, dropout_U=0.5, 
                              return_sequences=True, name='document_lstm'), merge_mode='concat')(encoded_sentences)
doc_attention = Attention(2*doc_lstm_hidden, name='doc_attention')(doc_lstm)
softmax = Dense(output_dim=nb_classes, activation='softmax', name='output_softmax')(doc_attention)
model = Model(input=document_input, output=softmax)

#model = load_model(r'C:\Users\Alex Measure\Desktop\Programming Projects\keras_experiments_2017\char_hierarchical\char_hierarchical_10k.hdf5')
print('Train...')
optimizer = Adam(clipnorm=5.)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
save_path = os.path.join(r'G:\Checkpoints\char_hierarchical_50k', '{epoch:02d}--{val_acc:.3f}.hdf5')
checkpointer = ModelCheckpoint(filepath=save_path, monitor='val_acc', 
                               save_best_only=True, verbose=1)
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=150,
          validation_data=(X_test, Y_valid), callbacks=[checkpointer])
