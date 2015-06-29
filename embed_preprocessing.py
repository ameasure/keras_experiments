# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 22:48:34 2015

@author: ameasure
"""
import numpy as np
import re
import cPickle as pickle

from gensim.models import Word2Vec
from msha_extractor import get_data


def get_model():
    f = r'C:\Users\ameasure\Desktop\Programming Projects\theano_test\GoogleNews-vectors-negative300.bin'
    model = Word2Vec.load_word2vec_format(f, binary=True)
    return model


def get_simple_model():
    return pickle.load(open('simple_model.pi', 'rb'))    


def get_vocabulary(raw):
    vocabulary = set([])
    rows = raw.to_dict(orient='records')    
    for row in rows:
        words = tokenize(row['NARRATIVE'])
        for word in words:
            vocabulary.add(word)
    return vocabulary
    
    
def make_simple_model(model, vocabulary):
    simple_model = {}
    for word in vocabulary:
        try:
            vector = model[word]
        except KeyError:
            print 'KeyError on %s, using random embedding instead' % word
            vector = get_random_embedding()
        simple_model[word] = vector
    simple_model['BLANK_EMBEDDING'] = np.zeros(300, dtype=np.float32)
    return simple_model


def make_and_save_simple_model():
    train, test = get_data(n_train=100000000, n_test=0)
    vocabulary = get_vocabulary(train)
    model = get_model()
    simple_model = make_simple_model(model=model, vocabulary=vocabulary)
    pickle.dump(simple_model, open('simple_model.pi', 'wb'))
    
    
TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")
def tokenize(document):
    return TOKEN_PATTERN.findall(document)


def vectorize(raw, model):
    rows = raw.to_dict(orient='records')
    embedded_documents = []
    for row in rows:
        words = tokenize(row['NARRATIVE'])
        vectors = []
        for word in words:
            vector = model[word]
            vectors.append(vector)
        while len(vectors) < 98:
            vector = model['BLANK_EMBEDDING']
            vectors.append(vector)
        embedded_documents.append(np.hstack(vectors).astype(np.float32))
    return np.vstack(embedded_documents)
    
    
EMBED_SIZE = 300
EMBED_MEAN = -.0017837381
EMBED_STD = .057699453      
def get_random_embedding():
    return np.random.normal(loc=EMBED_MEAN, scale=EMBED_STD, size=EMBED_SIZE)