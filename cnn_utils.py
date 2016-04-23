# -*- coding: utf-8 -*-
"""
Tools for building convolutional neural networks for text

Created on Sun Jul 05 17:58:38 2015

@author: ameasure
"""

def get_max_len(X):
    """ X is a list of lists of word indexes
    """
    max_len = 0
    for row in X:
        length = len(row)
        if length > max_len:
            max_len = length
    return max_len
    
def get_avg_len(X):
    n_units = len(X)
    ct = 0
    for row in X:
        ct += len(row)
    return ct / float(n_units)
    
def get_new_token_id(X):
    token_ids = set()
    for row in X:
        for element in row:
            token_ids.add(element)
    return max(token_ids) + 1
    
def pad_sequence(X, pad_token_id, length):
    print 'received input of len %s' % len(X)
    print 'output length set to %s' % length
    print 'pad_token_id set to %s' % pad_token_id
    padded_X = []
    for row in X:
        row = row[:length]
        while len(row) < length:
            row.append(pad_token_id)
        padded_X.append(row)
    return padded_X
        
def prepare_sequence(X, length=None):
    pad_token_id = get_new_token_id(X)
    if not length:
        length = get_max_len(X)
        print 'no length specified, setting length to %s' % length
    return pad_sequence(X, pad_token_id, length)