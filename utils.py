# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 18:35:08 2016

@author: MEASURE_A
"""
import sys
import os
import numpy as np
import random
import pandas as pd
import re
from sklearn.externals import joblib

import cac_net.constants

RE_PERIOD = re.compile('\.')
RE_NOT_ALPHA_NUM = re.compile('[^a-z0-9 ]')
RE_EXTRA_SPACES = re.compile(' {2,}')
RE_LEFT_OR_RIGHT = re.compile(r'\bleft\b|\bright\b|\bl\b|\br\b')


def normalize(text):
    """Converts a string of text into a list of normalized words.
    >>> normalize("A dog can't do 15 things at once!")
    [u'a', u'dog', u'can', u't', u'do', u'15', u'things', u'at', u'once']

    Parameters:
    ----------
    text: string

    Returns:
    -------
    list of normalized words
    """
    # Remove periods
    text = re.sub(RE_PERIOD, '', text)
    # Replace anything that isn't a letter or number with a space
    text = re.sub(RE_NOT_ALPHA_NUM, ' ', text.lower())
    # Replace 2 or more spaces with a single space
    text = re.sub(RE_EXTRA_SPACES, ' ', text.strip())
    # Separate words by spaces
    return text.split(' ')


class Labeler:
    def __init__(self):
        pass
    
    def fit(self, labels):
        labels = set([i for i in labels])
        self.label_map = {}
        for n, label in enumerate(labels):
            self.label_map[label] = n
    
    def transform(self, labels):
        transformed = []
        for label in labels:
            transformed.append(self.label_map[label])
        transformed = np.array(transformed)
        one_hot = np.zeros(shape=(len(labels), len(self.label_map)))
        one_hot[np.arange(len(labels)), transformed] = 1
        return one_hot
        
 
class WordTokenizer:
    def __int__(self):
        pass
    
    def fit_on_texts(self, texts):
        self.tokens = set()
        for text in texts:
            for word in normalize(text):
                self.tokens.add(word)
        self.token_map = {}
        for n, token in enumerate(self.tokens):
            self.token_map[token] = n
    
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = []
            for word in normalize(text):
                if word in self.tokens:
                    sequence.append(self.token_map[word])
            sequences.append(sequence)
        return sequences
        
        
class CharTokenizer:
    def __int__(self):
        pass
    
    def fit_on_texts(self, texts):
        self.characters = set()
        for text in texts:
            for char in text:
                self.characters.add(char)
        self.character_map = {}
        for n, character in enumerate(self.characters):
            self.character_map[character] = n
    
    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = []
            for character in text:
                sequence.append(self.character_map[character])
            sequences.append(sequence)
        return sequences


class NAICSTokenizer:
    def __init__(self):
        pass        

    def fit_on_texts(self, naics_codes):
        for naics_code in naics_codes:
            assert len(naics_code) == 6, 'naics code must be exactly 6 digits long, found: %d' % len(naics_code)
            assert naics_code.isdigit(), 'naics code must be a number, found: %s' % naics_code
        
    def texts_to_sequences(self, naics_codes):
        outputs = []
        for naics_code in naics_codes:
            assert len(naics_code) == 6, 'naics code must be exactly 6 digits long, found: %d' % len(naics_code)
            assert naics_code.isdigit(), 'naics code must be a number, found: %s' % naics_code
            arrays = []
            for digit in naics_code:
                array = np.zeros(shape=(1,10))
                array[0, int(digit)] = 1
                arrays.append(array)
            outputs.append(np.hstack(arrays))
        return np.vstack(outputs)

class GenericTokenizer:
    def __init__(self, extractors):
        """ extractors - list of functions that extract lists of features
                from iterable inputs. 
        """
        self.extractors = extractors
        
    def fit_on_rows(self, iterable):
        self.tokens = set()
        for i in iterable:
            for extractor in self.extractors:
                self.tokens.add(extractor(i))
        self.token_map = {}
        self.index_map = {}
        for n, token in enumerate(self.tokens):
            self.token_map[token] = n
            self.index_map[n] = token

    def rows_to_matrix(self, iterable):
        x = np.zeros(len(iterable), len(self.token_map))
        for i in iterable:
            for extractor in self.extractors:
                tokens = extractor(i)
                for token in tokens:
                    token_index = self.token_map[token]
                    x[i][token_index] = 1
        return x
        
    def rows_to_sequences(self, iterable):
        sequences = []
        for i in iterable:
            sequence = []
            for extractor in self.extractors:
                for token in extractor(i):
                    sequence.append(token)
            sequences.append(sequence)
        return sequences            
                          
                          
# define the generator that will create one hot outputs on the fly
def generate_one_hot(X, Y, vocab_size, batch_size):
    """
    Inputs:
    X: [n_samples, timesteps] each value is the index of a token
    Y: [n_samples, n_categories]
    
    Returns: training tuple of x_batch [batch_size, n_timesteps, vocab_size] and y_batch [batch_size, n_categories]
    """
    if not hasattr(Y, 'shape'):
        Y = np.asarray(Y)
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
            print('reshuffling, %s + %s > %s' % (start, batch_size, n_samples))
            remaining_X = X[start: start + batch_size]
            remaining_Y = Y[start: start + batch_size]
            random_index = np.random.permutation(n_samples)
            X = np.concatenate((remaining_X, X[random_index]), axis=0)             
            Y = np.concatenate((remaining_Y, Y[random_index]), axis=0)
            start = 0
            n_samples = len(X)
        yield (X_out, Y_out)

# define the generator that will create one hot outputs on the fly
def generate_plus(X_indexes, X_vectors, Y_vectors, vocab_sizes, batch_size):
    """
    We use a generator to create one hot inputs when working with large vocabularies
    because tensorflow does not (to my knowledge) yet support sparse matrix inputs
    and dense inputs would exceed memory capabilities if the full dataset
    was loaded into memory. We resolve this problem by generating dense
    inputs on the fly in small batches.
    
    Inputs:
    X_indexes: dictionary of [n_samples, timesteps] matrices where each matrix
        value is the index of a token. This function converts each of these 
        matrices to [batch_size, n_timesteps, vocab_size] matrices for input 
        to an RNN.
        Ex: {'occupation_text': X_occ_text, 'narrative_text': X_nar_text}
    X_vectors: dictionary of [n_samples, n_features] matrices where each 
        matrix value is the value of a feature. We do not modify these, only 
        divide them into batches that match X_indexes.
        Ex: {'naics': X_naics, 'fips_state_code': X_state_code}
    Y_vectors: dictionary of [n_samples, n_categories] output matrices. 
        Ex: {'soc': Y_soc, 'nature_code': Y_nature_code, 'part_code': Y_part_code}
    vocab_sizes: dictionary of vocabulary sizes for the X_index matrices. Each
        key links to a key in the X_indexes dictionary and the value defines
        the corresponding vocabulary size.
        
    Returns: training batch tuple containing 2 dictionaries, one for inputs, one for outputs.
        Ex: ({'occupation_text': X_occ_text, 'naics': X_naics}, 
             {'soc': Y_soc, 'nature_code': Y_nature})
    """
    # Make sure all Y_vectors are numpy arrays
    if Y_vectors:
        for name, matrix in Y_vectors.items():
            if not hasattr(matrix, 'shape'):
                Y_vectors[name] = np.asarray(matrix)
    # Verify that all input and output matrices (where present) are equal length
    named_matrices = {}
    for input_name, input_value in [('X_indexes', X_indexes), 
                                    ('X_vectors', X_vectors), 
                                    ('Y_vectors', Y_vectors)]:
        if input_value:
            msg = 'dict expected for %s, %s found' % (input_name, type(input_value))
            assert(type(input_value)) == dict, msg
            for name, matrix in input_value.items():
                named_matrices[name] = matrix
    matrix_items = [(name, matrix) for name, matrix in named_matrices.items()]
    base_matrix_name, base_matrix = matrix_items[0]
    conditions = [base_matrix.shape[0] == matrix.shape[0] for matrix in named_matrices.values()]
    msg = 'Not all input matrices have the same number of observations (rows)'
    matrix_sizes = '\n'.join(['%s %s' % (name, matrix.shape) for name, matrix in named_matrices.items()])
    assert all(conditions), msg + '\n' + matrix_sizes
    n_samples = base_matrix.shape[0]
    seq_lens = {name: matrix.shape[1] for name, matrix in X_indexes.items()} 
    start = 0
    while 1:
        stop = start + batch_size
        X_out = {}
        Y_out = {}
        # Check if our batch is going to exceed remaining samples
        # If it is, reshuffle and reset the start and stop
        if stop > n_samples:
            # reshuffle the order of our matrices
            random_index = np.random.permutation(n_samples)
            for matrix_type in [X_indexes, X_vectors, Y_vectors]:
                if matrix_type:
                    for name, matrix in matrix_type.items():
                        matrix_type[name] = matrix_type[name][random_index]
            # reset our batch start and end points            
            start = 0
            stop = start + batch_size         
        # Generate the one hot outputs
        for name, matrix in X_indexes.items():
            seq_len = seq_lens[name]
            vocab_size = vocab_sizes[name]
            matrix_subset = matrix[start: stop]
            matrix_out = np.zeros([batch_size, seq_len, vocab_size])
            index_1 = np.repeat(np.arange(batch_size), seq_len).reshape(batch_size, seq_len)
            index_2 = np.arange(seq_len)
            matrix_out[index_1, index_2, matrix_subset] = 1
            X_out[name] = matrix_out
        # Generate slices of the other input matrices
        if X_vectors:
            for name, matrix in X_vectors.items():
                X_out[name] = matrix[start: stop]
        if Y_vectors:
            for name, matrix in Y_vectors.items():
                Y_out[name] = matrix[start: stop]
        # Increment start
        start = stop
        yield (X_out, Y_out)
        
        
def get_train_test(n_train, n_test, 
                   required_fields=cac_net.constants.CODE_TYPES, 
                   data_source=None,
                   n_rows=None):
    """
    Retrieve examples for training and test.
    """
    print('retrieving rows from %s' % data_source)
    # python 2/3 compatibility
    if sys.version_info[0] >= 3:
        data_file = open(data_source, 'r')
    else:
        data_file = open(data_source, 'rU')
    try:
        df = pd.read_csv(data_file, encoding='latin-1', dtype=object, nrows=n_rows)
    except pd.parser.CParserError:
        df = pd.read_csv(data_source, encoding='latin-1', dtype=object, nrows=n_rows)
    df.dropna(subset=[required_fields], inplace=True)
    df.fillna(value='', inplace=True)
    rows = df.to_dict(orient='records')
    print('%d rows loaded' % len(rows))
    # Randomize
    random.seed(31)
    random.shuffle(rows)
    train = rows[0: n_train]
    test = rows[n_train: n_train + n_test]
    print('%d rows retrieved for training' % len(train))
    print('%d rows retrieved for testing' % len(test))
    return train, test

    
def clean_and_shuffle_data_source(data_source):
    dirname, fname = os.path.split(data_source)
    df = pd.read_csv(data_source, encoding='latin-1', dtype=object)
    df.dropna(subset=[cac_net.constants.CODE_TYPES], inplace=True)
    df.fillna(value='', inplace=True)
    df = df.reindex(np.random.permutation(df.index))
    out_fname = 'clean_%s' % fname
    df.to_csv(os.path.join(dirname, out_fname), encoding='utf-8', index=False)
    
    
def valid_codes(label_field, code_detail=None):
    """ Return the list of valid codes for a specific label field
    """
    type_map = {'socc' : 'soc',
                'part' : 'part_code',
                'even' : 'event_code',
                'sour' : 'source_code',
                'natu' : 'nature_code'}
    if label_field == 'sec_source_code':
        label_field = 'source_code'
    code_set = set()
    df = pd.read_csv(os.path.join(cac_net.constants.DATA_DIR, 'code_set.csv'))
    rows = df.to_dict(orient='records')
    for row in rows:
        code_type = type_map[row['code_type']]
        code = row['code'][0: code_detail]
        if code_type == label_field:
            code_set.add(code)
    return list(code_set)
    
def dump_tokenizers(tokenizers, dir_name):
    output_dir = os.path.join(cac_net.constants.CHECKPOINT_DIR, dir_name)
    # Make the directories if they don't exist
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    out_path = os.path.join(output_dir, 'tokenizers.pkl')
    joblib.dump(tokenizers, out_path)
    print('dumping tokenizers to %s' % out_path)

def load_tokenizers(dir_name=None):
    input_dir = os.path.join(cac_net.constants.CHECKPOINT_DIR, dir_name)
    input_path = os.path.join(input_dir, 'tokenizers.pkl')
    print('loading tokenizers from %s' % input_path)
    return joblib.load(input_path)  