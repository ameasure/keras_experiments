# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 11:52:10 2015

@author: ameasure
"""

import pandas as pd
import numpy as np
import datetime


def get_data(n_train=10000, n_valid=10000, n_test=10000):
    df = pd.io.parsers.read_csv('Accidents.txt', sep='|', parse_dates=['ACCIDENT_DT'])
    df = df[df['ACCIDENT_DT'] > datetime.datetime(2004, 1, 1)]
    random_state = np.random.RandomState(42)
    random_indexes = random_state.permutation(df.index.values)
    df = df.ix[random_indexes]
    train = df[0: n_train]
    valid = df[n_train: n_train + n_valid]
    test = df[n_train + n_valid: n_train + n_valid + n_test]
    return train, valid, test
