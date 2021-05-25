import os
import sys
import pytest
import FLOPART
import numpy as np
import pandas as pd

# Debug for more info when printing
pd.set_option('display.max_rows', 10000)


sampleData = os.path.join('data', 'sampleData.csv')
data = pd.read_csv(sampleData, sep='\t', header=None).fillna(0).T
data.columns = ['data']
data = data['data']


def test_FLOPARTNoLabels():
    # Empty Labels
    labels = pd.DataFrame(columns=['start', 'end', 'change'])
    penalty = 10
    output = FLOPART.runFLOPART(data, labels, penalty)

    print(output)

    assert 1 == 0


def test_FLOPARTWithLabel():
    # Empty Labels
    labels = pd.DataFrame({'start': [786], 'end': [787], 'change': [1]})
    penalty = 10
    output = FLOPART.runFLOPART(data, labels, penalty)

    print(output)

    assert 1 == 0