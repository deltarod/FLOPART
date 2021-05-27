import os
import math
import pandas as pd

# Debug for more info when printing
pd.set_option('display.max_rows', 10000)


# https://arxiv.org/pdf/2012.06848.pdf Section 2 Probability Distributions
def anscombeTransformation(data, column='mean'):
    if isinstance(data, pd.DataFrame):
        data['anscombe'] = data[column].apply(anscombeApply)

    elif isinstance(data, pd.Series):
        return data.apply(anscombeApply)


def anscombeApply(thing):
    return math.sqrt(thing + 3/8)


sampleData = os.path.join('data', 'sampleData.csv')
testData = pd.read_csv(sampleData, sep='\t', header=None).fillna(0).T
testData.columns = ['data']
testData = testData['data']

output = anscombeTransformation(testData)

print(output)