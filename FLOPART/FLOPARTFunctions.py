import sys
import numpy as np
import pandas as pd
import FLOPARTInterface
np.set_printoptions(threshold=sys.maxsize)

defaultWeight = 1


def runFLOPART(data, labels, penalty, weights=-1):
    if weights == -1:
        weight_vec = np.full(len(data), defaultWeight, dtype=np.double)
        weights = defaultWeight
    elif isinstance(weights, np.ndarray):
        weight_vec = weights
        weights = np.average(weights)
    else:
        if weights <= 0:
            raise ValueError('Weight value must be greater than ')
        weight_vec = np.full(len(data), weights, dtype=np.double)

    lenData = len(data)
    upscale = False

    if isinstance(data, list):
        data = np.array(data).astype(np.intc)
    elif not isinstance(data, np.intc):
        upscale = True
        data = (data * weights).to_numpy(dtype=np.intc)

    flopartOutput = FLOPARTInterface.interface(data,
                                               weight_vec,
                                               lenData,
                                               (labels['start'] - 1).to_numpy(dtype=np.intc),
                                               (labels['end'] - 1).to_numpy(dtype=np.intc),
                                               (labels['change'].to_numpy(dtype=np.intc)),
                                               len(labels.index),
                                               penalty)

    print(flopartOutput.keys())

    print('cost\n', flopartOutput['cost'])
    print('end\n', flopartOutput['end'])
    print('mean\n', flopartOutput['mean'])
    print('intervals_mat\n', flopartOutput['intervals_mat'])
    print('state_vec\n', flopartOutput['state_vec'])


def runSlimFLOPART(data, labels, penalty, n_updates=-1, penalty_unlabeled=-1):
    if not float(n_updates).is_integer():
        raise Exception
    if n_updates < 1:
        n_updates = len(data)

    lenData = len(data)

    penalty_labeled = penalty
    if not penalty_unlabeled > 0:
        penalty_unlabeled = penalty

    if isinstance(data, list):
        data = np.array(data).astype(np.double)

    lopartOutput = FLOPARTInterface.interface(data,
                                              lenData,
                                              (labels['start'] - 1).to_numpy(dtype=np.intc),
                                              (labels['end'] - 1).to_numpy(dtype=np.intc),
                                              labels['change'].to_numpy(dtype=np.intc),
                                              len(labels.index),
                                              penalty_labeled,
                                              penalty_unlabeled,
                                              n_updates)

    outputdf = pd.DataFrame(lopartOutput)

    outputLen = len(outputdf.index)

    addOne = outputdf['last_change'] + 1

    changeVec = addOne[0 <= addOne]

    segmentsTemp = outputdf[outputdf['last_change'] != -2]

    starts = [1, ]

    changeStarts = (changeVec + 1).tolist()

    starts.extend(changeStarts)

    ends = changeVec.tolist()

    ends.extend([outputLen])

    segmentRanges = {'start': starts, 'end': ends}

    segmentsdf = pd.DataFrame(segmentRanges)

    outputSegments = segmentsdf[segmentsdf['start'] < segmentsdf['end']].copy()

    heights = segmentsTemp['mean'].tolist()

    outputSegments['height'] = heights

    return outputSegments
