from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import resample
import os


def flattenList(l=None):
    """
    flatting a nest list of lists to a list
    :param l: a nested list of lists
    :return: a list
    """
    flat_list = [item for sublist in l for item in sublist]
    return flat_list


def checkDir(dirName=None):
    """
    check if a given directory exist, if not make a new directory
    :param dirName: a given directory
    :return:
    """
    if not os.path.exists(dirName):
        os.makedirs(dirName)
    return 0

def indexAll(inputList=None, value=None):
    """
    find the index of a given value in a given list
    :param inputList: input as a list
    :param value: the value that targeted to index
    :return: the index of value lies in a list
    """
    if not isinstance(inputList, list):
        raise TypeError('Input list must be a list object.')
    return [i for i, x in enumerate(inputList) if x == value]


def dataToCsv(data=None,fileName=None, floatPrecision='%.10f'):
    """
    write the data to csv file
    :param data: data to write as numpy array
    :param fileName: the name of the file
    :param floatPrecision: the precision of the float
    :return:
    """
    df = pd.DataFrame(data=data)
    df.to_csv(path_or_buf=fileName,
              header=False, index=False,
              float_format=floatPrecision)
    return 0


# TODO: set the data directory

dataDir = './datasets/UJIIndoorLoc/trainingData/'

# TODO: load the training data (Raw)
# set to the filename
# data_type = 'training_data'
# data = pd.read_csv(filepath_or_buffer=dataDir + '1478167720_9233432_trainingData.csv', header=1).values
# rss = dataTrain[:, :-9]
# coor = dataTrain[:, -9:]

# # TODO: load the validation data (Raw) [uncomment this if for cleaning validation data]
data_type = 'validation_data'
data = pd.read_csv(filepath_or_buffer=dataDir + '1478167721_0345678_validationData.csv', header=1).values
rss = data[:, :-9]
coor = data[:, -9:]

# rss = np.concatenate((rssTrain, rssVal), axis=0)
# coor = np.concatenate((coorTrain, coorVal))

# TODO: clean the invalid measurement: all RSS values are indicated as missing
missingValue = 100.
zerosObsId = []
zerosObsCounter = 0
for rowId in range(rss.shape[0]):
    if all(rss[rowId, :] == missingValue):
        # print("ZeroObs: {}".format(rowId))
        zerosObsId.append(rowId)
        zerosObsCounter += 1
print('The number of invalid examples is {}.'.format((zerosObsCounter)))
rss = np.delete(rss, zerosObsId, axis=0)
coor = np.delete(coor, zerosObsId, axis=0)

# TODO: write to file: without invalid measurement
output_dir = dataDir + 'without_invalid_examples/'
checkDir(output_dir)
data = np.delete(data, zerosObsId, axis=0)
dataToCsv(data=data, fileName=output_dir + '{}_w_o_invalids.csv'.format(data_type))

# TODO: find all the measurements at the same location (replicas)
# Here we just take the location difference into account

rssB0 = rss[:, :]
coorB0 = np.concatenate((coor[:, :4], coor[:, 6:8]), axis=1)

mutualDistances = euclidean_distances(coorB0[:, :], coorB0[:, :])

replicasIndex = []
uniqueObsId = []
for pointId in range(rssB0.shape[0]):
    #     search the replicas row-wisely
    if pointId + 1 <= (rssB0.shape[0] - 1):
        if len(replicasIndex) == 0:
            uniqueObsId.append(pointId)
            tempIndex = indexAll(inputList=list(np.reshape(mutualDistances[pointId, pointId+1:], newshape=(-1, ))),
                                     value=0.0)
            if len(tempIndex) != 0:
                replicasIndex.append([a + pointId + 1 for a in tempIndex])
            else:
                replicasIndex.append([])
        else:
            if not (pointId in flattenList(replicasIndex)):
                uniqueObsId.append(pointId)
                tempIndex = indexAll(inputList=list(np.reshape(mutualDistances[pointId, pointId + 1:],
                                                                   newshape=(-1,))), value=0.0)
                if len(tempIndex) != 0:
                    replicasIndex.append([a + pointId + 1 for a in tempIndex])
                else:
                    replicasIndex.append([])
            else:
                replicasIndex.append([])


# TODO: processing scheme 1 - arbitrarily sample one of the replicas
finalTrainIds = []
for obsId in uniqueObsId:
    if len(replicasIndex[obsId]) != 0:
        tempList = [obsId]
        for replicasId in replicasIndex[obsId]:
            tempList.append(replicasId)
        tempVal = resample(tempList, replace=False, n_samples=1)
        finalTrainIds.append(tempVal[0])
    else:
        finalTrainIds.append(obsId)

dataTrain_without_replicas = data[finalTrainIds, :]

# TODO: write to file: without invalid measurement
output_dir = dataDir + 'without_replicated_examples/'
checkDir(output_dir)
dataToCsv(data=dataTrain_without_replicas, fileName=output_dir + '{}_w_o_replicas.csv'.format(data_type))
