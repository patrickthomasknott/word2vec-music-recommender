###########################
###########################
# User Inputs
fileLocation = '/'  # location of input and output files
windowSize = 3  # target word +/-
negativeSamples = 10  # negative samples per positive sample
neurons = 40  # list of values to test
sgBatchSize = 1000  # list of values to test
cbowBatchSize = 500  # list of values to test
SGepochs = 5
CBOWepochs = 2
song2compare = "Daft Punk: Burnin' / Too Long"  # song to find closest songs to in both SG and CBOW
###########################
###########################
import pickle
import pandas as pd
import collections
from keras.models import load_model
import numpy as np
import operator
import time

start = time.time()
'''
This script is used to qualitatively test the models by finding the closest songs to a particular input song in both 
vector spaces. NB target song is hardcoded as a string in user input section.

Data inputs
•	UserSonglist.csv
•	dictionary.txt
•	reverseDictionary.txt
•	SGmodel... .h5
•	CBOWmodel... .h5
Data outputs
•	none
Printed outputs:
•	closest songs to input song in each vector space
'''
dataFile = fileLocation + 'UserSonglist.csv'
df = pd.read_csv(dataFile, dtype='object', sep=',')

dataFile = fileLocation + 'dictionary.txt'
with open(dataFile, "rb") as myFile:
    dictionary = pickle.load(myFile)

dataFile = fileLocation + 'reverseDictionary.txt'
with open(dataFile, "rb") as myFile:
    reverseDictionary = pickle.load(myFile)
vocabSize = len(reverseDictionary) - 1

SGmodelName = 'SGmodelW' + str(windowSize) + "NS" + str(negativeSamples) + 'Neurons' + \
              str(neurons) + 'Batchsize' + str(sgBatchSize) + 'Epochs' + str(SGepochs)

CBOWmodelName = 'CBOWmodelW' + str(windowSize) + "NS" + str(negativeSamples) + 'Neurons' + \
                str(neurons) + 'Batchsize' + str(cbowBatchSize) + 'Epochs' + str(CBOWepochs)

sgLoadName = fileLocation + 'SGmodelW' + str(windowSize) + "NS" + str(negativeSamples) + 'Neurons' + \
             str(neurons) + 'Batchsize' + str(sgBatchSize) + 'Epochs' + str(SGepochs) + ".h5"

cbowLoadName = fileLocation + 'CBOWmodelW' + str(windowSize) + "NS" + str(negativeSamples) + 'Neurons' + \
               str(neurons) + 'Batchsize' + str(cbowBatchSize) + 'Epochs' + str(CBOWepochs) + ".h5"

sgModel = load_model(sgLoadName)

cbowModel = load_model(cbowLoadName)

songs = pd.Series(df.iloc[0][:]).dropna()
for i in range(1, df.shape[0]):
    songs = songs.append(pd.Series(df.iloc[i][:]).dropna(), ignore_index=True)

count = [['UNK', -1]]
count.extend(collections.Counter(songs).most_common(len(songs) - 1))

songDicNum = dictionary[song2compare]  # 1
sgInputSongWeights = sgModel.get_weights()[0][songDicNum - 1]  # -1 coz dictionary starts with 'Unk'
cbowInputSongWeights = cbowModel.get_weights()[0][songDicNum - 1]

print("Fidning the songs closest to:", song2compare + ", which has", count[songDicNum][1], "play-events.")

# using SG model
sgSongSimDic = {}
for i in range(vocabSize):
    targetSongWeights = sgModel.get_weights()[0][i]

    thetaSum = np.dot(sgInputSongWeights, targetSongWeights)
    thetaDen = np.linalg.norm(sgInputSongWeights) * np.linalg.norm(targetSongWeights)
    theta = thetaSum / thetaDen

    song = reverseDictionary[i + 1]  # coz dic starts at 1 but weights start at zero
    sgSongSimDic[song] = theta
sgSongsSorted = sorted(sgSongSimDic.items(), key=operator.itemgetter(1), reverse=True)
print("\nClosest 20 songs to", reverseDictionary[songDicNum], ' using the SkipGram model:')
for song, sim in sgSongsSorted[1:21]:
    print(song, sim)

# using CBOW model
cbowSongSimDic = {}
for i in range(vocabSize):
    targetSongWeights = cbowModel.get_weights()[0][i]

    thetaSum = np.dot(cbowInputSongWeights, targetSongWeights)
    thetaDen = np.linalg.norm(cbowInputSongWeights) * np.linalg.norm(targetSongWeights)
    theta = thetaSum / thetaDen

    song = reverseDictionary[i + 1]  # coz dic starts at 1 but weights start at zero
    cbowSongSimDic[song] = theta
cbowSongsSorted = sorted(cbowSongSimDic.items(), key=operator.itemgetter(1), reverse=True)
print("\nClosest 20 songs to", reverseDictionary[songDicNum], ' using the CBOW model:')
for song, sim in cbowSongsSorted[1:21]:
    print(song, sim)
print("Runtime:", time.time()-start)