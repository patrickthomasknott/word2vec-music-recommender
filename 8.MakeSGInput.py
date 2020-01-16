###########################
###########################
# User Inputs
fileLocation = '/'  # location of input and output files
windowSize = 3  # target word +/-
negativeSamples = 10  # negative samples per positive sample
###########################
###########################
###########################
from numpy.random import seed
from tensorflow import set_random_seed
import pickle
import numpy as np
from keras.preprocessing.sequence import skipgrams
import time

''' 
This script creates input-prediction pairs, with a separate boolean to indicate if a positive or negative sample.

Data inputs
•	songIndices.txt
•	listenLengths.txt
Data outputs
•	SGcouples... .txt
•	SGlabels... .txt
Printed outputs:
•	timings
'''

set_random_seed(2)
seed(1)
totalStart = time.time()


dataFile = fileLocation + 'songIndices.txt'
with open(dataFile, "rb") as myFile:
    songIndices = pickle.load(myFile)

dataFile = fileLocation + 'listenLengths.txt'
with open(dataFile, "rb") as myFile:
    listenLengths = pickle.load(myFile)

couples = list()
labels = list()
vocabSize = max(songIndices)+1
firstIndex = 0
lastIndex = 0
for i in range(listenLengths.shape[0]):
    start = time.time()
    lastIndex += listenLengths[i, 1]
    tempCouples, tempLabels = skipgrams(songIndices[firstIndex:lastIndex],  vocabSize, \
                                        window_size=windowSize, sampling_table=None, \
                                        negative_samples=negativeSamples, shuffle=True, seed=0)

    couples.extend(tempCouples)
    labels.extend(tempLabels)
    firstIndex = lastIndex
    print("finished skipgrams:", time.time() - start, "\nListen session", i, "of", listenLengths.shape[0])

couplesArray = np.asarray(couples)
labelsArray = np.asarray(labels)

dataFile = fileLocation + 'SGcouplesW' + str(windowSize) + "NS" + str(negativeSamples) + ".txt"
with open(dataFile, "wb") as myFile:
    pickle.dump(couplesArray, myFile)
dataFile = fileLocation + 'SGlabelsW' + str(windowSize) + "NS" + str(negativeSamples) + ".txt"
with open(dataFile, "wb") as myFile:
    pickle.dump(labelsArray, myFile)
print("total run time:", time.time() - totalStart)

