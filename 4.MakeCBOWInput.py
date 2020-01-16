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
import random
import time
import numpy as np

'''
This script creates input-prediction pairs, with a separate boolean to indicate if a positive or negative sample.

Data inputs
•	songIndices.txt
•	listenLengths.txt
Data outputs
•	CBOWcouples... .txt 
•	CBOWlabels... .txt 
Printed outputs:
•	timings

'''

set_random_seed(2)
seed(1)
start = time.time()

dataFile = fileLocation + 'songIndices.txt'
with open(dataFile, "rb") as myFile:
    songIndices = pickle.load(myFile)

dataFile = fileLocation + 'listenLengths.txt'
with open(dataFile, "rb") as myFile:
    listenLengths = pickle.load(myFile)

# to hold cumulative couples and labels while looping through users
couplesList = []
labelsList = []

userStartSongIndex = 0
for user in range(listenLengths.shape[0]):
    numPlayEvents = listenLengths[user, 1]
    userSongIndices = songIndices[userStartSongIndex:userStartSongIndex + numPlayEvents]
    userCouples = []
    userLabels = []
    # lower boundary, context less than target
    for target in range(1, windowSize + 1):
        negSampList = list(range(target + windowSize + 1, numPlayEvents + 1))
        for diff in range(1, windowSize + 1):
            if target - diff > 0:
                userCouples.extend([[userSongIndices[target - diff - 1], userSongIndices[target - 1]]])
                userLabels.extend([1])
                for i in range(negativeSamples):
                    userCouples.extend([[userSongIndices[random.choice(negSampList) - 1], userSongIndices[target - 1]]])
                    userLabels.extend([0])
    print("Finished lower boundary, context less than target:", time.time() - start)

    # mid range, , context less than target
    for target in range(windowSize + 1, numPlayEvents + 1):
        negSampList = list(range(1, target - windowSize))
        negSampList.extend(list(range(target + windowSize + 1, numPlayEvents + 1)))
        for diff in range(1, windowSize + 1):
            userCouples.extend([[userSongIndices[target - diff - 1], userSongIndices[target - 1]]])
            userLabels.extend([1])
            for i in range(negativeSamples):
                userCouples.extend([[userSongIndices[random.choice(negSampList) - 1], userSongIndices[target - 1]]])
                userLabels.extend([0])
    print("Finished mid range, , context less than target:", time.time() - start)

    # upper boundary, context more than target
    for target in range(numPlayEvents - windowSize, numPlayEvents + 1):
        negSampList = list(range(1, target - windowSize))
        for diff in range(1, windowSize + 1):
            if target + diff < numPlayEvents + 1:
                userCouples.extend([[userSongIndices[target + diff - 1], userSongIndices[target - 1]]])
                userLabels.extend([1])
                for i in range(negativeSamples):
                    userCouples.extend([[userSongIndices[random.choice(negSampList) - 1], userSongIndices[target - 1]]])
                    userLabels.extend([0])
    print("Finished upper boundary, context more than target:", time.time() - start)

    # mid range, context more than target
    for target in range(1, numPlayEvents - windowSize + 1):
        negSampList = list(range(1, target - windowSize))
        negSampList.extend(list(range(target + windowSize + 1, numPlayEvents + 1)))
        for diff in range(1, windowSize + 1):
            userCouples.extend([[userSongIndices[target + diff - 1], userSongIndices[target - 1]]])
            userLabels.extend([1])
            for i in range(negativeSamples):
                userCouples.extend([[userSongIndices[random.choice(negSampList) - 1], userSongIndices[target - 1]]])
                userLabels.extend([0])
    # add this users couples and labels to the cumulative list
    couplesList.extend(userCouples)
    labelsList.extend(userLabels)
    print("Finished user:", user)

# Shuffle couples and labels because currently sequential
lenCou = len(couplesList)
newOrder = random.sample(range(lenCou), lenCou)
couplesShuffled = list(range(lenCou))
labelsShuffled = list(range(lenCou))
for i in range(lenCou):
    couplesShuffled[newOrder[i]] = couplesList[i]
    labelsShuffled[newOrder[i]] = labelsList[i]
couplesArray = np.asarray(couplesShuffled)
labelsArray = np.asarray(labelsShuffled)

dataFile = fileLocation + 'CBOWcouplesW' + str(windowSize) + "NS" + str(negativeSamples) + ".txt"
with open(dataFile, "wb") as myFile:
    pickle.dump(couplesArray, myFile)
dataFile = fileLocation + 'CBOWlabelsW' + str(windowSize) + "NS" + str(negativeSamples) + ".txt"
with open(dataFile, "wb") as myFile:
    pickle.dump(labelsArray, myFile)

print("Total CBOW runtime", time.time() - start)
