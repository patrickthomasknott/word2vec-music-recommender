###########################
###########################
# User Inputs
fileLocation = '/701/'  # location of input and output files
windowSize = 3  # target word +/-
negativeSamples = 10  # negative samples per positive sample
neurons = 40  # list of values to test
batchSize = 1000  # list of values to test
epochs = 5
minNumListeners = 5
###########################
###########################
'''
This script calculates predictions for each user's mean-centred pseudo-rating for each song using a collaborative 
filtering technique. The predictions are compared to the actual pseudo-ratings calculated in 2.ProcessSourceData.py. 
It allows the user to input the minumum number of listeners for a song to be included in the analysis

Data inputs
•	dfMX.csv
•	dfMaxMean.txt
•	SGUser toVec... .txt
Data outputs
•	Actual vs Predicted Pseudo-ratings user... .png (for each user)
Printed outputs:
•	number of distinct songs per user
•	number of songs listened to by more than one person (showing count for each of 2, 3, 4...)
•	figure: Actual vs Predicted Pseudo-ratings user...  (for each user)
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import Counter

dataFile = fileLocation + 'cfMX.csv'
df = pd.read_csv(dataFile)

numUsers = df.shape[0]

dataFile = fileLocation + 'dfMaxMean.txt'
with open(dataFile, "rb") as myFile:
    dfMaxMean = pickle.load(myFile)

# load user2vec vectors
user = []  # list of user2vec vectors
for i in range(numUsers):
    modelName = 'SGmodelW' + str(windowSize) + "NS" + str(negativeSamples) + 'Neurons' + \
                str(neurons) + 'Batchsize' + str(batchSize) + 'Epochs' + str(epochs)
    loadName = fileLocation + 'SG User' + str(i) + 'toVec ' + modelName + '.txt'

    with open(loadName, "rb") as myFile:
        user2vec = pickle.load(myFile)
    user.append(user2vec)

# make lookup table of user2vec cosine similarities
lookup = pd.DataFrame(data=None, dtype='float', index=range(numUsers), columns=range(numUsers))
for i in range(numUsers):
    for j in range(numUsers):
        thetaSum = np.dot(user[i], user[j])
        thetaDen = np.linalg.norm(user[i]) * np.linalg.norm(user[j])
        lookup.iloc[i, j] = np.abs(thetaSum / thetaDen)

# count how many distinct songs are in each user's history that at least one other person also listened to
userSongCount2plus = []
for i in range(numUsers):
    tally = 0
    for j in range(df.shape[1]):
        if df.iloc[i, j] != 0:
            tally += 1
    userSongCount2plus.append(tally)
print("Distinct songs listened to by each user with at least one other person :",
      userSongCount2plus)

# count number of songs listened to by more than one person
listenerCountList2plus = []
for j in range(df.shape[1]):
    numListeners = (df.iloc[:, j] != 0).sum()
    if numListeners > 1:
        listenerCountList2plus.append(numListeners)
for i in range(min(Counter(listenerCountList2plus)), max(Counter(listenerCountList2plus)) + 1):
    print("There are", Counter(listenerCountList2plus)[i], "songs listened to by",
          i, "people.")

# find indices of songs with minNumListeners or more listeners
mask = []
for j in range(df.shape[1]):
    numListeners = (df.iloc[:, j] != 0).sum()
    if numListeners > minNumListeners - 1:
        mask.append(j)

# only keep songs with minNumListeners or more listeners
df = df.iloc[:, mask]
print("Number of distinct songs left in dataframe:", df.shape[1])

# predict unseen ratings
for user in range(numUsers):  # user is target user aka 'u' in collaborative filtering formula
    predVsActualRating = pd.DataFrame(index=["ActualRating", "PredictedRating"])
    for j in range(df.shape[1]):  # j is song column
        if df.iloc[user, j] != 0:
            predScore = 0
            denom = 0
            for i in range(numUsers):  # i is other user aka 'v' in collaborative filtering formula
                if i == user:
                    continue
                if df.iloc[i, j] != 0:
                    denom += np.abs(lookup.iloc[user, i])  # add absolute value of similarity to denominator
                    predScore += lookup.iloc[user, i] * df.iloc[i, j]  # add other rating * similarity to numerator
            tempDf = pd.DataFrame(data=None, index=["ActualRating", "PredictedRating"], columns=[df.columns[j]])
            tempDf.iloc[0, 0] = round(df.iloc[user, j], 4)
            # divide sum of similarity * other user's rating by sum of absolute similarities, ...
            # then add target user's mean rating
            tempDf.iloc[1, 0] = round(predScore / denom + dfMaxMean[user][1], 4)
            predVsActualRating = pd.concat([predVsActualRating, tempDf], axis=1)

    plt.figure(figsize=(11, 3))
    plt.plot(predVsActualRating.iloc[0, :], 'r', label="Actual", marker='o')
    plt.plot(predVsActualRating.iloc[1, :], 'g', label="Predicted", marker='o')
    plt.title("User " + str(user) + " Actual Ratings vs. Collaborative Filtering Predictions")
    plt.legend(loc=2)
    x = range(predVsActualRating.shape[1])
    plt.xticks(x, " ")
    saveName = fileLocation + 'Actual vs Predicted Pseudo-ratings user' + str(user) + '.png'
    plt.savefig(saveName)
    plt.show()
