###########################
###########################
# User Inputs
fileLocation = '/'  # location of input and output files
windowSize = 3  # target word +/-
negativeSamples = 10  # negative samples per positive sample
neurons = 40  # list of values to test
batchSize = 500  # list of values to test
epochs = 2
numSongs = 20  # number of closest songs to print out
###########################
###########################
###########################
from numpy.random import seed
import operator
import numpy as np
import pickle
from keras.models import load_model
import pandas as pd
import time
from tensorflow import set_random_seed
import collections
import warnings

warnings.filterwarnings("ignore", 'This pattern has match groups')
'''
This script represents each user in the word-embedding space by averaging the song vectors of the user's play-event
history. This is done separately for the SG and CBOW models.
 
Data inputs
•	UserSonglist.csv
•	CBOWmodel... .h5
•	dictionary.txt
•	reverseDictionary.txt
Data outputs
•	CBOWsongssortedUser... .txt (for each user)
•	CBOWsong2similarityDictionaryUser... .txt (for each user)
•	CBOWsongssortedUser... .txt (for each user)
•	CBOW User toVec ... .txt  (for each user)
Printed outputs:
•	closest songs to user
•	cosine similarity of user's most listened to songs
'''

modelName = 'CBOWmodelW' + str(windowSize) + "NS" + str(negativeSamples) + 'Neurons' + \
            str(neurons) + 'Batchsize' + str(batchSize) + 'Epochs' + str(epochs)

seed(1)
set_random_seed(2)
totalStart = time.time()
start = time.time()

dataFile = fileLocation + "UserSonglist.csv"
df = pd.read_csv(dataFile, dtype='object', sep=',')

dataFile = fileLocation + modelName + ".h5"
model = load_model(dataFile)

dataFile = fileLocation + "reverseDictionary.txt"
with open(dataFile, "rb") as myFile:
    reverseDictionary = pickle.load(myFile)

dataFile = fileLocation + "dictionary.txt"
with open(dataFile, "rb") as myFile:
    dictionary = pickle.load(myFile)

vocabSize = len(reverseDictionary)

for user in range(df.shape[0]):
    start = time.time()
    tempUserVec = np.zeros(len(model.get_weights()[0][0]))  # number of neurons in model
    songs = df.iloc[user, :].dropna()
    count = collections.Counter(songs).most_common(len(songs) - 1)  # returns [ [song-name],[song listen count] ]
    denominator = 0
    for i in range(len(count)):
        tempSongIndex = dictionary[count[i][0]]
        tempUserVec = tempUserVec + count[i][1] * model.get_weights()[0][tempSongIndex]
        denominator += count[i][1]

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("User", str(user), " user2vec consists of ", i, "songs.")

    tempUserVec = tempUserVec / denominator
    print(tempUserVec.mean())
    songSimDic = {}
    for i in range(vocabSize):
        targetSongWeights = model.get_weights()[0][i]
        thetaSum = np.dot(tempUserVec, targetSongWeights)
        thetaDen = np.linalg.norm(tempUserVec) * np.linalg.norm(targetSongWeights)
        theta = thetaSum / thetaDen
        song = reverseDictionary[i]
        songSimDic[song] = abs(theta)

    songsSorted = sorted(songSimDic.items(), key=operator.itemgetter(1), reverse=True)
    print("\nClosest song runtime:", time.time() - start)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Songs closest to user", str(user), ":")
    notInHistory = 0
    for song, sim in songsSorted[:numSongs]:
        print("\n", song, round(sim, 4))
        if (df.iloc[user, :].str.replace(')', '').
                replace('()', '').str.contains(
            song.replace(')', '').replace('(', '').replace("\\", '').replace('[', '').replace(']', ''))).sum() > 0:
            print("\tSimilarity:", round(sim, 4))
        else:
            print("NOT in user's history")
            notInHistory += 1
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(notInHistory, " of user", str(user) + "'s top 20 recommended songs are not in their history.")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # what are the similarities of the user's 20 most frequently listened to songs
    print("\n\nSimilarities of the user", str(user) + "'s 20 most frequently listened to songs:")
    avgTheta = 0
    for i in range(0, 20):
        theta = songSimDic[count[i][0]]
        print("#" + str(i) + ":", round(theta, 5), count[i][0])
        avgTheta += theta
    print("Average similarity of user", str(user) + "'s 20 songs:", avgTheta / 20)

    #     Save sorted songs, song-to-similarity dictionary, and user vector
    saveName = fileLocation + 'CBOWsongssortedUser ' + str(user) + " " + modelName + '.txt'
    with open(saveName, "wb") as myFile:
        pickle.dump(songsSorted, myFile)

    saveName = fileLocation + 'CBOWsong2similarityDictionaryUser ' + str(user) + " " + modelName + '.txt'
    with open(saveName, "wb") as myFile:
        pickle.dump(songSimDic, myFile)

    saveName = fileLocation + 'CBOW User' + str(user) + 'toVec ' + modelName + '.txt'
    with open(saveName, "wb") as myFile:
        pickle.dump(tempUserVec, myFile)

print("Total run time:", time.time() - totalStart)
