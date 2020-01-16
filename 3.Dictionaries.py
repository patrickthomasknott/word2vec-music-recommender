###########################
###########################
# User Inputs
fileLocation = '/'  # location of input and output files
###########################
###########################
###########################
from numpy.random import seed
import pickle
import numpy as np
import pandas as pd
import collections
import time
from tensorflow import set_random_seed

'''
This script converts the ordered lists of songs for each user into numeric encoding. It creates dictionaries for song
name to song index, and the reverse. It converts the full set of user song histories into a vector of song indices, 
with listenLengths containing the first and last song for each user.

Data inputs
•	UserSonglist.csv
Data outputs
•	dictionary.txt
•	reverseDictionary.txt
•	songIndices.txt
•	listenLengths.txt
Printed outputs:
•	timings

'''

seed(1)
set_random_seed(2)
start = time.time()
dataFile = fileLocation + 'UserSonglist.csv'
df = pd.read_csv(dataFile, dtype='object', sep=',')

# dataframe of first and last song indices for each users, because songIndices is just a list
listenLengths = np.empty(shape=(df.shape[0], 2),
                         dtype=int)  # number of songs listened to per user
for i in range(df.shape[0]):
    PlayEvents = len(df.iloc[i, :].dropna())
    listenLengths[i, 0] = i
    listenLengths[i, 1] = PlayEvents
    print("Row:", i, \
          "- #PlayEvents:", PlayEvents, \
          "- Average plays:", round(len(df.iloc[i, :].dropna()) / len(df.iloc[i, :].dropna().unique()), 1))

songs = pd.Series(df.iloc[0][:]).dropna()
for i in range(1, df.shape[0]):
    songs = songs.append(pd.Series(df.iloc[i][:]).dropna(), ignore_index=True)

count = [['UNK', -1]]  # UNK is unknown, required for SkipGram function
count.extend(collections.Counter(songs).most_common(len(songs) - 1))  # NB len(songs) >> len(count)

vocabSize = len(count)
start = time.time()
dictionary = dict()
for song, _ in count:
    dictionary[song] = len(dictionary)
songIndices = list()
unkCount = 0
for song in songs:
    if song in dictionary:
        index = dictionary[song]
    else:  # UNK is unknown, for later when I drop rare songs
        index = 0  # dictionary['UNK']
        unkCount += 1
    songIndices.append(index)
count[0][1] = unkCount
reverseDictionary = dict(zip(dictionary.values(), dictionary.keys()))

dataFile = fileLocation + 'dictionary.txt'
with open(dataFile, "wb") as myFile:
    pickle.dump(dictionary, myFile)

dataFile = fileLocation + 'reverseDictionary.txt'
with open(dataFile, "wb") as myFile:
    pickle.dump(reverseDictionary, myFile)

dataFile = fileLocation + 'songIndices.txt'
with open(dataFile, "wb") as myFile:
    pickle.dump(songIndices, myFile)

dataFile = fileLocation + 'listenLengths.txt'
with open(dataFile, "wb") as myFile:
    pickle.dump(listenLengths, myFile)

print("Runtime:", time.time() - start)
