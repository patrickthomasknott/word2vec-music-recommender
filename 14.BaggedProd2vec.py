###########################
###########################
# User Inputs
fileLocation = '/'  # location of input and output files
windowSize = 3  # target word +/-
negativeSamples = 10  # negative samples per positive sample
neurons = 40
batchSize = 10000
epochs = 5
numSongs = 20  # number of closest songs to print out
loss = 'mean_squared_error'
optimizer = 'Adam'
activation = 'relu'
validationSplit = 0.1
###########################
###########################
###########################
'''
This script is for assessing the bagged-prod2vec model. It makes the input pairs, builds a model, builds user2vec for 
each user, finds the closest songs to each user, calculates the cosing proximity of each user to their twenty 
closest songs.

Data inputs
•	UserSessionSonglist.csv
•	listenLengths.txt
Data outputs
•	Accuracy... .png
•	Loss... .png
•	BaggedSGmodel... .h5
•	BaggedSGsongssortedUser... .txt
•	BaggedSGsong2similarityDictionaryUser... .txt
•	BaggedSGUser toVec... .txt
Printed outputs:
•	timings
•	progress during training
•	closest song to each user
•	cosine proximity of user's most listened to songs
'''
from numpy.random import seed
import pickle
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import skipgrams
import time
from tensorflow import set_random_seed
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.callbacks import Callback
import operator
import collections
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", 'This pattern has match groups')
start = time.time()

seed(1)
set_random_seed(2)
dataFile = fileLocation + 'UserSessionSonglist.csv'
df = pd.read_csv(dataFile, dtype='object', sep=',')
df = np.transpose(df)

dataFile = fileLocation + 'listenLengths.txt'
with open(dataFile, "rb") as myFile:
    listenLengths = pickle.load(myFile)

songs = pd.Series(df.iloc[0][:]).dropna()
for i in range(1, df.shape[0]):
    songs = songs.append(pd.Series(df.iloc[i][:]).dropna(), ignore_index=True)

count = [['UNK', -1]]  # UNK is unknown, required for SkipGram function
count.extend(collections.Counter(songs).most_common(len(songs) - 1))  # NB len(songs) >> len(count)

vocabSize = len(count)
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

set_random_seed(2)
seed(1)

couples = list()
labels = list()
vocabSize = max(songIndices) + 1
firstIndex = 0
lastIndex = 0
for i in range(listenLengths.shape[0]):
    lastIndex += listenLengths[i, 1]
    tempCouples, tempLabels = skipgrams(songIndices[firstIndex:lastIndex], vocabSize, \
                                        window_size=windowSize, sampling_table=None, \
                                        negative_samples=negativeSamples, shuffle=True, seed=0)

    couples.extend(tempCouples)
    labels.extend(tempLabels)
    firstIndex = lastIndex
    print("finished skipgrams listen session", i, "of", listenLengths.shape[0])

couples = np.asarray(couples)
labels = np.asarray(labels)


modelName = 'BaggedSGmodelW' + str(windowSize) + "NS" + str(negativeSamples) + 'Neurons' + \
            str(neurons) + 'Batchsize' + str(batchSize) + 'Epochs' + str(epochs)

vocabSize = len(reverseDictionary)

model = Sequential()
model.add(Embedding(input_dim=vocabSize, output_dim=neurons, input_length=2, embeddings_initializer='RandomNormal'))
model.add(Flatten())
model.add(Dense(1, activation=activation))
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
modelStart = time.time()


class EpochIdentification(Callback):
    def onEpochEnd(self, epoch, logs=None):
        print("Epoch index just finished:", epoch, "out of", self.model.history.params['epochs'], "(exclusive)")
        print("Optimizer:", self.model.optimizer.__class__.__name__)
        print("Activation:", self.model.get_config()['layers'][2]['config']['activation'])
        print("Neurons:", self.model.get_config()['layers'][0]['config']['output_dim'])
        print("Batch size:", self.model.history.params['batch_size'])
        print("Model loss function:", self.model.loss)
        print("Time since model fit start:", time.time() - modelStart)
        print("\n\n")


class LossHistory(Callback):
    def onTrainBegin(self, logs={}):
        self.losses = []

    def onBatchEnd(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


lh = LossHistory()
eid = EpochIdentification()

model.fit(couples, labels, batch_size=batchSize, verbose=1, epochs=epochs,
          validation_split=validationSplit, callbacks=[eid, lh])

plt.figure(0)
plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
saveNameAccFigure = fileLocation + 'Accuracy ' + modelName + '.png'
plt.savefig(saveNameAccFigure)
# plt.show()
plt.figure(1)
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
saveNameLossFigure = fileLocation + 'Loss ' + modelName + '.png'
plt.savefig(saveNameLossFigure)
# plt.show()

saveNameModel = fileLocation + modelName + '.h5'
model.save(saveNameModel)  # creates a HDF5 file

dataFile = fileLocation + 'UserSonglist.csv'
df = pd.read_csv(dataFile, dtype='object', sep=',')
numSongs = 20
vocabSize = len(reverseDictionary)

for user in range(df.shape[0]):
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
    print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
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
    print("Average similarity of user", str(user) + "'s 20 most frequently listened to songs:", avgTheta / 20)

    #     Save sorted songs, song-to-similarity dictionary, and user vector
    #     Save sorted songs, song-to-similarity dictionary, and user vector
    saveName = fileLocation + 'BaggedSGsongssortedUser ' + str(user) + " " + modelName + '.txt'
    with open(saveName, "wb") as myFile:
        pickle.dump(songsSorted, myFile)

    saveName = fileLocation + 'BaggedSGsong2similarityDictionaryUser ' + str(user) + " " + modelName + '.txt'
    with open(saveName, "wb") as myFile:
        pickle.dump(songSimDic, myFile)

    saveName = fileLocation + 'BaggedSGUser' + str(user) + 'toVec ' + modelName + '.txt'
    with open(saveName, "wb") as myFile:
        pickle.dump(tempUserVec, myFile)
print("Total run time:", time.time() - start)
