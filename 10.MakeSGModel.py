###########################
###########################
# User Inputs
fileLocation = '/'  # location of input and output files
windowSize = 3  # target word +/-
negativeSamples = 10  # negative samples per positive sample
neurons = 40  # list of values to test
batchSize = 1000  # list of values to test
epochs = 5
loss = 'mean_squared_error'
optimizer = 'Adam'
activation = 'relu'
validation_split = 0.2
###########################
###########################
###########################
'''
This script builds a SkipGram model with the best parameters chosen from the gridsearch.

Data inputs
•	SGcouples... .txt
•	SGlabels... .txt
•	reverseDictionary.txt
Data outputs
•	Accuracy... .png 
•	Loss... .png
•	SGmodel... .h5
Printed outputs:
•	progress during training
•	figure: Accuracy
•	figure: Loss
'''
from numpy.random import seed
from tensorflow import set_random_seed
import pickle
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import time

set_random_seed(2)
totalStart = time.time()
seed(1)

modelName = 'SGmodelW' + str(windowSize) + "NS" + str(negativeSamples) + 'Neurons' + \
            str(neurons) + 'Batchsize' + str(batchSize) + 'Epochs' + str(epochs)

dataFile = fileLocation + 'SGcouplesW' + str(windowSize) + "NS" + str(negativeSamples) + ".txt"
with open(dataFile, "rb") as myFile:
    couples = pickle.load(myFile)

dataFile = fileLocation + 'SGlabelsW' + str(windowSize) + "NS" + str(negativeSamples) + ".txt"
with open(dataFile, "rb") as myFile:
    labels = pickle.load(myFile)

dataFile = fileLocation + "reverseDictionary.txt"
with open(dataFile, "rb") as myFile:
    reverseDictionary = pickle.load(myFile)

vocabSize = len(reverseDictionary)

model = Sequential()
model.add(Embedding(input_dim=vocabSize, output_dim=neurons, input_length=2, embeddings_initializer='RandomNormal'))
model.add(Flatten())
model.add(Dense(1, activation='relu'))
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])


class EpochIdentification(Callback):
    def onEpochEnd(self, epoch, logs=None):
        print("Epoch index just finished:", epoch, "out of", self.model.history.params['epochs'], "(exclusive)")
        print("Optimizer:", self.model.optimizer.__class__.__name__)
        print("Activation:", self.model.get_config()['layers'][2]['config']['activation'])
        print("Neurons:", self.model.get_config()['layers'][0]['config']['output_dim'])
        print("Batch size:", self.model.history.params['batch_size'])
        print("Model loss function:", self.model.loss)
        print("Time since model fit start:", time.time() - start)
        print("\n\n")


eid = EpochIdentification()
start = time.time()
model.fit(couples, labels, batch_size=batchSize, verbose=1, epochs=epochs, validation_split=validation_split,
          callbacks=[eid])
end = time.time()
print("Model fit time:", round(end - start))

plt.figure(0)
plt.plot(model.history.history['acc'])
plt.plot(model.history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
saveName = fileLocation + 'Accuracy ' + modelName + '.png'
plt.savefig(saveName)
plt.show()
plt.figure(1)
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
saveName = fileLocation + 'Loss ' + modelName + '.png'
plt.savefig(saveName)
plt.show()

# save model, validation_model
saveName = fileLocation + modelName + '.h5'
model.save(saveName)  # creates a HDF5 file
time.time() - start
