###########################
###########################
# User Inputs
fileLocation = '/'  # location of input and output files
windowSize = 3  # target word +/-
negativeSamples = 10  # negative samples per positive sample
optimizer = 'Adam'
activation = 'relu'
neurons = [10, 40]
loss = 'mean_squared_error'
batch_size = [500, 1000, 50000]
epochs = 6
validation_split = 0.2
###########################
###########################
###########################
'''
This script build multiple models with different parameter combinations, and ouputs figures showing model 
testing-accuracy and model testing-loss during from each epoch of training. NB this model uses sklearn's GridSearchCV
function, where as the CBOW model uses nested for loops.

Data inputs
•	SGcouples... .txt
•	SGlabels... .txt
•	reverseDictionary.txt
Data outputs
•	SGmodel.... .h5
•	Accuracy... .png (for each model parameter combination)
•	Loss... .png (for each model parameter combination)
'''
from numpy.random import seed

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)
import pickle
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import Callback
from sklearn.model_selection import GridSearchCV
from pandas import DataFrame
import time

dataFile = fileLocation + 'CBOWcouplesW' + str(windowSize) + "NS" + str(negativeSamples) + ".txt"
with open(dataFile, "rb") as myFile:
    couples = pickle.load(myFile)

dataFile = fileLocation + 'CBOWlabelsW' + str(windowSize) + "NS" + str(negativeSamples) + ".txt"
with open(dataFile, "rb") as myFile:
    labels = pickle.load(myFile)

dataFile = fileLocation + "reverseDictionary.txt"
with open(dataFile, "rb") as myFile:
    reverseDictionary = pickle.load(myFile)

vocab_size = len(reverseDictionary)


def createModel(optimizer='Adam', activation='sigmoid', neurons=40, loss='mean_absolute_error',
                embeddings_initializer='RandomNormal'):
    # NB have to supply default values in method definition, replace with param_grid when called
    modelInternal = Sequential()
    modelInternal.add(Embedding(vocab_size, neurons, input_length=2, embeddings_initializer=embeddings_initializer))
    modelInternal.add(Flatten())
    modelInternal.add(Dense(1, activation=activation))
    modelInternal.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return modelInternal


class EpochIdentification(Callback):
    def onEpochEnd(self, epoch, logs=None):
        print("Epoch index just finished:", epoch, "out of", self.model.history.params['epochs'], "(exclusive)")
        print("Optimizer:", self.model.optimizer.__class__.__name__)
        print("Activation:", self.model.get_config()['layers'][2]['config']['activation'])
        print("Neurons:", self.model.get_config()['layers'][0]['config']['output_dim'])
        print("Batch size:", self.model.history.params['batch_size'])
        print("Model loss function:", self.model.loss)
        print("\n\n")


eid = EpochIdentification()
model = KerasClassifier(build_fn=createModel, verbose=1)
param_grid = dict(neurons=neurons, batch_size=batch_size)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
start = time.time()
grid_result = grid.fit(couples, labels, callbacks=[eid], validation_split=0.1)
print("Gridsearch runtime:", time.time() - start)

# Save summary
tempDF = DataFrame(grid_result.cv_results_)
dataFile = fileLocation + 'SG_GridSearchResults.csv'
tempDF.to_csv(dataFile, encoding='utf-8', index=False)
