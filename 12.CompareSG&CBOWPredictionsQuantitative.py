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
###########################
###########################
import pickle

'''
This script finds the top 20 songs for a user in one of the two models (SkipGram or CBOW), then checks how similar
they are to the user's vector in the other model's word embedding vector space.

Data inputs
•	SGmodel... .txt
•	SGsongssortedUser... .txt (for each user)
•	SGsong2similarityDictionaryUser... .txt (for each user)
•	CBOWmodel... .txt
•	CBOWsongssortedUser... .txt (for each user)
•	CBOWsong2similarityDictionaryUser... .txt (for each user)
Data outputs
•	none
Printed outputs:
•	SkipGram cosine similarity of the twenty songs closest to user in CBOW vector space
•	CBOW cosine similarity of the twenty songs closest to user in SkipGram vector space

'''
for user in range(8):
    SGmodelName = 'SGmodelW' + str(windowSize) + "NS" + str(negativeSamples) + 'Neurons' + \
                  str(neurons) + 'Batchsize' + str(sgBatchSize) + 'Epochs' + str(SGepochs)

    CBOWmodelName = 'CBOWmodelW' + str(windowSize) + "NS" + str(negativeSamples) + 'Neurons' + \
                    str(neurons) + 'Batchsize' + str(cbowBatchSize) + 'Epochs' + str(CBOWepochs)

    sgLoadName = fileLocation + 'SGsongssortedUser ' + str(user) + " " + SGmodelName + '.txt'
    with open(sgLoadName, "rb") as myFile:
        sg = pickle.load(myFile)

    cbowLoadName = fileLocation + 'CBOWsongssortedUser ' + str(user) + " " + CBOWmodelName + '.txt'
    with open(cbowLoadName, "rb") as myFile:
        cbow = pickle.load(myFile)

    sgLoadName = fileLocation + 'SGsong2similarityDictionaryUser ' + str(user) + " " + SGmodelName + '.txt'
    with open(sgLoadName, "rb") as myFile:
        sgDic = pickle.load(myFile)

    cbowLoadName = fileLocation + 'CBOWsong2similarityDictionaryUser ' + str(user) + " " + SGmodelName + '.txt'
    with open(sgLoadName, "rb") as myFile:
        cbowDic = pickle.load(myFile)

    print('CBOW model similarity score of top 20 SG songs:')
    total = 0
    for i in range(20):
        total += cbowDic[sg[i][0]]
        print(cbowDic[sg[i][0]])  # CBOW relevance of top 20 SG songs
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Average relevance:", total / 20)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    print('\n\nSG model similarity score of top 20 CBOW songs:')
    total = 0
    for i in range(20):
        total += sgDic[cbow[i][0]]
        print(sgDic[cbow[i][0]])  # SG relevance of top 20 CBOW songs
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Average relevance:", total / 20)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
