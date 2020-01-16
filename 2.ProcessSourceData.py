###########################
###########################
# User Inputs
fileLocation = '/'  # location of LastFM tsv file
fileName = 'LastFM.tsv'  # input file
users2keep = [1, 11, 20, 21, 32, 66, 67, 73]  # which users to keep from LastFM tsv file
###########################
###########################
###########################
'''
This script is to process the raw data into the format required for the word2vec dictionary creation, and for the 
collaborative filtering mean-centred npseudo-rating matrix.

Data inputs
•	LastFM.tsv
Data outputs
•	cfMX.csv
•	dfMaxMean.txt
•	UserSonglist.csv
•	UserSessionSonglist.csv
Printed outputs:
•	tally of play-events and average listen count per user
•	timings of various computations

'''
import pandas as pd
import time
from datetime import datetime
import numpy as np
import pickle
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# read original data file
totalStart = time.time()
start = time.time()

dataFile = fileLocation + fileName
df = pd.read_csv(dataFile, sep='\t', error_bad_lines=False, header=None,
                 usecols=[0, 1, 3, 5])
df.columns = ['User', 'Timestamp', 'ArtName', 'TraName']
end = time.time()
print("Read tsv file:", round(end - start))

# drop columns
df["Song"] = df["ArtName"] + ": " + df["TraName"]
df.drop(['ArtName', 'TraName'], axis=1, inplace=True)

# mask to select which user's to keep
userMask = '('
for i in users2keep:
    userMask = userMask + "df.User == '" + df.User.unique()[i] + "') | ("

# reorder songs chronologically, they're backward in input file
df.sort_values(['User', 'Timestamp'], ascending=[True, True], inplace=True)
df.reset_index(inplace=True, drop=True)

# convert timestamp to seconds since epoch
start = time.time()
df['Timestamp'] = df.loc[:, 'Timestamp'].apply(lambda x: datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ').timestamp())
end = time.time()
print("Convert timestamp to seconds-from-epoch:", round(end - start))

# Make prod2vec dataframe
start = time.time()
UserSonglist = df.groupby('User', as_index=True)['Song'].apply(list)  # decompose into user's entire songlist
UserSonglist = UserSonglist.apply(lambda x: pd.Series(x))  # turn list of lists into a dataframe
end = time.time()
print("Convert data to prod2vec input:", round(end - start))

# keep these users
UserSonglist = UserSonglist.iloc[users2keep, :]
# count playlist events remaining
total = 0
for i in range(UserSonglist.shape[0]):
    total += len(UserSonglist.iloc[i, :].dropna())

# calculate average play-event of each listener
totalSongs = 0
uniqueSongs = 0
for i in range(1, 9):
    totalSongs = len(UserSonglist.iloc[i - 1, :].dropna())
    uniqueSongs = UserSonglist.iloc[i - 1, :].dropna().nunique()
    print("User:", i,
          "Play-events:", totalSongs,
          "Average:", round(totalSongs / uniqueSongs, 1))

# # save user songlist
start = time.time()
dataFile = fileLocation + 'UserSonglist.csv'
UserSonglist.to_csv(dataFile, encoding='utf-8', index=False)
end = time.time()
print("UserSonglist:", round(end - start))

# Make pseudo-rating matrix for collaborative filtering  NB actually a dataframe
# mask of rows for the user's selected above
start = time.time()
dfCFMX = df.loc[eval(userMask[:-4]), ['User', 'Song']]

dfCFMX = dfCFMX.groupby('User')['Song'].value_counts().unstack().fillna(0)
dfCFMX.reset_index(drop=True, inplace=True)

dfMaxMean = np.zeros([dfCFMX.shape[0], 2])  # to hold the max and average scaled-listen-count per user
for i in range(dfCFMX.shape[0]):  # calculate max listen count per user
    dfMaxMean[i][0] = max(dfCFMX.iloc[i, :])

for i in range(dfCFMX.shape[0]):
    for j in range(dfCFMX.shape[1]):
        if dfCFMX.iloc[i, j] != 0:
            dfCFMX.iloc[i, j] = dfCFMX.iloc[i, j] / dfMaxMean[i][0]  # divide each listen count by max for the user

# calc mean of the scaled ratings for each user
for i in range(dfCFMX.shape[0]):
    dfMaxMean[i][1] = (dfCFMX.iloc[i, :]).mean()

# subtract each user's mean rating from each of their rating
for i in range(dfCFMX.shape[0]):
    for j in range(dfCFMX.shape[1]):
        if dfCFMX.iloc[i, j] != 0:
            dfCFMX.iloc[i, j] = dfCFMX.iloc[i, j] - dfMaxMean[i][1]  # mean centered

# save dataframe of user mean and max values for use with predictions
dataFile = fileLocation + 'dfMaxMean.txt'
with open(dataFile, "wb") as myFile:
    pickle.dump(dfMaxMean, myFile)

# save CF matrix to a csv file
dataFile = fileLocation + 'cfMX.csv'
dfCFMX.to_csv(dataFile, encoding='utf-8', index=False)
end = time.time()
print("Create UserRating.csv:", round(end - start))

end = time.time()
print("Total runtime:", round(end - totalStart))

# Make user listen-session input
df = df.loc[eval(userMask[:-4]), ['User', 'Song', 'Timestamp']]
df = df.dropna()

# boolean, is this song started within 15 minutes of previous song
df['ConsecutiveSongs'] = df['Timestamp'].diff().abs() < 600
df.drop(['Timestamp'], axis=1, inplace=True)
session = 0
tempList = [df.iloc[0, 1]]
listPlaylist = []
start = time.time()
for inputRow in range(1, df.shape[0]):  # for each row in full dataset
    if df['User'].iloc[inputRow] == df['User'].iloc[inputRow - 1]:  # if current and previous row are same user
        if df['ConsecutiveSongs'].iloc[inputRow]:  # if <15 minutes between song starts
            tempList.append(df.iloc[inputRow, 1])  # add song to current 'session'
        else:
            listPlaylist.append(tempList)  # append current tempList to dfPlaylist, then start a new tempList
            tempList = [df.iloc[inputRow, 1]]
    else:  # current row and previous row are different users, ...
        # append current tempList to dfPlaylist, then start a new tempList
        listPlaylist.append(tempList)
        tempList = []
if len(tempList) > 0:  # if there are songs still to add to current session, then do so
    listPlaylist.append(tempList)
end = time.time()
print("Convert data to bagged-prod2vec input:", round(end - start))
UserSessionSonglist = pd.Series(listPlaylist)

UserSessionSonglist = UserSessionSonglist.apply(lambda x: pd.Series(x)).T
dataFile = fileLocation + 'UserSessionSonglist.csv'
UserSessionSonglist.to_csv(dataFile, encoding='utf-8', index=False)
end = time.time()
print("UserSessionSonglist:", round(end - start))
