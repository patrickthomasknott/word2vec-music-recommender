###########################
###########################
# User Inputs
fileLocation = '/'  # location of raw data file
fileName = 'LastFM.tsv'  # input file name
###########################
###########################
###########################
'''
This script is to inspect the user listening history data. It prints out the number of play events and average
 play-events per song, per user
 
Data inputs
•	LastFM tsv file
Data outputs
•	none
Printed outputs:
•	number of play events and average play events per song per user
'''
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# read original data file
dataFile = fileLocation + fileName
df = pd.read_csv(dataFile, sep='\t', error_bad_lines=False, header=None,
                 usecols=[0, 1, 3, 5])
df.columns = ['User', 'Timestamp', 'ArtName', 'TraName']

# drop columns
df["Song"] = df["ArtName"] + ": " + df["TraName"]
df.drop(['ArtName', 'TraName'], axis=1, inplace=True)

# reorder songs chronologically, they're backward in input file
df.sort_values(['User', 'Timestamp'], ascending=[True, True], inplace=True)
df.reset_index(inplace=True, drop=True)

# Make prod2vec dataframe
UserSonglist = df.groupby('User', as_index=True)['Song'].apply(list)  # decompose into user's entire songlist
UserSonglist = UserSonglist.apply(lambda x: pd.Series(x))  # turn list of lists into a dataframe

# select which users to get data from
for i in range(UserSonglist.shape[0]):
    PlayEvents = len(UserSonglist.iloc[i, :].dropna())
    print("Row:", i,
          "- #PlayEvents:", PlayEvents,
          "- Average plays:",
          round(len(UserSonglist.iloc[i, :].dropna()) / len(UserSonglist.iloc[i, :].dropna().unique()), 1))
