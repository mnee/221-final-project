import pickle
with open('train_song_data_1543840506.251837.pkl', 'rb') as unpkl:
    trainSongData = pickle.load(unpkl)
file = open('popularity_rankings.txt', 'w')
for song in trainSongData:
    print(song[2])
    file.write(str(song[2]) + '\n')

    # print('song name: {}, \n song popularity: {}'.format(song[0]['name'], song[2]))
