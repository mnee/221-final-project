import numpy as np
import glob, os, re, math, spotipy
import lyricsgenius as genius
import pickle
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import random
from nltk.corpus import words
from nltk.stem import PorterStemmer
from collections import Counter
from spotify_data_loader import SpotifyDataLoader
from k_means import k_means_vectorized
import string, time
from sklearn.feature_extraction import DictVectorizer

num_centroids = 8 #Move back to 18
api = genius.Genius("VVjnq1C3_tUwoQEOpt3rkCEaWCENVvGkCvHKwEfK7ZqnNrth6OTuzl4nAX2lC88w")

def mult_round(x, base=5):
	return int(base * math.ceil(float(x)/base))

def unpack_feature_data(cur_song, feature_data, train_keys=None, isTest=False):
	feature_dict = {}
	for k, v in feature_data.items():
		if k == 'audio_features':
			feature_dict['energy'], feature_dict['liveness'], feature_dict['tempo'],\
			 	feature_dict['speechiness'], feature_dict['acousticness'], \
				feature_dict['instrumentalness'], feature_dict['danceability'],\
				feature_dict['duration_ms'], feature_dict['valence'] = v[0], v[1], \
				v[2], v[3], v[4], v[5], v[6], v[7], v[8]
		elif k in {'unigrams', 'bigrams', 'trigrams'}:
			for ngram in v:
				if not isTest or (isTest and str(ngram) in train_keys):
					feature_dict[str(ngram)] = 1
		else:
			feature_dict[k] = v
		# Add year
		year_key = cur_song['album']['release_date'][:4]
		if not isTest or (isTest and year_key in train_keys):
			feature_dict[year_key] = 1
	return feature_dict

def make_filled_feature_vec(train_keys):
	feature_dict = {}
	for k in train_keys:
		feature_dict[k] = 1
	return feature_dict

def get_models(assignments, songData, curr_num_centroids):
	# Separate data by centroid
	song_data_by_centroid = [([],[]) for _ in range(curr_num_centroids)]

	for song_i, centroid_i in enumerate(assignments):
		curr_song = songData[song_i]
		curr_feature_dict, curr_popularity = unpack_feature_data(curr_song[0], curr_song[1]), curr_song[2]
		song_data_by_centroid[centroid_i][0].append(curr_feature_dict)
		song_data_by_centroid[centroid_i][1].append(mult_round(curr_popularity))

	v_train = [DictVectorizer(sparse=True) for _ in range(curr_num_centroids)]
	song_data_by_centroid = [(v_train[i].fit_transform(D[0]), D[1]) for i, D in enumerate(song_data_by_centroid)]

	models = []
	for data in song_data_by_centroid:
		m = LogisticRegression()
		try:
			m.fit(data[0],np.array(data[1]))
		except:
			print("One cluster contains only one class, not enough data")
			# TODO: Write code to not crash on this exception, or ensure varied data earlier
		models.append(m)

	return models, v_train

# Returns a feature vec of n-grams,

#total words, len of set of words, avg words per stanza, chorus to verse ratio, overall repetition score
def process_lyrics(lyric_path):
	total_words, stanza_count, chorus_count, verse_count = 0, 0, 0, 0
	brackets_regex = re.compile(r'^\[.*\]$')
	chorus_regex = re.compile(r'^\[Chorus.*\]$')
	verse_regex = re.compile(r'^\[Verse.*\]$')
	unigrams, bigrams, trigrams = set(), set(), set()
	with open(lyric_path, 'r') as f:
		for line in f:
			line = line.strip()
			if brackets_regex.search(line) or not line:
				stanza_count += 1
				if chorus_regex.search(line):
					chorus_count += 1
				elif verse_regex.search(line):
					verse_count += 1
			else:
				# Lyric line
				for punct in string.punctuation:
					line = line.replace(punct,'')
					#get number of words per line
					words = line.split()
					total_words += len(words)
				ps = PorterStemmer()
				words = ['<s>','<s>'] + [ps.stem(w) for w in words] + ['</s>','</s>']
				for i, w in enumerate(words):
					unigrams.add(w)
					try:
						bigrams.add((words[i-1],w))
						trigrams.add((words[i-2],words[i-1],w))
					except:
						pass
	return {'unigrams': unigrams, 'bigrams': bigrams, 'trigrams': trigrams,\
	 'UWC': len(unigrams), 'total_words': total_words, \
		'WPS':total_words/float(max(stanza_count, 1.0)), 'CVR': chorus_count/float(max(verse_count, 1.0))}

def load_lyrics(allSongs, sp_data_loader, isTest=False, trainSongs=[]):
	newSongs = []
	# Saves song lyrics to lyrics folder
	for i, ts in enumerate(allSongs):
		print("Iteration {} of {}".format(i, len(allSongs)))
		songLyrics, audio_features, popularity = None, np.array([]), -1
		newSong = ts
		songName, songArtist = None, None

		while (songLyrics == None or audio_features == np.array([]) or popularity == -1)\
		  and (not isTest or (isTest and newSong not in trainSongs)):
			songName, songArtist = newSong['name'], newSong['artists'][0]['name']
			try:
				songLyrics = api.search_song(songName, songArtist)
			except:
				songLyrics, audio_features, popularity = None, np.array([]), -1
				pass
			newSong = sp_data_loader.load_song_data(num_songs=1, num_iters=1, \
				offset_max=300)[0]
			while newSong in allSongs:
				newSong = sp_data_loader.load_song_data(num_songs=1, num_iters=1, \
					offset_max=300)[0]
			audio_features, popularity = sp_data_loader.get_features(newSong)
		if songName and songArtist:
			songName, songArtist = songName.replace('/', ''), songArtist.replace('/', '')
			lyric_path = './lyrics/' + songName + '_' + songArtist + '.txt'
			exists = os.path.isfile(lyric_path)
			if not exists:
				songLyrics.save_lyrics(filename=lyric_path)
			else: print('************************************************************')
			print("Popularity of song used is : {}".format(popularity))
			plFeatures = process_lyrics(lyric_path)
			plFeatures['audio_features'] = audio_features
			newSong = (newSong, plFeatures, popularity)
			newSongs.append(newSong)
		else: songLyrics, audio_features, popularity = None, np.array([]), -1

	return newSongs

def main():
	# Load train data
	# sp_train = SpotifyDataLoader()
	# sp_train.load_credentials()
	# trainSongs = sp_train.load_song_data(num_songs=2, num_iters=250, offset_max=300)
	# print('LEN TRAIN SONGS: {}'.format(len(trainSongs)))
	# trainSongData = load_lyrics(trainSongs, sp_train)
	# with open("train_song_data_" + str(time.time()) + ".pkl", 'wb') as pkl:
	# 	pickle.dump(trainSongData, pkl)

	# UNPICKLE CODE
	trainSongData = None
	with open('train_song_data_1543840506.251837.pkl', 'rb') as unpkl:
		trainSongData = pickle.load(unpkl)

	# # Get features after lyrics in order to filter trainSongs for just those
	# # lyrics
	# featureVecTrain, popularitiesTrain = sp.get_all_features(trainSongs)
	#
	# print('failed songs to add:{}'.format((failed/float(len(trainSongs)))))

	# os.chdir('./lyrics')
	#
	# #Iterates through all lyric files in lyrics folder
	# for file in glob.glob('*.txt'):
	# 	print(file)
	# 	plFeatures = process_lyrics('./' + file, sp)
	# 	print(plFeatures['bigrams'])

	# Load test data
	sp_test = SpotifyDataLoader()
	sp_test.load_credentials()
	testSongs = sp_test.load_song_data(num_songs=1, num_iters=50, offset_max=300)
	testSongData = load_lyrics(testSongs, sp_test, isTest=True, trainSongs=[t[0] for t in trainSongData])
	with open('test_sample.pkl', 'wb') as pkl:
		pickle.dump(testSongData, pkl)

	kmeans_feature_vecs = np.array([])
	for song in trainSongData:
		cur_vec = song[1]
		X = cur_vec['audio_features']
		X = np.append(X, [cur_vec['UWC'], cur_vec['total_words'], \
		cur_vec['WPS'], cur_vec['CVR']])
		if kmeans_feature_vecs.size == 0:
			kmeans_feature_vecs = np.array([X])
		else:
			print('X:{}, featureVec:{}'.format(X.shape, kmeans_feature_vecs.shape))
			kmeans_feature_vecs = np.vstack((kmeans_feature_vecs, X))
	topk = []
	for curr_num_centroids in range(6, 12, 1):
		num_centroids = curr_num_centroids
		print('***********************************************************************************')
		print('                                    k = {}                                         '.format(num_centroids))
		print('***********************************************************************************')
		trials = []
		for trial_count in range(10):
			# Assign by k-means
			print('first vec: \n {}'.format(kmeans_feature_vecs[0]))
			assignments, centroids = k_means_vectorized(kmeans_feature_vecs, k = num_centroids) #(k = 18, song_num = 2000) was v good
			print('assignments: {}'.format(assignments))
			print('centroids: {}'.format(centroids))

			# Build logistic regression models for each centroid
			models, v_train = get_models(assignments, trainSongData, curr_num_centroids)

			# Assign test examples to centroids
			testAssignments = []
			for t_features in [t[1] for t in testSongData]:
				X = t_features['audio_features']
				X = np.append(X, [t_features['UWC'], t_features['total_words'], \
				t_features['WPS'], t_features['CVR']])
				# Get distance to each centroid
				distances = [(np.linalg.norm(c-X), i) for i, c in enumerate(centroids)]
				closest_c = min(distances, key=lambda x: x[0])
				testAssignments.append(closest_c[1])

			print("Test Assignments: \n {}".format(testAssignments))
			avg_diff = 0
			v_test = DictVectorizer(sparse=True)
			train_keys = [v_centroid_i.get_feature_names() for v_centroid_i in v_train]
			all_key_entries = [make_filled_feature_vec(v_keys_i) for v_keys_i in train_keys]

			for song_i, centroid_i in enumerate(testAssignments):
				pred_popularity = models[centroid_i].predict(v_test.fit_transform(\
				[unpack_feature_data(testSongData[song_i][0], testSongData[song_i][1], train_keys=train_keys[centroid_i], isTest=True), all_key_entries[centroid_i]]))[0]
				test_popularity = mult_round(testSongData[song_i][2])
				avg_diff += abs(pred_popularity - test_popularity)
				print("Predicted: {} \n Actual: {}\n".format(pred_popularity, test_popularity))
			print("Average diff: {}".format(float(avg_diff)/len(testAssignments)))
			trials.append(float(avg_diff)/len(testAssignments))
		topk.append((num_centroids, min(trials)))
	print('topk: \n {}'.format(topk))
	print('best: {}'.format(min(topk, key = lambda x:x[1] )))
	#Clears lyrics
	# print('CLEANING UP YOUR SHIT!')
	files = glob.glob('./lyrics/*')
	for f in files: os.remove(f)

if __name__ == "__main__":
	start = time.time()
	main()
	end = time.time()
	print('TIME TAKEN TO RUN: {}'.format(end - start))
