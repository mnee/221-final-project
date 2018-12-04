import spotipy
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials
import random
from nltk.corpus import words

class SpotifyDataLoader:
	def __init__(self):
		self.sp = None
		self.search_word_file_path = 'song_words.txt'
		self.all_track_data = []

	def load_credentials(self):
		client_credentials_manager = SpotifyClientCredentials(client_id="c155fe505c654167966a78cf97bd0423",
        client_secret="7b2c4ca8d4744e94ad276639ed71cac9")
		self.sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

	def load_search_words(self):
		words = []
		with open(self.search_word_file_path, 'r') as f:
			for line in f:
				words.append(line.strip())
		return words

	def load_song_data(self, num_songs=20, num_iters=500, offset_max=300):
		words = self.load_search_words()

		# Clear it out if already loaded
		self.all_track_data = []
		for i in range(num_iters):
			print("Iteration {} of {}".format(i, num_iters))
			offset_max_temp, num_songs_temp = offset_max, num_songs
			try:
				word = words.pop(random.randint(0, len(words)-1))
				print(word)
				tracks = None
				while not tracks:
					print("Offset is {}. Num_songs is {}.".format(offset_max_temp, num_songs_temp))
					try:
						offset_idx = random.randint(0, offset_max_temp)
						tracks = self.sp.search(q=word, limit=num_songs_temp, offset=offset_idx, type='track')['tracks']['items']
					except:
						offset_max_temp = max(0, offset_max_temp - 50)
						if offset_max_temp == 0:
							if num_songs_temp > 0:
								num_songs_temp -= 1
							else:
								break
			except:
				print("Used up all words in song_word_list after {} iterations".format(i))

			self.all_track_data.extend(tracks)

		return self.all_track_data

	def get_features(self, song):
		uri, popularity = song['uri'], song['popularity']
		f = self.sp.audio_features(str(uri))[0]
		if f:
			X = np.array([f['energy'], f['liveness'], f['tempo'], f['speechiness'], f['acousticness'],\
				 f['instrumentalness'], f['danceability'], f['duration_ms'], f['valence']])
			return X, popularity
		print("Error failed to load features of song {}".format(uri))
		return np.array([]), -1


	# NOTE: Now broken due to messing around with get_features
	def get_all_features(self, track_data):
		featureVec = np.array([[]])
		popularities = []
		for i, song in enumerate(track_data):
			print('iteration {}'.format(i))
			X, popularity = self.get_features(song)
			if popularity != None:
				if featureVec.size == 0: featureVec = np.array([X])
				else:
					print('X:{}, featureVec:{}'.format(X.shape, featureVec.shape))
					featureVec = np.vstack((featureVec, X))
				popularities.append(popularity)
		return featureVec, np.array(popularities)
