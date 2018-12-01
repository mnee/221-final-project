import spotipy
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import random
from nltk.corpus import words
from spotify_data_loader import SpotifyDataLoader
from k_means import k_means_vectorized

def main():
	# Load train data
	sp = SpotifyDataLoader()
	sp.load_credentials()
	trainSongs = sp.load_song_data(num_songs=10, num_iters=20, offset_max=300)
	featureVecTrain, popularitiesTrain = sp.get_all_features()

	# Assign by k-means
	assignments, centroids = k_means_vectorized(featureVecTrain, k = 18) #(k = 18, song_num = 2000) was v good
	print('assignments: {}'.format(assignments))
	print('centroids: {}'.format(centroids))

	# Load test data
	testSongs = sp.load_song_data(num_songs=1, num_iters=100, offset_max=300)
	featureVecTest, popularitiesTest = sp.get_all_features()

	# Assign test examples to centroids
	testAssignments = []
	for t_features in featureVecTest:
		# Get distance to each centroid
		distances = [(np.linalg.norm(c-t_features), i) for i, c in enumerate(centroids)]
		closest_c = min(distances, key=lambda x: x[0])
		testAssignments.append(closest_c[1])

	print(testAssignments)

if __name__ == "__main__":
	main()




	

