import spotipy
import numpy as np
import math
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import random
from nltk.corpus import words
from spotify_data_loader import SpotifyDataLoader
from k_means import k_means_vectorized

num_centroids = 18

def mult_round(x, base=5):
	return int(base * math.ceil(float(x)/base))

def get_models(assignments, featureVecTrain, popularitiesTrain):
	# Separate data by centroid
	song_data_by_centroid = [([],[]) for _ in range(num_centroids)]
	for song_i, centroid_i in enumerate(assignments):
		song_data_by_centroid[centroid_i][0].append(featureVecTrain[song_i])
		song_data_by_centroid[centroid_i][1].append(mult_round(popularitiesTrain[song_i]))

	models = []
	for data in song_data_by_centroid:
		m = LogisticRegression()
		try:
			m.fit(np.array(data[0]),np.array(data[1]))
		except:
			print("One cluster contains only one class, not enough data")
			# TODO: Write code to not crash on this exception, or ensure varied data earlier
		models.append(m)

	return models
	
def main():
	# Load train data
	sp = SpotifyDataLoader()
	sp.load_credentials()
	trainSongs = sp.load_song_data(num_songs=20, num_iters=500, offset_max=300)
	featureVecTrain, popularitiesTrain = sp.get_all_features()

	# Assign by k-means
	assignments, centroids = k_means_vectorized(featureVecTrain, k = num_centroids) #(k = 18, song_num = 2000) was v good
	print('assignments: {}'.format(assignments))
	print('centroids: {}'.format(centroids))

	# Build logistic regression models for each centroid
	models = get_models(assignments, featureVecTrain, popularitiesTrain)

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

	avg_diff = 0
	for song_i, centroid_i in enumerate(testAssignments):
		pred_popularity = models[centroid_i].predict([featureVecTest[song_i]])[0]
		test_popularity = mult_round(popularitiesTest[song_i])
		avg_diff += abs(pred_popularity - test_popularity)
		print("Predicted: {} \n Actual: {}\n").format(pred_popularity, test_popularity)
	print("Average diff: {}".format(float(avg_diff)/len(testAssignments)))

if __name__ == "__main__":
	main()




	

