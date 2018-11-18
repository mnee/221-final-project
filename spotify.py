import spotipy
import numpy as np
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import random
from nltk.corpus import words

client_credentials_manager = SpotifyClientCredentials(client_id="c155fe505c654167966a78cf97bd0423",
        client_secret="7b2c4ca8d4744e94ad276639ed71cac9")
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def k_means_vectorized(features, k = 6, num_iters = 100):
	""" Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """

	N, D = features.shape
	assert N >= k, 'Number of clusters cannot be greater than number of points'

	# Randomly initalize cluster centers
	idxs = np.random.choice(N, size=k, replace=False)
	centers = features[idxs]
	assignments = np.zeros(N)
	print(N*k)
	for n in range(num_iters):
	    prev = centers.copy()
	    means = np.asarray([np.array([]) for i in range(k)])
	    feat = np.repeat(features, (k), axis = 0)
	    cent_init = np.tile(centers, (len(features), 1))
	    diff = np.linalg.norm(cent_init - feat, axis = 1).reshape(N, k)
	    assignments = np.argmin(diff, axis = 1)
	    for i in range(k):
	        want = features[assignments == i,:]
	        if len(want) == 0: centers[i] = 0
	        else: centers[i] = sum(want)/float(len(want))
	    if np.array_equal(centers, prev): break

	return assignments, centers


def main():
	words = []
	with open('song_words.txt', 'r') as f:
		for line in f:
			words.append(line.strip())

	all_tracks = []
	for _ in range(200):
		word = words.pop(random.randint(0, len(words)-1))
		num_songs = 10
		offset_max = 300
		tracks = None
		while not tracks:
			try:
				offset_idx = random.randint(0, offset_max)
				tracks = sp.search(q=word, limit=num_songs, offset=offset_idx, type='track')['tracks']['items']
				# print(word)
				# print(offset_idx)
				# print([(t['name'], t['popularity']) for t in tracks])
			except:
				offset_max = max(0, offset_max - 100)
				if offset_max < 100:
					num_songs -= 1

		all_tracks.extend(tracks)

	# featureVec = np.array([[]])
	featureVec = np.array([[]])
	popularities = []
	# print('init_shape: {}'.format(featureVec.shape))
	iterations = 0
	for t in all_tracks:
		uri, popularity = t['uri'], t['popularity']
		f = sp.audio_features(str(uri))[0]
		if f:
			X = np.array([f['energy'], f['liveness'], f['tempo'], f['speechiness'], f['acousticness'],\
				 f['instrumentalness'], f['danceability'], f['duration_ms'], f['valence']])
			if featureVec.size == 0: featureVec = np.array([X])
			else:featureVec = np.vstack((featureVec, X))
			popularities.append(popularity)
		print('iteration {}'.format(iterations))
		iterations += 1
	assignments, centroids = k_means_vectorized(featureVec, k = 18) #(k = 18, song_num = 2000) was v good
	print('assignments: {}'.format(assignments))
	print('centroids: {}'.format(centroids))

	storage_map = {}
	for i, a in enumerate(assignments):

		curr = storage_map.get(a, (0, 0)) # (popularity, num_ass)
		storage_map[a] = (curr[0]+popularities[i], curr[1]+1)
	storage_map = {k:(float(v[0])/float(v[1])) for k, v in storage_map.iteritems()}
	print('averages: {}'.format(storage_map))

	# mean = np.mean(storage_map.values())
	# std = np.std(storage_map.values())



	mini = min(storage_map.values())
	maxi = max(storage_map.values())
	print('min: {}, max:{}'.format(mini, maxi))

	for key, val in storage_map.iteritems():
		if maxi == mini: print('std is 0!!!!!!!!!!!!!!!!!!!!!')
		else: storage_map[key] = 100*((val - mini)/float(maxi - mini))
	print('normalized map: {}'.format(storage_map))
	if assignments[a] not in storage_map: storage_map[assignments[a]] = (popularities[a], centroids[assignments[a]])


	# RUN TESTING
	test_tracks = []
	for _ in range(100):
		# Use same words ensures no overlap of training and test set
		word = words.pop(random.randint(0, len(words)-1))
		num_songs = 1
		offset_max = 300
		tracks = None
		while not tracks:
			try:
				tracks = sp.search(q=word, limit=num_songs, offset=random.randint(0, offset_max), type='track')['tracks']['items']
			except:
				offset_max = max(0, offset_max - 100)
				if offset_max < 100:
					limit -= 10

		test_tracks.extend(tracks)

	testFeatureVec = np.array([[]])
	testPopularities = np.array([])
	for t in test_tracks:
		uri, true_popularity = t['uri'], t['popularity']
		f = sp.audio_features(str(uri))[0]
		if f:
			X = np.array([f['energy'], f['liveness'], f['tempo'], f['speechiness'], f['acousticness'],\
				 f['instrumentalness'], f['danceability'], f['duration_ms'], f['valence']])
			testPopularities = np.append(testPopularities, true_popularity)
			if testFeatureVec.size == 0: testFeatureVec = np.array([X])
			else:testFeatureVec = np.vstack((testFeatureVec, X))

	predPopularities = []
	for t_features in testFeatureVec:
		# Get distance to each centroid
		distances = [(np.linalg.norm(c-t_features), i) for i, c in enumerate(centroids)]
		total = sum([d[0] for d in distances])
		weights = [(float(total)/d[0], d[1]) for d in distances]
		total = sum([w[0] for w in weights])
		weights = [(w[0]/float(total), w[1]) for w in weights]

		total_popularity = 0.0
		for (w, centroid_idx) in weights:
			total_popularity += storage_map[centroid_idx]*w
		predPopularities.append(total_popularity)

	print(testPopularities)
	print(predPopularities)
	print('dist: {}'.format(np.linalg.norm(testPopularities - predPopularities)))

		# print([(track['name'], track['popularity']) for track in tracks])


if __name__ == "__main__":
	main()

# Avg. feature vecs
# 2010: [0.7654680851063831, 0.20110638297872344, 115.86755319148938, 0.09742340425531915, 0.08838229787234042, 0.0015148402127659576, 0.6624042553191489, 228792.12765957447, 0.5902978723404255]
# 2012: [0.7311702127659576, 0.18523829787234045, 126.51521276595744, 0.09273617021276596, 0.08766689361702126, 0.00432178744680851, 0.669617021276596, 225792.17021276595, 0.5729361702127658]
# 2015: [0.7169200000000001, 0.19386799999999998, 122.21504, 0.080222, 0.15636884, 0.0001772404, 0.6813999999999999, 215109.9, 0.58394]
# song_train_uris = {\
# 	2010: ["spotify:track:5OMwQFBcte0aWFJFqrr5oj", "spotify:track:11EX5yhxr9Ihl3IN1asrfK", "spotify:track:0KpfYajJVVGgQ32Dby7e9i", "spotify:track:6tS3XVuOyu10897O3ae7bi", "spotify:track:3r04p85xiJh9Wqk59YDYdc", "spotify:track:6lV2MSQmRIkycDScNtrBXO", "spotify:track:15JINEqzVMv3SvJTAXAKED", "spotify:track:5P5cGNzqh6A353N3ShDK6Y", "spotify:track:1DqdF42leyFIzqNDv9CjId", "spotify:track:1CdqVF1ywD0ZO1zXtB9yWa", "spotify:track:59dLtGBS26x7kc0rHbaPrq", "spotify:track:3iL2l5gUqyPS6vDwJFgJTR", "spotify:track:0TyOpxlWwDx98bjkIVHUgY", "spotify:track:60jzFy6Nn4M0iD1d94oteF", "spotify:track:1IaYWv32nFFMdljBIjMY5T", "spotify:track:55qBw1900pZKfXJ6Q9A2Lc", "spotify:track:7BqBn9nzAq8spo5e7cZ0dJ", "spotify:track:2V4bv1fNWfTcyRJKmej6Sj", "spotify:track:5OiLJ8tjUPFiPX2gVM8fxJ", "spotify:track:5tXyNhNcsnn7HbcABntOSf", "spotify:track:2M9ULmQwTaTGmAdXaXpfz5", "spotify:track:7Ie9W94M7OjPoZVV216Xus", "spotify:track:5vlEg2fT4cFWAqU5QptIpQ", "spotify:track:6pL7puHxh1psrLzrAlobxQ", "spotify:track:38xWaVFKaxZlMFvzNff2aW", "spotify:track:4AboqNl74jNDpJhPfqYDmj", "spotify:track:4vp2J1l5RD4gMZwGFLfRAu", "spotify:track:3DamFFqW32WihKkTVlwTYQ", "spotify:track:2fQ6sBFWaLv2Gxos4igHLy", "spotify:track:4BycRneKmOs6MhYG9THsuX", "spotify:track:15pu8u4n3q4BKl4tF20c5v", "spotify:track:7ksYJ95P5vP87A0GH34CIk", "spotify:track:3ZdJffjzJWFimSQyxgGIxN", "spotify:track:1WtTLtofvcjQM3sXSMkDdX", "spotify:track:4DvhkX2ic4zWkQeWMwQ2qf", "spotify:track:2DHc2e5bBn4UzY0ENVFrUl", "spotify:track:4fIWvT19w9PR0VVBuPYpWA", "spotify:track:0xcl9XT60Siji6CSG4y6nb", "spotify:track:7LP4Es66zdY7CyjepqmvAg", "spotify:track:6lUY6MoqGgPnA27PHYxem5", "spotify:track:2zJZwWF7BTGIIvrAlgzJEx", "spotify:track:6epn3r7S14KUqlReYr77hA", "spotify:track:2rDwdvBma1O1eLzo29p2cr", "spotify:track:6jAsmDJI8iPhGWtS27kZ67", "spotify:track:51YhN4y2tOvfI0Sv1hoBRo", "spotify:track:3YJkAQNEhmCZGLdmPsu6Ye", "spotify:track:0dBW6ZsW8skfvoRfgeerBF"],\
# 	2011: ["spotify:track:1CkvWZme3pRgbzaxZnTl5X", "spotify:track:0IkKz2J93C94Ei4BvDop7P", "spotify:track:4wIjXMeLH3MrXEiaXNfYwC", "spotify:track:4S1qeuIO9X6oOW38r3O0R3", "spotify:track:5oBPpIwObzAbBwdINug5I6", "spotify:track:4lLtanYk6tkMvooU0tWzG8", "spotify:track:7AqISujIaWcY3h5zrOqt5v", "spotify:track:2708Nq8H3EEBcahVCNsfTe", "spotify:track:50jy5FwPSIUp8olK1KuaVS", "spotify:track:1nPPh0CiKqVvYl4Mx1aZ21", "spotify:track:3STTVKfJGTdhbg5aLppEzX", "spotify:track:08Bfk5Y2S5fCxgxk371Eel", "spotify:track:7w87IxuO7BDcJ3YUqCyMTT", "spotify:track:3wllvxhwIbGYY5rte1bZFn", "spotify:track:47Slg6LuqLaX0VodpSCvPt", "spotify:track:4270XtiZeN9yYPR81dn8Ow", "spotify:track:4zzL7ATa95g88MwzuQ1v8G", "spotify:track:6r2BECwMgEoRb5yLfp0Hca", "spotify:track:1H9AbpMAShkCVuaPAR6CHf", "spotify:track:0yjf2c8T4Nvbkw6GwamHPS", "spotify:track:5ZxCk43qwHrzVUahCyRLEm", "spotify:track:32eXkt1gAj4J9Dyav2bvSZ", "spotify:track:4etypRtRn8IgsxZu6BhRS6", "spotify:track:7oOfd5BP16mS0Vrh1PdRHt", "spotify:track:386RUes7n1uM1yfzgeUuwp", "spotify:track:737WLYKWrI76tlunnexOsq", "spotify:track:4NTWZqvfQTlOMitlVn6tew", "spotify:track:6lCTs5NDYctSQutADkeEsA", "spotify:track:3VP78k3jzm0Q5OM08E383k", "spotify:track:0pSIJCxYqZNIZBTAnIXOkv", "spotify:track:6XUIbizGHk6EJiUDWEJtpY", "spotify:track:4u26EevCNXMhlvE1xFBJwX", "spotify:track:4ywvnllm7kpaXDsa1VNNpY", "spotify:track:3hsBI1UxLFrIfYgfl56Nih", "spotify:track:47hs3xNT3iOGvgmC4eXBAi", "spotify:track:7xnp3wEx1gVT3xofhAVgwg", "spotify:track:1VDXQhu7YGdbM6UeEIfsaX", "spotify:track:7gdGdDgpCDy2kACTqvXbYa", "spotify:track:2nRvKHh2mtOfIQ4jcAkP9q", "spotify:track:0adTN3vBO3pimO3yfxm9vg", "spotify:track:2srTtSrzY4n10C7abVTrBm", "spotify:track:70Wb2xKOAOte8iNkBpezQs", "spotify:track:18qk1M9s2xtu53fEQtM3bE", "spotify:track:4Fmwzj3lxLrIVsLyEt4p8G", "spotify:track:3XqdUqkc3t3klNV0ykFQu3", "spotify:track:7Ee0JfcUAJq7pDFrnx5WYF", "spotify:track:2SsYUdeL0WBlL6CipICSN5"],\
# 	2012: ["spotify:track:4wCmqSrbyCgxEXROQE6vtV", "spotify:track:3TGRqZ0a2l1LRblBkJoaDx", "spotify:track:3ehrxAhYms24KLPG8FZe0W", "spotify:track:0iyEaciAmtiv8xMkBg97Fy", "spotify:track:7gUpO6td4OOnu0Lf9vhcIV", "spotify:track:3AGOgQzp0YcPH41u9p7dOp", "spotify:track:6D60klaHqbCl9ySc8VcRss", "spotify:track:3t4HeVfURZ5DuqMWz0Cmqg", "spotify:track:1T07kafGtJbDvosN0nu4Q2", "spotify:track:3CKCZ9pfwAfoMZlMncA1Nc", "spotify:track:0obBFrPYkSoBJbvHfUIhkv", "spotify:track:6t6oULCRS6hnI7rm0h5gwl", "spotify:track:5JLv62qFIS1DR3zGEcApRt", "spotify:track:62RX5FpSBIfVBheNMtiDMX", "spotify:track:0ltBH1JNzSvQJPjJpvTu9B", "spotify:track:3icobpUUcKEbt2zQUqRpvM", "spotify:track:0KAiuUOrLTIkzkpfpn9jb9", "spotify:track:49ySwzAyvxcNXOkOP6ZB1L", "spotify:track:3tyPOhuVnt5zd5kGfxbCyL", "spotify:track:1akkzUk0nUq90X56xDVGHZ", "spotify:track:4wTMBYRE6xVTIUQ6fEudsJ", "spotify:track:47Z5890IcjSed81ldeLgqc", "spotify:track:53QF56cjZA9RTuuMZDrSA6", "spotify:track:0RUGuh2uSNFJpGMSsD1F5C", "spotify:track:4qikXelSRKvoCqFcHLB2H2", "spotify:track:2L7rZWg9RLxIwnysmxm4xk", "spotify:track:70dWrqAp30TmWeibQkn0i7", "spotify:track:4VySpxhRGy32u5zPCprzDn", "spotify:track:0zJlM4BzA3riILBJZ8uCvs", "spotify:track:5HQVUIKwCEXpe7JIHyY734", "spotify:track:53HDOVjPio9hPhpE935MAu", "spotify:track:0pwYLVXVknPSGUQb39cePC", "spotify:track:0UIiTEMdonJvQX2Z2gVm8j", "spotify:track:2LNbLA2JSQlv6NRz1xdZRj", "spotify:track:6LS6pltO7YBgjwNVhxMwtp", "spotify:track:1kPpge9JDLpcj15qgrPbYX", "spotify:track:3sP3c86WFjOzHHnbhhZcLA", "spotify:track:6hkOqJ5mE093AQf2lbZnsG", "spotify:track:1gihuPhrLraKYrJMAEONyc", "spotify:track:4kflIGfjdZJW4ot2ioixTB", "spotify:track:5sAlmL7Qp9N9BJTADkwEt9", "spotify:track:7LBohG6plAhWaHjuzi4CpY", "spotify:track:6MAdEUilV2p9RQUqE5bMAK", "spotify:track:03UrZgTINDqvnUMbbIMhql", "spotify:track:62zFEHfAYl5kdHYOivj4BC", "spotify:track:6RM5RZ6HF1pxQLCaJgMcuQ", "spotify:track:6NFE85sGFYgJj9y1EvjOV3"],\
# 	2013: [],\
# 	2014: [],\
# 	2015: ["spotify:track:09CtPGIpYB4BrO8qb1RGsF", "spotify:track:6FXNfYIY6iEeCKuIeSYWgg", "spotify:track:7jv4Mb21bdd9SDVOm9Fm9l", "spotify:track:2sC2P3BN0IXujNaaSyDmtP", "spotify:track:1KK0j3djUgx5tuOMwesdAT", "spotify:track:7x5xYW5W42OGPAdHUyyguy", "spotify:track:0ENSn4fwAbCGeFGVUbXEU3", "spotify:track:2fykwqa2RQzI6nBCwhTfqD", "spotify:track:6ukMqDxnOPOgoHdak7Kyp3", "spotify:track:3JiockjOTd8m2VGcTGkmew", "spotify:track:33NHr7lu7tCMjK05jUj1v0", "spotify:track:6DEaND0SHv3sC11xobZLiy", "spotify:track:3oWkBnpg6cYYwUka7kmomo", "spotify:track:6nRwc5GgNvBMkKaynhQzrm", "spotify:track:0HMjXBAZmSYOTTi33WpMso", "spotify:track:0yN4fNzmVnmgC0dsOoi9Wh", "spotify:track:70AoHzwXbJBdz3DZ5fSDfY", "spotify:track:5ByIHT8s38diBQf6dkEWbt", "spotify:track:3ulIErpIehDSLfmQmUax5g", "spotify:track:4KcVVhAaHxqtX2ANt4b3tc", "spotify:track:33okDqzPPYo8vwC3Mc1pOr", "spotify:track:33okDqzPPYo8vwC3Mc1pOr", "spotify:track:6YhDHby2eVeENKJNa7C2z6", "spotify:track:34ONrmvZfttfTZR5NXrC6e", "spotify:track:2NUywKUCM9T0Csy1e6sllv", "spotify:track:2bZMOs3RjmhhGca6MEzjyl", "spotify:track:0NTMtAO2BV4tnGvw9EgBVq", "spotify:track:2sNvitW3TxiTeC9xT9f2ZZ", "spotify:track:62ke5zFUJN6RvtXZgVH0F8", "spotify:track:53b7YqVw2Vc1oRrYeOm7z7", "spotify:track:4rmPQGwcLQjCoFq5NrTA0D", "spotify:track:7J4gq1xNP3IsG6lDk0eSa7", "spotify:track:34gCuhDGsG4bRPIf9bb02f", "spotify:track:5MhsZlmKJG6X5kTHkdwC4B", "spotify:track:5a5so8nqDGq75MI5WMbBtT", "spotify:track:3s4U7OHV7gnj42VV72eSZ6", "spotify:track:41Ypl7Pzkod2H0VlcZH5DS", "spotify:track:6NkRjdInQuM5qRgeYUDCZe", "spotify:track:0ifSeVGUr7py5GggttDhXw", "spotify:track:2YlZnw2ikdb837oKMKjBkW", "spotify:track:1BECwm5qkaBwlbfo4kpYx8", "spotify:track:3qhobDAfBcVoOWZP8Ck8R2", "spotify:track:4NphdFxnXNaqYM0XkdrRUT", "spotify:track:1cdzfFjEbUbgTm5nv3FgXR", "spotify:track:5NQbUaeTEOGdD6hHcre0dZ", "spotify:track:4oHmgneU9dwYoqg0SJSOCf", "spotify:track:7m2t7klcyyN0mqxVweaslO"],\
# 	2016: ["spotify:track:1MDoll6jK4rrk2BcFRP5i7", "spotify:track:69bp2EbF7Q2rqc5N3ylezZ", "spotify:track:3hB5DgAiMAQ4DzYbsMq1IT", "spotify:track:0IKK48xF4eEdfofyaeKWWO", "spotify:track:14WWzenpaEgQZlqPq2nk4v", "spotify:track:14WWzenpaEgQZlqPq2nk4v", "spotify:track:46THN9jjPWhSqFUu6YsBhv", "spotify:track:0JoaFxLgrqbWutREzcZBzS", "spotify:track:6JV2JOEocMgcZxYSZelKcc", "spotify:track:3CRDbSIZ4r5MsZ0YwxuEkn", "spotify:track:1i1fxkWeaMmKEB4T7zqbzK", "spotify:track:7BKLCZ1jbUBVqRi2FVlTVw", "spotify:track:6b3b7lILUJqXcp6w9wNQSm", "spotify:track:5kqIPrATaCc2LqxVWzQGbk", "spotify:track:3kfxarilcBr81mb2hmZLeh", "spotify:track:3jomjC6H7YQBRr2CHPtc4y", "spotify:track:3CCyVdprlcXui4ZwMw1hNS", "spotify:track:0SJPThTy7ynySPF4euczx7", "spotify:track:0azC730Exh71aQlOt9Zj3y", "spotify:track:7L5jgZtAyfiU7elB8DIqCx", "spotify:track:0KMYgSe9JWHloFeEVBU6qq", "spotify:track:2Z8WuEywRWYTKe1NybPQEW", "spotify:track:6i0V12jOa3mr6uu4WYhUBr", "spotify:track:1WP1r7fuvRqZRnUaTi2I1Q", "spotify:track:1UfBAJfmofTffrae5ls6DA", "spotify:track:6DNtNfH8hXkqOX1sjqmI7p", "spotify:track:3KZcrZ36LW9RnChK1iIkth", "spotify:track:3vv9phIu6Y1vX3jcqaGz5Z", "spotify:track:53B2XmmjJ9rriW1qciMBeX", "spotify:track:11KJSRSgaDxqydKYiD2Jew", "spotify:track:7EiZI6JVHllARrX9PUvAdX", "spotify:track:3pzjHKrQSvXGHQ98dx18HI", "spotify:track:25khomWgBVamSdKw7hzm3l", "spotify:track:7K5dzhGda2vRTaAWYI3hrb", "spotify:track:0qy5D3OJre7SPJNMOL9I71", "spotify:track:3pXF1nA74528Edde4of9CC", "spotify:track:7l94dyN2hX9c6wWcZQuOGJ", "spotify:track:27GmP9AWRs744SzKcpJsTZ", "spotify:track:7vRriwrloYVaoAe3a9wJHe", "spotify:track:1wYZZtamWTQAoj8B812uKQ", "spotify:track:1BZG99C7Co1r6QUC3zaS59", "spotify:track:4CpKEkdGbOJV51cSvx7SoG", "spotify:track:2YlZnw2ikdb837oKMKjBkW", "spotify:track:7soJgKhQTO8hLP2JPRkL5O", "spotify:track:13HVjjWUZFaWilh2QUJKsP", "spotify:track:0l0CvurVUrr2w3Jj1hOVFc", "spotify:track:6hmhG1b4LEyNuashVvuIAo"]}

# song_test_uris = {\
# 	2010: ["spotify:track:7jmqSz6yTrGWI1kNqNYI28", "spotify:track:2IpGdrWvIZipmaxo1YRxw5", "spotify:track:7yws3pF3FFguwT2Psi6c15", "spotify:track:5XRHGXut00SrJUFmcn2lQF", "spotify:track:72phhQtfwZcRweweRxxkmU", "spotify:track:3bMNprrp2JDKZsGbiXpsJl", "spotify:track:3evTG0FJEHFYmgOeFgPo0s", "spotify:track:3ubtdCmWZIuX8FuRv3S3KX", "spotify:track:6BdgtqiV3oXNqBikezwdvC", "spotify:track:1fBl642IhJOE5U319Gy2Go", "spotify:track:6KBYk8OFtod7brGuZ3Y67q", "spotify:track:5uHYcK0nbEYgRaFTY5BqnP", "spotify:track:3FF6jZY1wDHXJUUN18qsLx", "spotify:track:4faiJXyBflUVVOOE9fxbeg", "spotify:track:3bGxiCPouq52ZtI8ybMCX5", "spotify:track:5m7T3eJlK0O92QOCT2SpVM", "spotify:track:3QkXEz9TxApuk1JpDmeWnS", "spotify:track:2d7fRuDlFZfKIoSuf8bhGv", "spotify:track:45EDI3rk0f4cAMt9f8b56R", "spotify:track:6HSqyfGnsHYw9MmIpa9zlZ", "spotify:track:1MaqkdFNIKPdpQGDzme5ss", "spotify:track:17tDv8WA8IhqE8qzuQn707", "spotify:track:7ceIquYtiTYlgSSm7PqUf9", "spotify:track:5K5LbSTVuKKe1KGMNfBgIW", "spotify:track:1NhPKVLsHhFUHIOZ32QnS2", "spotify:track:1YaVmBh7EAeR54FIjuFcb5", "spotify:track:4JOP8ELK6AaeySe7sKe996", "spotify:track:6u5M4jPpYkoRV4vVHDQvkd", "spotify:track:6H2wnX7ytNeCKERIVqCwgs", "spotify:track:0JcKdUGNR7zI4jJDLyYXbi", "spotify:track:6mjDmlsMxnRDati2rC1QFL", "spotify:track:6tp0eVhr6SI7FWPJasG86O", "spotify:track:7qje9qxLncMESiPeI27SAn", "spotify:track:1FKxKGONukVFXWVJxAKmlz", "spotify:track:5VGlqQANWDKJFl0MBG3sg2", "spotify:track:4Oq3R4drZZfGWdxbhKrPvb", "spotify:track:75vEuMA15pb8JmgrGugHEs", "spotify:track:1kMuU3TNQvHbqvXCWBodmP", "spotify:track:02eD9ymfJOJOhM97HYp5R9", "spotify:track:4u26EevCNXMhlvE1xFBJwX", "spotify:track:7JIuqL4ZqkpfGKQhYlrirs", "spotify:track:4xInIiKipU1mtUogJ3ZdYr", "spotify:track:3VA8T3rNy5V24AXxNK5u9E", "spotify:track:4gWTBq5Jftq5CEKYpLXD8R", "spotify:track:4ZC8hXXqu2hPcDLw9QTdtQ", "spotify:track:61LtVmmkGr8P9I2tSPvdpf", "spotify:track:1yK9LISg5uBOOW5bT2Wm0i", "spotify:track:5UzU0qw21okODBNUnxptVo", "spotify:track:3GCL1PydwsLodcpv0Ll1ch"],\
# 	2011: ["spotify:track:3G1Na74xxuAhiWLyXDPCFN", "spotify:track:3ZdJffjzJWFimSQyxgGIxN", "spotify:track:5ryMR3slEKQXaGYGdZVSSw", "spotify:track:74yEgTOgJIohfN1zh52AjN", "spotify:track:2p3ygTLthnqH1XmE8b7mtS", "spotify:track:7CELQKrSCYpEi9OXWCBDek", "spotify:track:0obBFrPYkSoBJbvHfUIhkv", "spotify:track:45sDIKapDyxPl307QpEAwl", "spotify:track:7zbDSQelDmlaEhUDnLMViZ", "spotify:track:5MtbCiKeHd86jvJm5B0P4q", "spotify:track:5UnX1hCXQzmUzw2dn3L9nY", "spotify:track:5zTzDqrEmseqL2G8ElgBu7", "spotify:track:61HQWI1Woxup7CnGwVUsdI", "spotify:track:2IpGdrWvIZipmaxo1YRxw5", "spotify:track:4356Typ82hUiFAynbLYbPn", "spotify:track:09ZcYBGFX16X8GMDrvqQwt", "spotify:track:0gY2iq0xJPRoIB1PScKSw4", "spotify:track:2CAf7WlHg8SlMLHmt55pqs", "spotify:track:3t4HeVfURZ5DuqMWz0Cmqg", "spotify:track:7JkSoesJ9odqqYuzXkwEGD", "spotify:track:6szOfaZn6H3rZXnUqgJ4R7", "spotify:track:4DvhkX2ic4zWkQeWMwQ2qf", "spotify:track:7rGMKCgeYXpBecQ1FPb3oc", "spotify:track:3DrjZArsPsoqbLzUZZV1Id", "spotify:track:0RSvhappnRHZYXeEBgXHHz", "spotify:track:1nVLAEzhBXMJLR5zAl90Nl", "spotify:track:46MDLc0Yip6xCMSsdePOAU", "spotify:track:1VENmdw9mA1R8qGKTHwjqk", "spotify:track:3ts6xK5GzfMAAriT9AIBmP", "spotify:track:1yEwEiTpsaPhQi9lb5EVV4", "spotify:track:0yD66650JxhqKbW76C2qCo", "spotify:track:4ABua0yuWcpTotImEEJTaw", "spotify:track:4VTErodIk9Z2rof8g4AqaK", "spotify:track:3qgZzdpfU8wXeUFV6Jxuzi", "spotify:track:3pVYaDSQ4z0RIrxXQOzbR9", "spotify:track:5OrX4PR74Ttezdj5soO1BV", "spotify:track:1r3myKmjWoOqRip99CmSj1", "spotify:track:3djPwpsQAnp8B8gC81e7fk", "spotify:track:7q7YfXDqbqteY6BZFZr577", "spotify:track:6AgeVPT9rhr5GxCLdKPFzj", "spotify:track:3wkKkFAtYSTRwqOydW6T0I", "spotify:track:0JXXNGljqupsJaZsgSbMZV", "spotify:track:6ezyEFqZDgQUvxpaDv4jKO", "spotify:track:0w9LJae3sVlZlH2CnxTInF", "spotify:track:1WH1mGKDXbvV3EApLMOaXn", "spotify:track:6xtXdS8ALTfK0g9hOG1PSX", "spotify:track:1fBl642IhJOE5U319Gy2Go", "spotify:track:1t2hJFgJyUPD2crOafMUEk", "spotify:track:0tDElYeVQUlbGdTHf7S0bK", "spotify:track:6Uj0DygqlzJqduyny4lMA6"],\
# 	2012: ["spotify:track:3bfqkspKABT4pPicm6wC9F", "spotify:track:0wb0UFUnUq8UiAUujRB8ni", "spotify:track:3b7CDTKB0SRTmQ6ytYi5vZ", "spotify:track:4TyY4vah59nsckspWdSGPn", "spotify:track:6vkBqOgDHCkb0PV2FaH4PU", "spotify:track:12Ns5IphkblPmHxpRILG9t", "spotify:track:4dMfcca0nQNXXj1etzeME4", "spotify:track:05ZACkzW8YbRq3eFgIfSNB", "spotify:track:4urcG6Nfubqsuqy3juMjBi", "spotify:track:6j7hih15xG2cdYwIJnQXsq", "spotify:track:0BgRaFy7qnwT1PuUmSTFkf", "spotify:track:2U2ONBrf1HJCDxQlynpD6J", "spotify:track:6rbeWjEavBHvX2kr6lSogS", "spotify:track:76N7FdzCI9OsiUnzJVLY2m", "spotify:track:60NgMTkfSGmcMUModciLN4", "spotify:track:20DfkHC5grnKNJCzZQB6KC", "spotify:track:2pnmi9VUgtiEnP1iAi2xfx", "spotify:track:5BSndweF91KDqyxANsZcQH", "spotify:track:6nek1Nin9q48AVZcWs9e9D", "spotify:track:0vFMQi8ZnOM2y8cuReZTZ2", "spotify:track:5xPazRvyrkVootu8pM9vUG", "spotify:track:6cNajGUxKP53LIg0LzlfQt", "spotify:track:62csqLJbdvAKb38rYijV1R", "spotify:track:1Ejsu5JglwIgrFW7Bt2GuL", "spotify:track:226rLMKyxPvLzEAP1oUMXj", "spotify:track:6BCrbWBpb8d6KWmEqZ41tr", "spotify:track:25CA8QVJQrh5R05UUCaODM", "spotify:track:3m8CQnnfJJp4eQMWWl3zay", "spotify:track:3FjPRoxtACgDhIK3u60CsP", "spotify:track:2NniAhAtkRACaMeYt48xlD", "spotify:track:47nm8czanMUzIqHsnr5x61", "spotify:track:0laYHRpNTS6i8FXdupHkJ4", "spotify:track:4x4RbsTuuqrmRoDFMyrhnA", "spotify:track:6mnjcTmK8TewHfyOp3fC9C", "spotify:track:2Rea6PxenESVsJfEaGBGsD", "spotify:track:1QUpqu8865jfasDr8M3IKN", "spotify:track:3SETFGPl3zT8LBvLbSEqHD", "spotify:track:2tSFTvHVaLxkqXHaofPKQk", "spotify:track:0mtPNQl5vePVwiI9Vykp7L", "spotify:track:40xtweuHBkLXNMO3xCK5AZ", "spotify:track:1POAx4NMLOBPVKZUSsBh92", "spotify:track:0cV4xwUA4ue2deqq4CZFko", "spotify:track:65YsalQBmQCzIPaay72CzQ", "spotify:track:4J5Ec6Xa2NMtqxLQU7Cnry", "spotify:track:6Ymvlzom4TQeoKqAWsZRD8", "spotify:track:3trS6e40JCVUOpPVt5OdHj", "spotify:track:5r6hrzcBhlnV9eDSKD2WQ9", "spotify:track:0Nu9WA8kEbBWEsay2s8Q0U", "spotify:track:7oVEtyuv9NBmnytsCIsY5I", "spotify:track:4EfN6bixdOOgoLYR5C4cWo"],\
# 	2015: ["spotify:track:2mCF8L0brIs88eH6Kf2h9p" , "spotify:track:05cXQMJcrM9msUYu11mOrs", "spotify:track:1PWnAEQcbwQwK759otUbta" , "spotify:track:3cU2wBxuV6nFiuf6PJZNlC", "spotify:track:2wA4C7X7O9hMjuemyHwLgd", "spotify:track:3NFuE3uDOr6QUw9UZ9HzKo", "spotify:track:72Bz4ciRZPBcVSw0nrZDHi", "spotify:track:285HeuLxsngjFn4GGegGNm", "spotify:track:72ojun8ufF5Y0YzssmvVHL", "spotify:track:21Go4aMyLGP40ANI3TU0Fn", "spotify:track:6sKAs2tdtTVTO2U0IgJAur" , "spotify:track:7fPHfBCyKE3aVCBjE4DAvl", "spotify:track:7exnxg6XItzllOVVvFgwQE", "spotify:track:5aVCrbz2wVTA4OFxYd8ZGA", "spotify:track:0FxnKrx0noMmQWusERQ2O7", "spotify:track:4Aep3WGBQlpbKXkW7kfqcU", "spotify:track:4AVUqcPZB0AaRF0tCmyNiI", "spotify:track:5J4ZkQpzMUFojo1CtAZYpn", "spotify:track:4yDLW1kpgN7RGa6wCtxlL8", "spotify:track:5XzmZjXhMjDHr7ZfJ6DELQ", "spotify:track:357p2KRTBXeGGw6YfUI1nH", "spotify:track:5zlC5d5umTrbcX9sLVVxzh", "spotify:track:0fioLzGM8ngbD1w6fMmm45", "spotify:track:6kwAbEjseqBob48jCus7Sz", "spotify:track:5eWgDlp3k6Tb5RD8690s6I", "spotify:track:7q6pXvSL7G6NkvSSt7YGJ0", "spotify:track:0ct6r3EGTcMLPtrXHDvVjc", "spotify:track:02M6vucOvmRfMxTXDUwRXu", "spotify:track:0YzCQcsgj6KqwBHVx1mZrH", "spotify:track:3DmW6y7wTEYHJZlLo1r6XJ", "spotify:track:6dshconh2KBbGxVh7GtSTC", "spotify:track:2VhPOtIQw2UpQmRVevdviU", "spotify:track:3vMCg20dt8lUiPozDINUBf", "spotify:track:4E8KoNTXZnWDE3cCjjw8FU", "spotify:track:1iXBApi39l5r6lJj9WEXXS", "spotify:track:49eplIHvgCLY8HAgcaWrUb", "spotify:track:2PIvq1pGrUjY007X5y1UpM", "spotify:track:46X0dXeliHXD73sxexJx7d", "spotify:track:6eLrCtqlpjroAkES2EHCx0", "spotify:track:66hayvUbTotekKU3H4ta1f", "spotify:track:57kR5SniQIbsbVoIjjOUDa", "spotify:track:0fgZUSa7D7aVvv3GfO0A1n", "spotify:track:7fBv7CLKzipRk6EC6TWHOB", "spotify:track:0PT7nlpo11hYYyfnBgtilT", "spotify:track:11qh54D0PKkBwelpDxxiEU", "spotify:track:2NVt7fxr5GsqTkGwYXcNTE", "spotify:track:1HNkqx9Ahdgi1Ixy2xkKkL", "spotify:track:4NYwy0R3NdvORX2B6OZXBT", "spotify:track:4bBrKDo8rchikwZOtmXbIH"],\
# 	2016: ["spotify:track:6ZSO7kPn8IMJFymyticbJO","spotify:track:43PuMrRfbyyuz4QpZ3oAwN","spotify:track:5NQbUaeTEOGdD6hHcre0dZ","spotify:track:0PJIbOdMs3bd5AT8liULMQ","spotify:track:1LoriJC05IrHIDwj3q0KC1","spotify:track:0wdKiSBUT7aZkXUIdJWcwC","spotify:track:7MXVkk9YMctZqd1Srtv4MB","spotify:track:4JDZl9nKIAhAhDjw753u4X","spotify:track:5WI2ltQIdwgzf1SNE76JyR","spotify:track:25KybV9BOUlvcnv7nN3Pyo","spotify:track:3KwwE4sgCzMaKWq6QBebmX","spotify:track:70eDxAyAraNTiD6lx2ZEnH","spotify:track:5kNe7PE09d6Kvw5pAsx23n","spotify:track:6eT7xZZlB2mwyzJ2sUKG6w","spotify:track:3lSDIJ2abCrOdDJ6pshUap","spotify:track:7lGKEWMXVWWTt3X71Bv44I","spotify:track:4dASQiO1Eoo3RJvt74FtXB","spotify:track:1LxKKYsJNPeBdOwdudsJzv","spotify:track:16Ah4QqH4mgYVXqfC4mdSd","spotify:track:22VdIZQfgXJea34mQxlt81","spotify:track:3RiPr603aXAoi4GHyXx0uy","spotify:track:2Gyc6e2cLxA5hoX1NOvYnU","spotify:track:1wHZx0LgzFHyeIZkUydNXq","spotify:track:4h0zU3O9R5xzuTmNO7dNDU","spotify:track:1pKeFVVUOPjFsOABub0OaV","spotify:track:0EGuSSpuu9wmHCtvb4PdLO","spotify:track:59HjlYCeBsxdI0fcm3zglw","spotify:track:0g5EKLgdKvNlln7TNqBByK","spotify:track:4nS1sut0R2mgmFitKIShVe","spotify:track:23NWj2izXAJ4yL6Nah73wf","spotify:track:7IWkJwX9C0J7tHurTD7ViL","spotify:track:0uyZ5ckiIUbFZd3P6RWmrj","spotify:track:5wldXGLEOoRXxMWJ8rIUWE","spotify:track:2GyA33q5rti5IxkMQemRDH","spotify:track:61QSuw5VlC0LTS8WMO356g","spotify:track:5RIVoVdkDLEygELLCniZFr","spotify:track:6dlpABcXrQKRU9G00i6Zba","spotify:track:5is2KiI0FDPLPLRq9hFybw","spotify:track:376KnY4TrgBITxjlnbnmIy","spotify:track:10I3CmmwT0BkOVhduDy53o","spotify:track:7nD9nN3jord9wWcfW3Gkcm","spotify:track:4Ce37cRWvM1vIGGynKcs22","spotify:track:69uxyAqqPIsUyTO8txoP2M","spotify:track:4Pn0JlCUusD2QHjADuOzuV","spotify:track:6BbINUfGabVyiNFJpQXn3x","spotify:track:3NJG6vMH1ZsectZkocMEm0","spotify:track:2JzZzZUQj3Qff7wapcbKjc","spotify:track:0tgVpDi06FyKpA1z0VMD4v"]}

# class PopularityPredictor:
# 	def __init__(self):
# 		self.model = None


# class YearFinder:
# 	def __init__(self):
# 		self.model = KNeighborsClassifier()

# 	def train_model(self):
# 		totalFeatures = []
# 		totalYears = []
# 		for year in range(2010, 2017):
# 			uris = song_train_uris.get(year, [])
# 			if uris:
# 				features = sp.audio_features(uris)
# 				for i, f in enumerate(features):
# 					try:
# 						X = [f['energy'], f['liveness'], f['tempo'], f['speechiness'], f['acousticness'],\
# 						f['instrumentalness'], f['danceability'], f['duration_ms'], f['valence']]
# 						totalFeatures.append(X)
# 						totalYears.append(year)
# 					except:
# 						print("ERROR: Invalid URI - " + uris[i])
# 		self.model.fit(np.array(totalFeatures), np.array(totalYears))

# 		# DEBUG - PRINT AVGS
# 		# avgs = []
# 		# for i in range(9):
# 		# 	avgs.append(float(sum([x[i] for x in totalFeatures]))/float(len(totalFeatures)))
# 		# print(avgs)

# 	def match_years(self, years):
# 		self.train_model()

# 		testFeatures = []
# 		testYears = []
# 		for year in years:
# 			uris = song_test_uris.get(year, [])
# 			if uris:
# 				features = sp.audio_features(uris)
# 				for f in features:
# 					X = [f['energy'], f['liveness'], f['tempo'], f['speechiness'], f['acousticness'],\
# 					f['instrumentalness'], f['danceability'], f['duration_ms'], f['valence']]
# 					# print(model.decision_function([X]))
# 					# print(model.predict([X]))
# 					# print(year)
# 					# print('')
# 					testFeatures.append(X)
# 					testYears.append(year)
# 		preds = self.model.predict(testFeatures)
# 		total_right = 0
# 		for i in range(len(testYears)):
# 			if testYears[i] == preds[i]:
# 				total_right += 1
# 		accuracy_score = float(total_right)/float(len(testYears))

# 		return preds, testYears, accuracy_score
	
# def perform_year_prediction():
# 	yf = YearFinder()
# 	preds, testYears, accuracy_score = yf.match_years(range(2010, 2017))

# 	print(testYears)
# 	print(preds)
# 	print(accuracy_score)




	

