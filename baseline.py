import os
import collections

TOP_ARTIST_PATH = "artist_data/top_artists_"
# type(year) -> int
def getPath(year):
	return TOP_ARTIST_PATH + str(year) + ".txt"

def loadArtistDict():
	artist_pop_score = collections.defaultdict(int)
	# Take popularity of artist over past 4 years 2006-09
	for year in range(2006,2010):
		f = open(getPath(year))
		for rank, name in enumerate(f):
			name = name.strip()
			if name:
				# Ex. 1 -> 100, 2 --> 99, ...
				rank = abs(rank-101)
				artist_pop_score[name] += rank
	return artist_pop_score

def loadTestSongs():
	path = "MillionSongSubset/AdditionalFiles/subset_tracks_per_year.txt"
	f = open(path)
	songs = set()
	for line in f:
		line = line.strip()
		if line[:4] == '2010':
			line = line[32:]
			sep_idx = line.find('<SEP>')
			artist = line[:sep_idx]
			song_title = line[sep_idx+5:]
			songs.add((artist, song_title))
	return songs

def main():
	artist_pop = loadArtistDict()
	songs = loadTestSongs()

	songs_with_rank = [(s[0], s[1], artist_pop[s[0]]) for s in songs]
	ranked_songs = sorted(songs_with_rank, key=lambda x: x[2], reverse=True)
	print(ranked_songs)


	

if __name__ == '__main__':
    main()
