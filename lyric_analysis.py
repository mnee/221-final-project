def get_repetition_score(self, responses):
		vect = TfidfVectorizer(min_df=1)
		tfidf = vect.fit_transform(responses)
		sim_arr = (tfidf*tfidf.T).A

		num_comps = 0.0 # C(len(responses),2)
		sim_score = 0.0
		for i in range(0,len(responses)-1):
			for j in range(i+1,len(responses)):
				sim_score += sim_arr[i][j]
				num_comps += 1.0
		return sim_score/(num_comps if num_comps > 0 else 1)