# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump, load
import os


class CrossRecommender:
    def __init__(self, data_path='all_items_description.csv', sim_matrix_path='all_cosine_sim.joblib'):
        self.data_path = data_path
        self.sim_matrix_path = sim_matrix_path
        self.data = None
        self.features = None
        self.cosine_sim = None
        self.load_data()
        self.prepare_features()

    def load_data(self):
        self.data = pd.read_csv(self.data_path, usecols=['title', 'description']).dropna()

    def prepare_features(self):
        if os.path.exists(self.sim_matrix_path):
            self.cosine_sim = load(self.sim_matrix_path)
        else:
            # Process text features
            tfidf = TfidfVectorizer(stop_words='english', min_df=5, max_df=0.9, max_features=10000)
            tfidf_matrix = tfidf.fit_transform(self.data['description'])

            self.features = tfidf_matrix
            self.cosine_sim = cosine_similarity(self.features)
            dump(self.cosine_sim, self.sim_matrix_path)

    def recommend(self, liked, disliked=[], n=5):
        recommendations = []
        for movie in liked:
            if movie not in self.data['title'].values:
                return []
            # Adjust the similarity for disliked books
            for title in disliked:
                idx = self.data.index[self.data['title'] == title].tolist()
                for i in idx:
                    self.cosine_sim[i, :] = self.cosine_sim[:, i] = 0

            # Recommend
            idx = self.data[self.data['title'] == movie].index[0]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_indices = [i[0] for i in sim_scores[1:n+1]]
            recommendations.extend(self.data['title'].iloc[top_indices].tolist())
        return recommendations


if __name__ == "__main__":
    recommender = CrossRecommender()
    print(recommender.recommend(['Game: Half-Life', 'Book: Chosen But Free']))
