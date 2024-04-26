# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from scipy.sparse import hstack
from sklearn.feature_extraction import FeatureHasher
from joblib import dump, load
import os


class MovieRecommender:
    def __init__(self, data_path='netflix_titles.csv', sim_matrix_path='cosine_sim.joblib'):
        self.data_path = data_path
        self.sim_matrix_path = sim_matrix_path
        self.data = None
        self.features = None
        self.cosine_sim = None
        self.load_data()
        self.prepare_features()

    def load_data(self):
        self.data = pd.read_csv(self.data_path, usecols=['type', 'title', 'director', 'cast', 'country', 'release_year',
                                                         'rating', 'listed_in', 'description'])

    def prepare_features(self):
        if os.path.exists(self.sim_matrix_path):
            self.cosine_sim = load(self.sim_matrix_path)
        else:
            # 文本描述处理
            tfidf = TfidfVectorizer(stop_words='english', min_df=5, max_df=0.9, max_features=10000)
            tfidf_matrix = tfidf.fit_transform(self.data['description'])

            # 多值类别特征处理
            self.data['director'] = self.data['director'].fillna('').map(lambda x: x.split(', '))
            fh1 = FeatureHasher(n_features=100, input_type='string')
            director_matrix = fh1.fit_transform(self.data['director'])

            self.data['cast'] = self.data['cast'].fillna('').map(lambda x: x.split(', '))
            cast_matrix = fh1.fit_transform(self.data['cast'])

            self.data['country'] = self.data['country'].fillna('').map(lambda x: x.split(', '))
            fh2 = FeatureHasher(n_features=50, input_type='string')
            country_matrix = fh2.fit_transform(self.data['country'])

            self.data['listed_in'] = self.data['listed_in'].fillna('').map(lambda x: x.split(', '))
            fh3 = FeatureHasher(n_features=10, input_type='string')
            listed_in_matrix = fh3.fit_transform(self.data['listed_in'])

            # 单值类别特征处理
            ohe = OneHotEncoder()
            rating_matrix = ohe.fit_transform(self.data[['rating']].fillna('Unknown'))

            # 合并所有特征矩阵
            self.features = hstack([tfidf_matrix, director_matrix, cast_matrix, listed_in_matrix, country_matrix, rating_matrix])
            self.cosine_sim = cosine_similarity(self.features)
            dump(self.cosine_sim, self.sim_matrix_path)

    def recommend(self, liked_movie, disliked_movies=[], n=5):
        recommendations = []
        for movie in liked_movie:
            if movie not in self.data['title'].values:
                return []
            # 对不喜欢的电影调整相似度
            for title in disliked_movies:
                idx = self.data.index[self.data['title'] == title].tolist()
                for i in idx:
                    self.cosine_sim[i, :] = self.cosine_sim[:, i] = 0

            # 推荐逻辑
            idx = self.data[self.data['title'] == movie].index[0]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_indices = [i[0] for i in sim_scores[1:n+1]]
            recommendations.extend(self.data['title'].iloc[top_indices].tolist())
        return recommendations


if __name__ == "__main__":
    recommender = MovieRecommender()
    print(recommender.recommend(['Blood & Water', 'Sin Senos sí Hay Paraíso']))
