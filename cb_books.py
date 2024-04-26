# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.sparse import hstack
from sklearn.feature_extraction import FeatureHasher
from joblib import dump, load
import os


class BookRecommender:
    def __init__(self, data_path='book_data.csv', sim_matrix_path='book_cosine_sim.joblib'):
        self.data_path = data_path
        self.sim_matrix_path = sim_matrix_path
        self.data = None
        self.features = None
        self.cosine_sim = None
        self.load_data()
        self.prepare_features()

    def load_data(self):
        self.data = pd.read_csv(self.data_path, usecols=['title', 'authors', 'categories', 'description',
                                                         'published_year', 'average_rating', 'num_pages'])

    def prepare_features(self):
        if os.path.exists(self.sim_matrix_path):
            self.cosine_sim = load(self.sim_matrix_path)
        else:
            # 文本描述处理
            tfidf = TfidfVectorizer(stop_words='english', min_df=5, max_df=0.9, max_features=10000)
            tfidf_matrix = tfidf.fit_transform(self.data['description'].fillna(''))

            # 多值类别特征处理
            self.data['authors'] = self.data['authors'].fillna('').map(lambda x: x.split(';'))
            fh1 = FeatureHasher(n_features=100, input_type='string')
            authors_matrix = fh1.fit_transform(self.data['authors'])

            self.data['categories'] = self.data['categories'].fillna('').map(lambda x: x.split(','))
            categories_matrix = fh1.fit_transform(self.data['categories'])

            # 数值型特征处理
            scaler = StandardScaler()
            average_published_year = self.data['published_year'].mean()
            published_year_scaled = scaler.fit_transform(self.data[['published_year']].fillna(average_published_year))

            scaler = StandardScaler()
            average_rating_mean = self.data['average_rating'].mean()
            rating_scaled = scaler.fit_transform(self.data[['average_rating']].fillna(average_rating_mean))

            scaler = StandardScaler()
            average_num_pages = self.data['num_pages'].mean()
            num_pages_scaled = scaler.fit_transform(self.data[['num_pages']].fillna(average_num_pages))

            # 合并所有特征矩阵
            self.features = hstack([tfidf_matrix, authors_matrix, categories_matrix, published_year_scaled,
                                    rating_scaled, num_pages_scaled])
            self.cosine_sim = cosine_similarity(self.features)
            dump(self.cosine_sim, self.sim_matrix_path)

    def recommend(self, liked_book, disliked_books=[], n=5):
        recommendations = []
        for book in liked_book:
            if book not in self.data['title'].values:
                return []
            # 对不喜欢的电影调整相似度
            for title in disliked_books:
                idx = self.data.index[self.data['title'] == title].tolist()
                for i in idx:
                    self.cosine_sim[i, :] = self.cosine_sim[:, i] = 0

            # 推荐逻辑
            idx = self.data[self.data['title'] == book].index[0]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            top_indices = [i[0] for i in sim_scores[1:n+1]]
            recommendations.extend(self.data['title'].iloc[top_indices].tolist())
        return recommendations


if __name__ == "__main__":
    recommender = BookRecommender()
    print(recommender.recommend(['If Tomorrow Comes']))
