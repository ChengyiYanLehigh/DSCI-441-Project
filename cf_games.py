import os
import random
import pandas as pd
import numpy as np
from collections import Counter
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score
from joblib import dump, load


class RecommenderSystem:
    def __init__(self, data_path='recommendations.csv'):
        self.data_path = data_path
        self.load_data()
        self.check_model()

    def load_data(self):
        self.df = pd.read_csv(self.data_path, usecols=['app_id', 'is_recommended', 'hours', 'user_id'])
        self.user_id_mapping = {id: i for i, id in enumerate(self.df['user_id'].unique())}
        self.item_id_mapping = {id: i for i, id in enumerate(self.df['app_id'].unique())}
        self.index_to_user_id = {v: k for k, v in self.user_id_mapping.items()}

        # Sparse matrix for user-game interactions
        self.user_game_matrix = coo_matrix(
            (self.df['hours'],
             (self.df['user_id'].map(self.user_id_mapping), self.df['app_id'].map(self.item_id_mapping))),
            shape=(len(self.user_id_mapping), len(self.item_id_mapping))
        ).tocsr()

        self.title_to_id = pd.read_csv('games.csv', usecols=['app_id', 'title']).set_index('title')['app_id'].to_dict()
        self.id_to_title = pd.read_csv('games.csv', usecols=['app_id', 'title']).set_index('app_id')['title'].to_dict()

    def check_model(self):
        if not os.path.exists('model/games_knn_model.joblib'):
            self.train_knn_model()
        self.model = load('model/games_knn_model.joblib')

    def train_knn_model(self):
        model = NearestNeighbors(metric='cosine', algorithm='brute')
        model.fit(self.user_game_matrix)
        dump(model, 'model/games_knn_model.joblib')

    def recommend_games(self, user_id, n=5):
        user_index = self.user_id_mapping[user_id]
        distances, indices = self.model.kneighbors(self.user_game_matrix.getrow(user_index), n_neighbors=6)
        similar_users = [self.index_to_user_id[i] for i in indices.flatten()[1:]]

        similar_games = []
        for user in similar_users:
            user_games = self.df[(self.df['user_id'] == user) & self.df['is_recommended']]['app_id'].unique()
            similar_games.extend(user_games)

        game_counts = Counter(similar_games)
        return [game for game, count in game_counts.most_common(n)]

    def get_user_owned_games(self, user_id):
        user_games = self.df[(self.df['user_id'] == user_id) & self.df['is_recommended']]['app_id'].unique().tolist()
        return user_games

    def evaluate_dataset(self, user_list, n_recommendations=5):
        results = {'precision': [], 'recall': [], 'f1_score': []}
        unique_game_ids = self.df['app_id'].unique()
        for user_id in user_list:
            recommended_games = self.recommend_games(user_id, n=n_recommendations)
            actual_games = self.get_user_owned_games(user_id)

            if len(recommended_games) == 0 or len(actual_games) == 0:
                continue

            actual_labels = [1 if game in actual_games else 0 for game in unique_game_ids]
            recommended_labels = [1 if game in recommended_games else 0 for game in unique_game_ids]

            precision = precision_score(actual_labels, recommended_labels)
            recall = recall_score(actual_labels, recommended_labels)
            f1 = f1_score(actual_labels, recommended_labels)

            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1_score'].append(f1)

        average_results = {
            'average_precision': sum(results['precision']) / len(results['precision']),
            'average_recall': sum(results['recall']) / len(results['recall']),
            'average_f1_score': sum(results['f1_score']) / len(results['f1_score'])
        }
        return average_results

    def recommend(self, liked, disliked=[], n=5):
        liked_ids = []
        disliked_ids = []
        for game in liked:
            liked_ids.append(self.title_to_id[game])
        for game in disliked:
            disliked_ids.append(self.title_to_id[game])
        # Create a user game vector, initialized to 0
        user_game_vector = np.zeros(self.user_game_matrix.shape[1])

        # Increase weight to liked games
        for game in liked_ids:
            if game in self.item_id_mapping:
                user_game_vector[self.item_id_mapping[game]] += 50

        # Decrease weight to disliked games
        for game in disliked_ids:
            if game in self.item_id_mapping:
                user_game_vector[self.item_id_mapping[game]] -= 50

        # Use the model to find similar users and recommendations
        distances, indices = self.model.kneighbors([user_game_vector], n_neighbors=20)
        similar_games = []
        for i in indices.flatten():
            similar_games.extend(self.df[self.df['user_id'] == self.index_to_user_id[i]]['app_id'].unique())

        # Calculate the frequency of recommended games and return the n most common ones
        game_counts = Counter(similar_games)
        recommended_games = [game for game, count in game_counts.most_common(n)]

        # Filter out games that are already in the like or dislike list
        final_recommendations = [self.id_to_title[game] for game in recommended_games if
                                 game not in liked_ids and game not in disliked_ids]
        if len(final_recommendations) < n:
            additional_games = np.random.choice(list(self.title_to_id.keys()), n - len(final_recommendations), replace=False)
            final_recommendations.extend(additional_games)
        return final_recommendations


if __name__ == "__main__":
    recommender = RecommenderSystem()
    '''
    # Warning: takes a little long time!
    user_ids = list(recommender.index_to_user_id.values())
    random.seed(123)
    random.shuffle(user_ids)
    print(recommender.evaluate_dataset(user_ids[:10]))
    '''
    print(recommender.recommend(["Monaco: What's Yours Is Mine", "Osmos"]))
