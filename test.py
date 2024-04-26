# -*- coding: utf-8 -*-

import pandas as pd
'''
data = pd.read_csv('netflix_titles.csv',
                   usecols=['type', 'title', 'director', 'cast', 'country', 'release_year', 'rating',
                            'listed_in', 'description'])

data['director'] = data['director'].fillna('').map(lambda x: x.split(', '))

data = pd.read_csv('netflix_titles.csv',
                   usecols=['title'])
s = list(data['title'].unique())


recommendations_df = pd.read_csv('recommendations.csv')
unique_user_ids = recommendations_df['user_id'].astype('category').cat.categories
print(recommendations_df.isnull().sum())
print("数据帧的行数: ", len(recommendations_df))
print("独特 user_id 的数量: ", len(unique_user_ids))
'''
'''
from scipy.sparse import coo_matrix

# Load data
recommendations_df = pd.read_csv('recommendations.csv')
print(recommendations_df.dtypes)
print(recommendations_df.isnull().sum())

# Create continuous index mappings for users and items
user_id_mapping = {id: i for i, id in enumerate(recommendations_df['user_id'].unique())}
item_id_mapping = {id: i for i, id in enumerate(recommendations_df['app_id'].unique())}

# Map the original user and item IDs to the new continuous indices
recommendations_df['user_id_idx'] = recommendations_df['user_id'].map(user_id_mapping)
recommendations_df['app_id_idx'] = recommendations_df['app_id'].map(item_id_mapping)

# Create a sparse matrix
user_game_matrix = coo_matrix(
    (recommendations_df['hours'], (recommendations_df['user_id_idx'], recommendations_df['app_id_idx'])),
    shape=(len(user_id_mapping), len(item_id_mapping))
)
user_game_matrix = user_game_matrix.tocsr()

from sklearn.neighbors import NearestNeighbors
from joblib import dump, load

model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(user_game_matrix)
dump(model, 'model/games_knn_model.joblib')
# model = load('model/games_knn_model.joblib')
user_id = 5
distances, indices = model_knn.kneighbors(user_game_matrix.getrow(user_id), n_neighbors=n_neighbors)
'''

# Now user_game_matrix is built with continuous indices and should not have IndexError issues

book = pd.read_csv('book_data.csv', usecols=['title', 'authors', 'categories', 'description',
                                                         'published_year', 'average_rating', 'num_pages'])
print(book.dtypes)
print(len(book['categories'].unique().tolist()))
