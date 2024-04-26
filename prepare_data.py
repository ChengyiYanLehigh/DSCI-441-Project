# -*- coding: utf-8 -*-
import pandas as pd

game = pd.read_csv('steam_recommender.csv', usecols=['name', 'short_description'])
book = pd.read_csv('book_data.csv', usecols=['title', 'description'])
movie = pd.read_csv('netflix_titles.csv', usecols=['title', 'description'])

game.rename(columns={'name': 'title', 'short_description': 'description'}, inplace=True)

game['title'] = 'Game: ' + game['title']
book['title'] = 'Book: ' + book['title']
movie['title'] = 'Movie/TV: ' + movie['title']

combined_df = pd.concat([game, book, movie], axis=0, ignore_index=True)
combined_df.to_csv('all_items_description.csv')

print(combined_df.head())
