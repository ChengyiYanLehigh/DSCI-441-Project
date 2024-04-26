# ML-Project
 
## Summary
This project aims to build a entertainment recommendation system for people. 
Uses can input their favorite books, movies, games, etc., 
and the system will suggest other titles that may interested them.\
The system will use Content-Based Filtering and User-Based Filtering to do the task. 
The system can also do the in-domain and cross-domain recommendations.
The model will receive multiple features from datasets, 
which include information about the entertainment items, 
and then use machine learning algorithms to uncover potential interests. 
Also, there will be a feedback mechanism which will allow users to dislike items, 
helping users get better experiences.

## Data Source
https://www.kaggle.com/datasets/shivamb/netflix-shows/data
https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam/data
https://www.kaggle.com/datasets/anilcogalan/steam-recommender-system
https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset

## Use
1. download all the datasets from the websites in the datasource. Unzip them and put all the files in the directory.
2. run prepare data if you want to try cross-domain recommendations.
3. run web_app.py to run the interactive recommendation systems. 
4. run cb_movies.py to do the content-based filtering on movies dataset.
5. run ub_games.py to do the user-based filtering on games dataset. 
6. run cb_books.py to do the content-based filtering on books dataset. 
7. run cross_recommend.py to do the cross-domain recommendations. 