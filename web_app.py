# -*- coding: utf-8 -*-

import dash, json
from dash import html, dcc, callback, Input, Output, State
from dash.dependencies import ALL, MATCH
from dash.exceptions import PreventUpdate
import pandas as pd
from cb_movies import MovieRecommender
from cb_books import BookRecommender
from cf_games import RecommenderSystem
from cross_recommend import CrossRecommender

'''
title = pd.read_csv('netflix_titles.csv', usecols=['title'])
database = list(title['title'].unique())

recommender = BookRecommender()
'''
def call_app(data_base):
    app = dash.Dash(__name__)

    # Application layout
    app.layout = html.Div([
        dcc.Store(id='memory'),  # Store for keeping state
        dcc.Store(id='disliked-memory', data=[]),  # 用于存储不喜欢的电影
        dcc.Dropdown(
            id='autocomplete-dropdown',
            options=[{'label': item, 'value': item} for item in database],
            placeholder="输入搜索内容...",
            search_value='',
            clearable=True
        ),
        html.Button('Enter', id='enter-button', n_clicks=0),
        html.Button('Clear', id='clear-button', n_clicks=0),
        html.Div(id='output-container'),
        html.Button('Recommend', id='recommend-button', n_clicks=0),
        html.Div(id='recommendation-output'),
    ])


    # 生成推荐列表和不喜欢按钮，每个按钮标记其对应的电影名称
    def generate_recommendation_layout(recommendations):
        return html.Div(className='container', children=[
            html.Div(className='list-item', children=[
                html.Span(item, className='movie-name'),
                html.Button('Dislike', className='button', id={'type': 'dislike-btn', 'movie': item}, n_clicks=0)
            ]) for item in recommendations
        ])



    @app.callback(
        [Output('recommendation-output', 'children', allow_duplicate=True),
         Output('disliked-memory', 'data', allow_duplicate=True)],
        [Input('recommend-button', 'n_clicks'), Input({'type': 'dislike-btn', 'movie': ALL}, 'n_clicks')],
        [State('memory', 'data'),
         State('disliked-memory', 'data')],
        prevent_initial_call=True
    )
    def update_recommendations(recommend_clicks, dislike_clicks, liked_data, disliked_data):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise PreventUpdate

        triggered_id = ctx.triggered[0]['prop_id']
        if 'recommend-button' in triggered_id:
            if not liked_data:
                return "请输入一些有效的数据。", dash.no_update
            recommendations = recommender.recommend(liked_data, disliked=disliked_data)
            return generate_recommendation_layout(recommendations), dash.no_update
        else:
            # 直接从按钮ID获取不喜欢的电影名称
            disliked_movie = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])['movie']
            if disliked_movie not in disliked_data:
                disliked_data.append(disliked_movie)

            # 更新喜欢的列表，移除不喜欢的电影
            new_liked_data = [item for item in liked_data if item != disliked_movie]
            new_recommendations = recommender.recommend(new_liked_data, disliked=disliked_data)
            return generate_recommendation_layout(new_recommendations), disliked_data


    @app.callback(
        [Output('memory', 'data', allow_duplicate=True),
         Output('output-container', 'children', allow_duplicate=True)],
        [Input('enter-button', 'n_clicks')],
        [State('autocomplete-dropdown', 'value'),
         State('memory', 'data')],
        prevent_initial_call=True
    )
    def add_to_memory(n_clicks, value, memory_data):
        # 判断是否有点击事件
        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate

        # 如果内存中没有数据，则初始化为空列表
        if memory_data is None:
            memory_data = []

        # 将选择的条目添加到内存数据列表中
        if value and value not in memory_data:
            memory_data.append(value)

        # 更新显示的用户输入
        children = html.Ul([html.Li(item) for item in memory_data])

        return memory_data, children


    @callback(
        Output('autocomplete-dropdown', 'value', allow_duplicate=True),
        Input('autocomplete-dropdown', 'options'),
        State('autocomplete-dropdown', 'value'),
        prevent_initial_call=True
    )
    def clear_input(options, value):
        # 当用户选择了一个选项后，清空输入框
        return ''

    @app.callback(
        [Output('autocomplete-dropdown', 'value', allow_duplicate=True),
         Output('memory', 'data', allow_duplicate=True),
         Output('disliked-memory', 'data', allow_duplicate=True),
         Output('output-container', 'children', allow_duplicate=True),
         Output('recommendation-output', 'children', allow_duplicate=True)],
        [Input('clear-button', 'n_clicks')],
        prevent_initial_call=True
    )
    def clear_data(n_clicks):
        if n_clicks is None or n_clicks == 0:
            raise PreventUpdate

        # 清除下拉菜单的值
        new_value = ''

        # 清空 memory 和 disliked-memory 的存储数据
        new_memory_data = []
        new_disliked_memory_data = []

        # 清除显示的输出
        new_output_container = None
        new_recommendation_output = None

        return new_value, new_memory_data, new_disliked_memory_data, new_output_container, new_recommendation_output

    return app


if __name__ == '__main__':
    domain = ''
    while domain not in ['1', '2', '3', '4']:
        domain = input('Select one domain (1, 2, 3, 4):\n'
                       '1. movies and tvs.\n'
                       '2. books.\n'
                       '3. games.\n'
                       '4. cross-domain.\n')

    if domain == '1':
        title = pd.read_csv('netflix_titles.csv', usecols=['title'])
        recommender = MovieRecommender()
    elif domain == '2':
        title = pd.read_csv('book_data.csv', usecols=['title'])
        recommender = BookRecommender()
    elif domain == '3':
        title = pd.read_csv('games.csv', usecols=['title'])
        recommender = RecommenderSystem()
    elif domain == '4':
        title = pd.read_csv('all_items_description.csv', usecols=['title'])
        recommender = CrossRecommender()

    database = list(title['title'].unique())
    app = call_app(data_base=database)
    app.run_server(debug=False)
