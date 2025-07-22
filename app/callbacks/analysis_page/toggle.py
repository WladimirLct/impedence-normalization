# File: callbacks/analysis_page/toggle.py

from dash import dcc, callback, Output, Input, State, no_update, ctx , html
import dash_bootstrap_components as dbc 

from config.cache import cache

@callback(
    Output('collapse_graph', 'is_open'),
    Output('collapse_graph_state', 'children'),
    #
    Input('collapse_graph_btn', 'n_clicks'),
)
def toggle_graph(n_clicks):
    options = ["Afficher", "Masquer"]
    value = n_clicks % 2
    return value, options[value]


@callback(
    Output('collapse_stats', 'is_open'),
    Output('collapse_stats_state', 'children'),
    #
    Input('collapse_stats_btn', 'n_clicks'),
)
def toggle_graph(n_clicks):
    options = ["Afficher", "Masquer"]
    value = n_clicks % 2
    return value, options[value]