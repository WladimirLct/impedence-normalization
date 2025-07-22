# File: callbacks/analysis_page/graph.py

import time
import re
import uuid

from dash import dcc, callback, Output, Input, State, no_update, ctx, html
import dash_bootstrap_components as dbc 

from utils.graph import plot_graph
from config.cache import cache
from config.options import DEFAULT_FREQ

@callback(
    Output('groups_enabled', 'data'),
    #
    Input('url-p3', 'pathname'),
    State('wells', 'data'),
    State('groups_enabled', 'data'),
    prevent_initial_call=True,
)
def plot_on_page_load(_, wells_data, groups_enabled):
    if groups_enabled is None:
        return False
     
    return no_update

@callback(
    Output('main-graph', 'figure', allow_duplicate=True),
    #
    Input('groups_enabled', 'data'),
    Input('selected_normalization', 'data'),
    Input('selected_frequency', 'data'),
    Input('std_bar_frequency', 'data'),
    State('single_well_groups_data', 'data'),
    State('wells', 'data'),
    State('well_groups_data', 'data'),
    prevent_initial_call=True,
)
def update_main_graph(groups_enabled, selected_norm, selected_freq, std_freq, single_well_groups_data, wells_data, well_groups_data):
    raw_data = cache["data"]

    t0 = cache.get("t0")
    if t0 is None:
        t0 = raw_data["date"].min()

    raw_data = raw_data[raw_data["date"] > t0]

    if raw_data is None:
        return {'data': [], 'layout': {'title': 'Aucune donnée'}}

    # Filter data by frequency if selected
    filtered_data = raw_data.copy()
    if selected_freq is None or 'frequency' not in filtered_data.columns:
        selected_freq = DEFAULT_FREQ
    filtered_data = filtered_data[filtered_data['frequency'] == selected_freq]
    
    # Use selected normalization parameter, default to 'absz'
    y_param = selected_norm if selected_norm else 'absz'
    
    # Determine which groups to use
    if not groups_enabled:
        groups_to_plot = single_well_groups_data
    else:
        groups_to_plot = well_groups_data

    return plot_graph(filtered_data, groups_to_plot, y_param=y_param, x_param='date', std_bar_freq=std_freq)