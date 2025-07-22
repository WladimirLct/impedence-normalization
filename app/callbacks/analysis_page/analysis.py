# File: callbacks/analysis_page/analysis.py

from datetime import datetime
from dash import dcc, callback, Output, Input, State, no_update, ctx , html
import dash_bootstrap_components as dbc 
import plotly.graph_objects as go


from utils.groups import get_list_wells, create_single_well_groups
from config.cache import cache

import zipfile
import io
import os

@callback(
    Output('url-p3', 'pathname', allow_duplicate=True),
    #
    Input('url-p3', 'pathname'),
    State('wells', 'data'),
    State('well_groups_data', 'data'),
    prevent_initial_call=True,
)
def on_analysis_page_load(pathname, wells_data, well_groups_data):
    # Allow navigation to the root page without redirection
    if pathname == '/':
        return no_update

    # Redirect to '/' if cache conditions are not met
    if len(cache["paths"]) < 1 or len(cache["data"]) < 1:
        return '/'

    return '/analysis'
    
@callback(
    Output('download-component', 'data'),
    Input('download-figures-info', 'n_clicks'),
    State('main-graph', 'figure'),
    State('stats-graph', 'figure'),
    prevent_initial_call=True
)
def download_files(n_clicks, main_graph_figure, stats_graph_figure):
    print("Download request received")
    if not n_clicks:
        print("No clicks detected, skipping download")
        return no_update
        
    if n_clicks:
        print("Downloading figures and info...")
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            # Add your files from tmp directory
            if os.path.exists('tmp/graph_info.txt'):
                zip_file.write('tmp/graph_info.txt', 'graph_info.txt')
            if os.path.exists('tmp/stats_info.txt'):
                zip_file.write('tmp/stats_info.txt', 'stats_info.txt')

            # Add figures (transform them to pdf)
            if main_graph_figure:
                # Create figure object from the figure data
                fig = go.Figure(main_graph_figure)
                fig.write_image('tmp/main_graph.pdf', width=1200, height=500)
                zip_file.write('tmp/main_graph.pdf', 'main_graph.pdf')
                
            if stats_graph_figure:
                fig = go.Figure(stats_graph_figure)
                fig.write_image('tmp/stats_graph.pdf', width=1200, height=500)
                zip_file.write('tmp/stats_graph.pdf', 'stats_graph.pdf')
        
        zip_buffer.seek(0)
        return dcc.send_bytes(zip_buffer.getvalue(), "analysis_results.zip")
    
    return no_update
    
@callback(
    Output('info-analysis-1', 'children'),
    Output('info-analysis-2', 'children'),
    Output('info-analysis-3', 'children'),
    Output('info-analysis-4', 'children'),
    Output('info-analysis-5', 'children'),
    Output('select_t0', 'max'),
    #
    Input('url-p3', 'pathname'),
)
def show_exp_info(url_path):
    if len(cache["paths"]) < 1 and len(cache["wells"]) < 1:
        return no_update, no_update, no_update, no_update, no_update

    i1 = cache["paths"].iloc[0]["base_dir"].split("/")[-1]
    i2 = len(cache["wells"])
    i3 = len(cache["paths"])
    start = cache["data"]['date'].min()
    i4 = str(start).replace("T", " ")
    i5 = round((cache["data"]['date'].max() - start).total_seconds() / 3600, 1)

    return i1, i2, i3, i4, i5, i5

@callback(
    Output('select_wells', 'options'),
    Output('wells', 'data'),
    Output('single_well_groups_data', 'data'),
    #
    Input('url-p3', 'pathname'),
    State('wells', 'data'),
    State('single_well_groups_data', 'data'),
    State('list_well_not_in_group', 'data'),
)
## Callback to fill the wells dropdown
def fill_wells_dropdown(_, wells_data, single_well_groups_data, list_well_not_in_group):
    if wells_data is None or len(wells_data) < 1:
        wells_data = get_list_wells()

    if not wells_data:
        return [], no_update, no_update

    single_well_groups_data = create_single_well_groups(wells_data)

    if list_well_not_in_group is None :
        options = [{'label': well, 'value': well} for well in wells_data] 
    else:
        options = [{'label': well, 'value': well} for well in list_well_not_in_group] 

    return options, wells_data, single_well_groups_data