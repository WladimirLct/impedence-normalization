# File: callbacks/p1.py

import time
import os
import re

from dash import dcc, callback, Output, Input, State, no_update

from config.cache import cache
from utils.loader import list_folders

@callback(
    # Inside the return
    Output('folder-btn', 'className', allow_duplicate=True),
    Output('spinner-div', 'className', allow_duplicate=True),
    Output('p1-layout', 'className'),
    Output('url-p1', 'pathname', allow_duplicate=True),
    # Func parameters
    Input('folder-btn', 'n_clicks'),
    State('folder-path', 'value'),
    State('folder-btn', 'className'),
    State('folder-btn', 'children'),
    State('spinner-div', 'className'),
    State('p1-layout', 'className'),
    # Necessary to have duplicate outputs
    prevent_initial_call=True
)
def hide_button(n_clicks, folder_path, btn_classes, btn_text, spinner_classes, p1_classes):
    if n_clicks:
        new_url = no_update
        if folder_path is not None:
            if os.path.exists(folder_path):
                if btn_text == 'Charger':
                    btn_classes += " d-none"
                    spinner_classes = re.sub("d-none", "", spinner_classes)
                elif btn_text == 'Analyser':
                    p1_classes += ' d-none'
                    new_url = '/loading'  # Change URL to /loading

    return btn_classes, spinner_classes, p1_classes, new_url


@callback(
    Output('loading-output', 'children'),
    Output('folder-btn', 'className'),
    Output('folder-btn', 'children'),
    Output('spinner-div', 'className'),
    #
    Input('folder-btn', 'className'),
    State('folder-path', 'value'),
    State('spinner-div', 'className'),
    prevent_initial_call=True,
)
def load_file(btn_classes, folder_path, spinner_classes):
    if ("d-none" not in btn_classes):
        return "", btn_classes, 'Charger', spinner_classes
    
    folders_df = list_folders(folder_path)
    cache["paths"] = folders_df
    time.sleep(1)

    btn_classes = re.sub("d-none", "", btn_classes)
    spinner_classes += " d-none"

    return "", btn_classes, 'Analyser', spinner_classes

@callback(
    Output('data-table', 'data'),
    Output('data-table', 'columns'),
    Output('data-info', 'children'),
    #
    Input('folder-btn', 'children'),
    prevent_initial_call=True,
)
def show_data_table(btn_text):
    if (btn_text != 'Analyser'):
        return None, None, None
    
    df_copy = cache["paths"].copy()
    df_copy["base_dir"] = df_copy["base_dir"].apply(lambda x: x.split("/")[-1])

    res = [df_copy.to_dict('records'), [{"name": i, "id": i} for i in df_copy.columns], None]
    res[2] = f"{df_copy['well'].nunique()} puits trouvés. {df_copy['well'].count()} fichiers au total."

    return res[0], res[1], res[2]