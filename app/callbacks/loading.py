# File: callbacks/loading.py

import os
import re
import time
import pandas as pd

from dash import dcc, callback, Output, Input, State, no_update

from config.cache import cache
from utils.loader import load_files, loading_data, loading_data_lock, update_well_list
from utils.normalizer import start_normalization_thread, get_normalization_progress

def get_base_info():
    return {
        'started': False,
        'finished': False,
    }

@callback(
    Output('url-p2', 'pathname', allow_duplicate=True),
    Output('load_info', 'data', allow_duplicate=True),
    Output('norm_info', 'data', allow_duplicate=True),
    #
    Input('url-p2', 'pathname'),
    State('load_info', 'data'),
    State('norm_info', 'data'),
    prevent_initial_call='initial_duplicate'
)
def on_page_load(url_path, load_info, norm_info):
    if url_path != '/loading':
        return no_update, no_update, no_update

    if (len(cache["paths"]) < 1):
        return '/', get_base_info(), get_base_info()
    
    load_i = load_info
    norm_i = norm_info

    if load_i is None:
        load_i = get_base_info()

    save_path = "./normalizations/" + cache["paths"].iloc[0]["base_dir"].split("/")[-1] + ".csv"

    if (os.path.exists(save_path)):
        cache["data"] = pd.read_csv(save_path)
        cache["data"]["date"] = pd.to_datetime(cache["data"]["date"])
        update_well_list(cache["data"])
        return '/analysis', no_update, no_update

    num_workers = max(2, os.cpu_count() - 2)

    if not load_i["started"]:
        load_i, norm_i = get_base_info(), get_base_info()
        load_i["started"] = True
        # use the cpu count to determine the number of workers
        load_files(num_workers = num_workers)

    if load_i["finished"]:
        if (len(loading_data) != len(cache["paths"])):
            load_i, norm_i = get_base_info(), get_base_info()
            load_i["started"] = True
            load_files(num_workers = num_workers)

    return no_update, load_i, norm_i
    
@callback(
    Output('file_load_idx', 'children'),
    Output('load_info', 'data'),
    #
    Input('load_interval', 'n_intervals'),
    State('load_info', 'data'),
)
def poll_loading_progress(_, load_info):
    if load_info is None:
        return no_update, no_update
    
    load_i = load_info

    if not load_i["started"]:
        print("Loading not started")
        return no_update, no_update

    with loading_data_lock:
        loaded = len(loading_data)
        total = len(cache['paths'])

        current_progress = f"{loaded}/{total}"
        
        if not load_i["finished"]:
            if loaded == total and total > 0:  # Also check total > 0
                load_i['finished'] = True
                return current_progress, load_i

    if load_i["started"] and not load_i["finished"]:
        return current_progress, no_update
    
    return current_progress, no_update

@callback(
    Output('norm_info', 'data'),
    #
    Input('load_info', 'data'),
    State('norm_info', 'data'),
    prevent_initial_call='initial_duplicate'
)
def normalize(load_info, norm_info):
    load_i = load_info
    norm_i = norm_info

    if norm_i is None:
        norm_i = get_base_info()

    if not norm_i["started"] and load_i["finished"]:
        with loading_data_lock:
            loaded = len(loading_data)
            total = len(cache["paths"])
            
            if loaded == total and loaded > 0:
                save_path = "./normalizations/" + cache["paths"].iloc[0]["base_dir"].split("/")[-1] + ".csv"
                norm_i = get_base_info()
                norm_i["started"] = True
                start_normalization_thread(loading_data, save_path)
                print(f"Started normalization with {loaded} files")
                return norm_i  # Enable the interval
                
    return no_update

@callback(
    Output('file_norm_idx', 'children'),
    Output('file_norm_step', 'children'),
    Output('norm_info', 'data', allow_duplicate=True),
    #
    Input('norm_interval', 'n_intervals'),
    State('norm_info', 'data'),
    prevent_initial_call='initial_duplicate'
)
def poll_normalization_progress(_, norm_info):
    progress = get_normalization_progress()
    
    if not progress['started']:
        return no_update, no_update, no_update
    
    # Format the step display
    step_idx = f"{progress['current_step']}/8"
    step_text = f" {progress['step_description']}"
    
    # Update norm_info
    norm_i = norm_info if norm_info else get_base_info()
    norm_i['started'] = progress['started']
    norm_i['finished'] = progress['finished']
    
    if progress['finished']:
        if progress['error']:
            step_text = f"error: {progress['error']}"
        else:
            step_idx = "8/8"
            step_text = " normalisation terminée"
    
    return step_idx, step_text, norm_i

@callback(
    Output('url-p2', 'pathname'),
    #
    Input('norm_info', 'data'),
    State('url-p2', 'pathname'),
    prevent_initial_call=True
)
def redirect_once_done(norm_info, url_path):
    if norm_info == None or url_path != '/loading':
        return no_update

    loaded, total = len(loading_data), len(cache["paths"])
    
    if loaded == total and norm_info["finished"]:
        return '/analysis'

    return no_update